from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from picroute.crossing3.benchmark_source import (
    BENCHMARK_OVERRIDE_ROOT,
    discover_benchmarks,
)
from picroute.crossing3.backend import (
    _failure_directed_priority,
    _orthogonal_terminal_clearance_shifts,
    _pdk_hard_violation_count,
    _renderer_lateral_access_failures,
    _route_result_first_pass_order,
    _score,
    _terminal_radius_feedback_candidates,
    _terminal_lateral_correction_is_actionable,
)
from picroute.crossing3.access_contract import (
    effective_hard_access,
    port_connection_feasibility,
)
from picroute.crossing3.capacity_motifs import (
    add_capacity_recrossing_motifs,
    suppress_overpacked_capacity_components,
)
from picroute.crossing3.case_io import NetGeometry, Port, load_case, two_pin_nets
from picroute.crossing3.model import CrossingEvent, Prediction
from picroute.crossing3.finite_clearance_refiner import _clearance_center
from picroute.crossing3.predictor import predict_crossings
from picroute.crossing3.small_braid_refiner import closed_three_net_braids
from picroute.crossing3.routing_lattice import (
    center_is_track_compatible,
    snap_to_routing_track,
)
from picroute.crossing3.pdk_angle_audit import (
    audit_renderer_geometry,
    audit_route_geometry,
)
from picroute.crossing3.source_legalizer import legalize_source_unplaced_instances


class SourcePlacementLegalizerTests(unittest.TestCase):
    def test_mrr8_spreads_only_unplaced_port_access_groups(self) -> None:
        benchmark = discover_benchmarks()["mrr_weight_bank_8x8"]
        case = load_case(benchmark.path)
        fixed_before = {
            name: list(node["settings"]["placement"][1])
            for name, node in case["instances"].items()
            if str(node["settings"]["placement"][0]).upper() == "FIXED"
        }
        report = legalize_source_unplaced_instances(
            case,
            {
                "enabled": True,
                "minimum_forward_corridor_um": 30.0,
                "displacement_grid_um": 2.0,
            },
        )
        self.assertGreater(report["moved_instance_count"], 0)
        self.assertEqual(report["component_overlap_count"], 0)
        for name, lower_left in fixed_before.items():
            self.assertEqual(case["instances"][name]["settings"]["placement"][1], lower_left)
        for net_name in ("n_0", "n_1", "n_2", "n_3", "n_4", "n_5"):
            net = two_pin_nets(case)[net_name]
            self.assertGreaterEqual(abs(net.target.x - net.source.x), 30.0)

    def test_mrr8_balances_external_root_and_leaf_corridors(self) -> None:
        benchmark = discover_benchmarks()["mrr_weight_bank_8x8"]
        case = load_case(benchmark.path)
        report = legalize_source_unplaced_instances(
            case,
            {
                "enabled": True,
                "minimum_forward_corridor_um": 12.0,
                "displacement_grid_um": 2.0,
            },
        )
        component = report["axis_constraints"]["x"]["components"][0]
        self.assertGreaterEqual(component["minimum_boundary_corridor_um"], 29.0)
        movements = {
            item["instance"]: item["displacement_um"][0]
            for item in report["movements"]
        }
        self.assertLess(movements["fanout_yb_0_0"], -8.0)
        self.assertLess(abs(movements.get("fanout_yb_2_0", 0.0)), 2.0)

    def test_mrr8_reserves_more_space_for_near_collinear_branches(self) -> None:
        benchmark = discover_benchmarks()["mrr_weight_bank_8x8"]
        case = load_case(benchmark.path)
        report = legalize_source_unplaced_instances(
            case,
            {
                "enabled": True,
                "minimum_forward_corridor_um": 14.0,
                "collinear_minimum_forward_corridor_um": 18.0,
                "collinear_lateral_threshold_um": 30.0,
                "displacement_grid_um": 2.0,
            },
        )
        shifts = report["axis_constraints"]["x"]["group_shifts_um"]
        self.assertEqual(shifts["fanout_yb_0_0"], -24.0)
        self.assertEqual(shifts["fanout_yb_1_0"], -14.0)
        self.assertEqual(shifts["fanout_yb_2_0"], 0.0)
        legalized = two_pin_nets(case)
        self.assertGreaterEqual(abs(legalized["n_0"].vector[0]), 14.0)
        self.assertGreaterEqual(abs(legalized["n_3"].vector[0]), 18.0)

    def test_mmi8_has_no_matching_short_opposed_corridor(self) -> None:
        benchmark = discover_benchmarks()["multiportmmi_8x8"]
        case = load_case(benchmark.path)
        report = legalize_source_unplaced_instances(
            case,
            {
                "enabled": True,
                "minimum_forward_corridor_um": 30.0,
                "displacement_grid_um": 2.0,
            },
        )
        self.assertEqual(report["moved_instance_count"], 0)



class AccessContractTests(unittest.TestCase):
    def test_short_same_direction_connection_is_hard(self) -> None:
        audit = port_connection_feasibility(
            (0.0, 0.0),
            0.0,
            (8.0, 0.0),
            0.0,
            10.0,
            30.0,
            5.0,
        )
        self.assertEqual(
            audit["hard_failure_reason"],
            "parallel_same_direction_requires_local_uturn",
        )
        self.assertTrue(effective_hard_access(audit))

    def test_long_connection_keeps_local_escape_as_soft_signal(self) -> None:
        audit = port_connection_feasibility(
            (0.0, 0.0),
            180.0,
            (100.0, 0.0),
            0.0,
            10.0,
            30.0,
            5.0,
        )
        self.assertTrue(audit["hard_infeasible"])
        self.assertFalse(effective_hard_access(audit))

    def test_opposite_collinear_ports_are_feasible(self) -> None:
        audit = port_connection_feasibility(
            (0.0, 0.0),
            0.0,
            (20.0, 0.0),
            180.0,
            10.0,
            30.0,
            5.0,
        )
        self.assertFalse(effective_hard_access(audit))
        self.assertTrue(audit["direct_sbend_feasible"])
        self.assertTrue(audit["direct_straight_feasible"])

    def test_native_shortcut_may_not_create_free_angle_sbend(self) -> None:
        audit = port_connection_feasibility(
            (0.0, 0.0),
            0.0,
            (8.0, 1.0),
            180.0,
            10.0,
            30.0,
            5.0,
        )
        self.assertEqual(
            audit["hard_failure_reason"],
            "short_opposite_ports_require_forbidden_free_angle_sbend",
        )
        self.assertFalse(audit["direct_straight_feasible"])
        self.assertTrue(effective_hard_access(audit))

    def test_short_opposite_offset_needs_four_radius_octilinear_detour(self) -> None:
        audit = port_connection_feasibility(
            (0.0, 0.0),
            0.0,
            (16.0, 2.0),
            180.0,
            10.0,
            30.0,
            5.0,
        )
        self.assertEqual(audit["minimum_pdk_detour_um"], 20.0)
        self.assertTrue(audit["insufficient_pdk_detour"])
        self.assertEqual(
            audit["hard_failure_reason"],
            "short_opposite_ports_insufficient_pdk_octilinear_detour",
        )
        self.assertTrue(effective_hard_access(audit))

    def test_long_opposite_offset_can_use_octilinear_detour(self) -> None:
        audit = port_connection_feasibility(
            (0.0, 0.0),
            0.0,
            (28.0, 2.0),
            180.0,
            10.0,
            30.0,
            5.0,
        )
        self.assertFalse(audit["insufficient_pdk_detour"])
        self.assertFalse(effective_hard_access(audit))


class RoutingLatticeTests(unittest.TestCase):
    def test_native_cell_center_track_uses_half_grid_offset(self) -> None:
        self.assertEqual(snap_to_routing_track(10.0, 0.0, 2.0), 11.0)
        self.assertEqual(snap_to_routing_track(9.0, 0.0, 2.0), 9.0)
        self.assertEqual(snap_to_routing_track(110.0, 100.0, 2.0), 111.0)

    def test_both_crossing_coordinates_must_share_native_track_class(self) -> None:
        die = (0.0, 0.0, 100.0, 100.0)
        self.assertTrue(center_is_track_compatible((11.0, 13.0), die, 2.0))
        self.assertFalse(center_is_track_compatible((10.0, 13.0), die, 2.0))
        self.assertFalse(center_is_track_compatible((11.0, 12.0), die, 2.0))


class PdkAngleAuditTests(unittest.TestCase):
    def test_renderer_access_sbend_is_always_rejected(self) -> None:
        geometry = {
            "recoveries": [],
            "access_junctions": [
                {
                    "label": "n_7/0/start",
                    "method": "oriented_radius_checked_sbend",
                    "endpoint_error_um": 0.0,
                    "orientation_error_deg": 0.0,
                    "radius_verified": True,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "geometry.json"
            path.write_text(json.dumps(geometry), encoding="utf-8")
            audit = audit_renderer_geometry(path)
        self.assertEqual(audit["prohibited_access_sbend_count"], 1)
        self.assertEqual(audit["prohibited_access_sbends"][0]["net"], "n_7")
        self.assertFalse(audit["clean"])

    def test_coincident_renderer_join_is_clean(self) -> None:
        geometry = {
            "recoveries": [],
            "access_junctions": [
                {
                    "label": "n_7/0/start",
                    "method": "coincident_tangent_join",
                    "endpoint_error_um": 0.0002,
                    "orientation_error_deg": 0.0,
                    "radius_verified": True,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "geometry.json"
            path.write_text(json.dumps(geometry), encoding="utf-8")
            audit = audit_renderer_geometry(path)
        self.assertEqual(audit["prohibited_access_sbend_count"], 0)
        self.assertTrue(audit["clean"])

    def test_endpoint_tangent_match_does_not_hide_lateral_sbend(self) -> None:
        route = {
            "nets": {
                "n_0": {
                    "routed": True,
                    "short_sbend": False,
                    "paths": [
                        {
                            "points": [[12.0, 1.0], [20.0, 1.0]],
                            "start_port": {
                                "name": "xc__a__b,o3",
                                "center": [0.0, 0.0],
                                "orientation": 0.0,
                            },
                            "end_port": {
                                "name": "target,o1",
                                "center": [20.0, 1.0],
                                "orientation": 180.0,
                            },
                        }
                    ],
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "route.yml"
            path.write_text(yaml.safe_dump(route), encoding="utf-8")
            audit = audit_route_geometry(path)
        self.assertEqual(audit["access_lateral_offset_count"], 1)
        self.assertFalse(audit["clean"])

    def test_collinear_access_and_45_degree_turn_are_clean(self) -> None:
        route = {
            "nets": {
                "n_0": {
                    "routed": True,
                    "short_sbend": False,
                    "paths": [
                        {
                            "points": [[5.0, 0.0], [10.0, 0.0], [15.0, 5.0]],
                            "start_port": {
                                "name": "source,o1",
                                "center": [0.0, 0.0],
                                "orientation": 0.0,
                            },
                            "end_port": {
                                "name": "target,o1",
                                "center": [20.0, 10.0],
                                "orientation": 225.0,
                            },
                        }
                    ],
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "route.yml"
            path.write_text(yaml.safe_dump(route), encoding="utf-8")
            audit = audit_route_geometry(path)
        self.assertTrue(audit["clean"])


class CrossingTopologyTests(unittest.TestCase):
    def test_closed_three_net_braid_detection_requires_isolation(self) -> None:
        events = [
            CrossingEvent("xc__a__b", "a", "b", (0.0, 0.0), ["test"]),
            CrossingEvent("xc__a__c", "a", "c", (1.0, 0.0), ["test"]),
            CrossingEvent("xc__b__c", "b", "c", (2.0, 0.0), ["test"]),
        ]
        prediction = Prediction(
            events=events,
            net_orders={
                "a": ["xc__a__b", "xc__a__c"],
                "b": ["xc__a__b", "xc__b__c"],
                "c": ["xc__a__c", "xc__b__c"],
            },
            braid_components=[
                {
                    "parents": ["a", "b", "c"],
                    "crossing_order": [
                        "xc__a__b",
                        "xc__a__c",
                        "xc__b__c",
                    ],
                }
            ],
            parity_contract={},
            diagnostics=[],
        )
        self.assertEqual(
            closed_three_net_braids(prediction),
            [("xc__a__b", "xc__a__c", "xc__b__c")],
        )
        prediction.net_orders["a"].append("xc__a__outside")
        self.assertEqual(closed_three_net_braids(prediction), [])

    def test_capacity_completion_preserves_inherited_even_parity(self) -> None:
        def port(instance: str, x: float, y: float) -> Port:
            return Port(instance, "o1", x, y, 0.0, 0.5)

        nets = {
            "a": NetGeometry("a", "a0,o1", "a1,o1", port("a0", 0, 0), port("a1", 20, 0)),
            "b": NetGeometry("b", "b0,o1", "b1,o1", port("b0", 0, 2), port("b1", 20, 2)),
        }
        events = [
            CrossingEvent(
                f"xc__a__b__r{index}",
                "a",
                "b",
                (5.0 + 10.0 * index, 1.0),
                ["corridor_pressure_recrossing"],
                order_on_net_a=5 + index,
                order_on_net_b=7 + index,
            )
            for index in range(2)
        ]
        prediction = Prediction(
            events=events,
            net_orders={"a": [event.event_id for event in events], "b": [event.event_id for event in events]},
            braid_components=[],
            parity_contract={
                "a|b": {
                    "nets": ["a", "b"],
                    "required_parity": "even",
                    "predicted_count": 2,
                    "parity_satisfied": True,
                }
            },
            diagnostics=[],
        )
        add_capacity_recrossing_motifs(
            prediction,
            nets,
            {"minimum_stage_source_pairs": 8},
            bend_radius_um=5.0,
            crossing_body_um=8.0,
            minimum_access_um=10.0,
            grid_um=2.0,
        )
        self.assertEqual(prediction.parity_contract["a|b"]["required_parity"], "even")
        self.assertTrue(prediction.parity_contract["a|b"]["parity_satisfied"])
        self.assertEqual(
            [(event.order_on_net_a, event.order_on_net_b) for event in events],
            [(5, 7), (6, 8)],
        )

    def test_overpacked_connected_capacity_motifs_are_removed_as_a_unit(self) -> None:
        def port(instance: str, x: float, y: float) -> Port:
            return Port(instance, "o1", x, y, 0.0, 0.5)

        nets = {
            name: NetGeometry(
                name,
                f"{name}0,o1",
                f"{name}1,o1",
                port(f"{name}0", 0.0, index * 2.0),
                port(f"{name}1", 50.0, index * 2.0),
            )
            for index, name in enumerate(("a", "b", "c"))
        }
        events = [
            CrossingEvent(
                f"xc__a__{other}__r{index}",
                "a",
                other,
                (10.0 + 10.0 * index, 1.0),
                ["corridor_pressure_recrossing"],
                order_on_net_a=2 * (other == "c") + index,
                order_on_net_b=index,
            )
            for other in ("b", "c")
            for index in range(2)
        ]
        prediction = Prediction(
            events=events,
            net_orders={
                "a": [event.event_id for event in events],
                "b": [event.event_id for event in events if event.net_b == "b"],
                "c": [event.event_id for event in events if event.net_b == "c"],
            },
            braid_components=[],
            parity_contract={
                f"a|{other}": {
                    "nets": ["a", other],
                    "required_parity": "even",
                    "predicted_count": 2,
                    "parity_satisfied": True,
                }
                for other in ("b", "c")
            },
        )
        report = suppress_overpacked_capacity_components(
            prediction,
            nets,
            {
                "crossing_body_um": 8.0,
                "minimum_access_um": 10.0,
                "waveguide_spacing_um": 1.0,
            },
        )
        self.assertEqual(report["removed_event_count"], 4)
        self.assertEqual(prediction.events, [])
        self.assertTrue(
            all(
                contract["predicted_count"] == 0
                and contract["parity_satisfied"]
                for contract in prediction.parity_contract.values()
            )
        )

    def test_endpoint_pinched_optional_motif_is_suppressed(self) -> None:
        def port(instance: str, x: float, y: float) -> Port:
            return Port(instance, "o1", x, y, 0.0, 0.5)

        nets = {
            name: NetGeometry(
                name,
                f"{name}0,o1",
                f"{name}1,o1",
                port(f"{name}0", 0.0, index * 2.0),
                port(f"{name}1", 500.0, index * 2.0),
            )
            for index, name in enumerate(("a", "b"))
        }
        events = [
            CrossingEvent(
                f"xc__a__b__r{index}",
                "a",
                "b",
                point,
                ["corridor_pressure_recrossing"],
                order_on_net_a=index,
                order_on_net_b=index,
            )
            for index, point in enumerate(((5.0, 1.0), (200.0, 1.0)))
        ]
        prediction = Prediction(
            events=events,
            net_orders={"a": [event.event_id for event in events], "b": [event.event_id for event in events]},
            braid_components=[],
            parity_contract={
                "a|b": {
                    "nets": ["a", "b"],
                    "required_parity": "even",
                    "predicted_count": 2,
                    "parity_satisfied": True,
                }
            },
        )
        report = suppress_overpacked_capacity_components(
            prediction,
            nets,
            {
                "crossing_body_um": 8.0,
                "minimum_access_um": 10.0,
                "waveguide_spacing_um": 1.0,
            },
        )
        self.assertEqual(report["removed_event_count"], 2)
        self.assertTrue(report["components"][0]["pinched_endpoints"])

    def test_finite_clearance_anchor_moves_one_grid_cell_into_corridor(self) -> None:
        near = NetGeometry(
            "near",
            "source,o1",
            "target,o1",
            Port("source", "o1", 0.0, 10.0, 0.0, 0.5),
            Port("target", "o1", 100.0, 10.0, 180.0, 0.5),
        )
        self.assertEqual(
            _clearance_center((24.1, 13.5), near, -0.02, 2.0),
            (26.0, 10.0),
        )

    def test_proper_intersection_creates_one_odd_event(self) -> None:
        def port(instance: str, x: float, y: float) -> Port:
            return Port(instance, "o1", x, y, 0.0, 0.5)

        nets = {
            "a": NetGeometry("a", "a0,o1", "a1,o1", port("a0", 0, 0), port("a1", 20, 20)),
            "b": NetGeometry("b", "b0,o1", "b1,o1", port("b0", 0, 20), port("b1", 20, 0)),
        }
        prediction = predict_crossings(nets)
        self.assertEqual(len(prediction.events), 1)
        self.assertEqual(prediction.events[0].ideal_center_um, (10.0, 10.0))
        self.assertEqual(prediction.net_orders, {"a": ["xc__a__b"], "b": ["xc__a__b"]})
        self.assertTrue(prediction.parity_contract["a|b"]["parity_satisfied"])

    def test_original_benchmark_topology_regression(self) -> None:
        config = yaml.safe_load((PROJECT_ROOT / "configs" / "crossing3.yml").read_text(encoding="utf-8"))
        technology = config["technology"]
        expected = {
            "clements_8x8": 0,
            "clements_16x16": 2,
            "mrr_weight_bank_4x4": 3,
            "mrr_weight_bank_8x8": 7,
            "mrr_weight_bank_16x16": 15,
            "multiportmmi_8x8": 33,
            "multiportmmi_16x16": 63,
            "multiportmmi_32x32": 204,
        }
        actual = {}
        for name, benchmark in discover_benchmarks().items():
            nets = two_pin_nets(load_case(benchmark.path))
            prediction = predict_crossings(nets)
            base_event_ids = [event.event_id for event in prediction.events]
            add_capacity_recrossing_motifs(
                prediction,
                nets,
                config["prediction"],
                bend_radius_um=technology["bend_radius_um"],
                crossing_body_um=technology["crossing_body_um"],
                minimum_access_um=technology["minimum_access_um"],
                grid_um=technology["grid_um"],
            )
            self.assertEqual(
                [event.event_id for event in prediction.events[: len(base_event_ids)]],
                base_event_ids,
            )
            actual[name] = len(prediction.events)
            self.assertTrue(
                all(item["parity_satisfied"] for item in prediction.parity_contract.values())
            )
        self.assertEqual(actual, expected)

    def test_algorithm_source_has_no_case_dispatch(self) -> None:
        package = SOURCE_ROOT / "picroute" / "crossing3"
        text = "\n".join(path.read_text(encoding="utf-8") for path in package.glob("*.py"))
        for forbidden in (
            "clements_16x16",
            "multiportmmi_32x32",
            "mrr_weight_bank_4x4",
        ):
            self.assertNotIn(forbidden, text)


class FileManagementTests(unittest.TestCase):
    def test_authoritative_catalog_is_complete_with_explicit_overrides(self) -> None:
        catalog = discover_benchmarks()
        self.assertEqual(len(catalog), 8)
        self.assertNotIn("toy_example", catalog)
        override_root = BENCHMARK_OVERRIDE_ROOT.resolve()
        explicit_overrides = {
            path.resolve().stem: path.resolve()
            for path in override_root.glob("*.yml")
        }
        self.assertTrue(explicit_overrides)
        for name, benchmark in catalog.items():
            if name in explicit_overrides:
                self.assertEqual(benchmark.path, explicit_overrides[name])
            else:
                self.assertFalse(benchmark.path.is_relative_to(override_root))

    def test_generated_roots_are_ignored(self) -> None:
        ignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
        for entry in ("/work/", "/results/", "/logs/", "/artifacts/"):
            self.assertIn(entry, ignore)


class BackendFeedbackTests(unittest.TestCase):
    def test_radius_feedback_interleaves_crossings_after_joint_candidate(self) -> None:
        options = [
            ("xc_a", ((11.0, 13.0), "n_a0", 8.0)),
            ("xc_a", ((15.0, 13.0), "n_a1", 12.0)),
            ("xc_b", ((31.0, 33.0), "n_b0", 6.0)),
        ]
        with patch(
            "picroute.crossing3.backend._geometry_legal_after_centers",
            return_value=True,
        ):
            candidates = _terminal_radius_feedback_candidates(
                options, [], [], (0.0, 0.0, 100.0, 100.0), {}
            )
        self.assertEqual(
            [(kind, event_id) for kind, event_id, _state in candidates],
            [
                ("terminal_radius_cluster", "xc_a|xc_b"),
                ("terminal_radius", "xc_a"),
                ("terminal_radius", "xc_b"),
                ("terminal_radius", "xc_a"),
            ],
        )
        self.assertEqual(
            candidates[0][2],
            (("xc_a", options[0][1]), ("xc_b", options[2][1])),
        )

    def test_orthogonal_terminal_clearance_reserves_two_bend_spans(self) -> None:
        source = Port("fanout", "o2", 173.516, 751.875, 0.0, 0.5)
        crossing_top = Port("crossing", "o2", 185.0, 735.0, 90.0, 0.5)
        shifts = _orthogonal_terminal_clearance_shifts(
            source, crossing_top, minimum_access_um=10.0, grid_um=2.0
        )
        self.assertEqual(len(shifts), 2)
        self.assertAlmostEqual(shifts[0][0], 10.0)
        self.assertAlmostEqual(shifts[0][1], 0.0)
        self.assertAlmostEqual(shifts[1][0], 20.0)
        self.assertAlmostEqual(shifts[1][1], 0.0)

    def test_subgrid_pdk_correction_is_not_discarded(self) -> None:
        pdk_segments = {"n_pdk"}
        self.assertTrue(
            _terminal_lateral_correction_is_actionable(
                0.5, "n_pdk", pdk_segments, 2.0
            )
        )
        self.assertFalse(
            _terminal_lateral_correction_is_actionable(
                0.5, "n_route", pdk_segments, 2.0
            )
        )
        self.assertFalse(
            _terminal_lateral_correction_is_actionable(
                0.001, "n_pdk", pdk_segments, 2.0
            )
        )

    def test_pdk_violation_reduction_is_a_real_feedback_improvement(self) -> None:
        base = {
            "accepted": False,
            "missing_routes": 0,
            "abnormal_segments": [],
            "db_drc_violations": 0,
            "pdk_angle_violation_counts": {
                "access_lateral_offsets": 2,
                "off_grid_segments": 0,
                "prohibited_short_sbends": 0,
                "short_sbends": 4,
                "unsupported_recoveries": 0,
                "unsupported_turns": 0,
            },
        }
        repaired = {
            **base,
            "pdk_angle_violation_counts": {
                **base["pdk_angle_violation_counts"],
                "access_lateral_offsets": 1,
                # A collinear renderer shortcut is not a PDK defect.
                "short_sbends": 8,
            },
        }
        self.assertLess(_score(repaired), _score(base))
        self.assertEqual(_pdk_hard_violation_count(repaired), 1)

    def test_mfot_off_remains_a_string_in_production_config(self) -> None:
        config = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "crossing3.yml").read_text(encoding="utf-8")
        )
        self.assertEqual(config["feedback"]["mfot_mode"], "off")

    def test_first_pass_order_stream_parser_skips_later_iterations(self) -> None:
        payload = """schema: picdb_lidar_route_result
flow:
  - iteration: 1
    net: n_2
    processed_path_um:
      - [0, 0]
  - iteration: 1
    net: n_1
  - iteration: 2
    net: n_3
instances: {}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "route.yml"
            path.write_text(payload, encoding="utf-8")
            self.assertEqual(_route_result_first_pass_order(path), ["n_2", "n_1"])

    def test_failure_priority_repairs_only_direct_blocker_precedence(self) -> None:
        priority = _failure_directed_priority(
            {
                "native_first_pass_order": ["n_0", "n_1", "n_2", "n_3"],
                "abnormal_segments": ["n_3", "n_1"],
                "native_failure_evidence": {
                    "start_blockers": {"n_1": ["n_0"], "n_3": ["n_2"]}
                },
            }
        )
        self.assertEqual(priority, ("n_1", "n_0", "n_3", "n_2"))

    def test_failure_priority_completes_native_trace_to_design_permutation(self) -> None:
        priority = _failure_directed_priority(
            {
                "native_first_pass_order": ["n_0", "n_2", "n_3"],
                "abnormal_segments": ["n_3"],
                "native_failure_evidence": {
                    "start_blockers": {"n_3": ["n_2"]}
                },
            },
            ["n_0", "n_1", "n_2", "n_3"],
        )
        self.assertEqual(priority, ("n_0", "n_3", "n_2", "n_1"))
        self.assertEqual(set(priority), {"n_0", "n_1", "n_2", "n_3"})

    def test_renderer_lateral_failure_recovers_segment_evidence(self) -> None:
        payload = (
            "ValueError: radius-correct render failed for n_51__s002/0: "
            "PDK forbids lateral access S-bend: "
            "longitudinal=18.500000um, lateral=-1.000000um\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "renderer.log"
            path.write_text(payload, encoding="utf-8")
            self.assertEqual(
                _renderer_lateral_access_failures(path),
                [
                    {
                        "net": "n_51__s002",
                        "longitudinal_um": 18.5,
                        "lateral_um": -1.0,
                        "reason": "pdk_forbids_lateral_access_sbend",
                    }
                ],
            )


if __name__ == "__main__":
    unittest.main()
