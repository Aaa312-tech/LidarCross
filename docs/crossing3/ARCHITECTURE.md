# HCTP architecture and modification boundary

HCTP (Hierarchical Crossing-Topology Placement) is a crossing-only frontend.
It may change crossing count, parent-net pairing, immutable per-net event
order, PCell orientation/arm mapping, and crossing placement.  Original device
placement and every detailed-routing stage remain frozen.

## Stages

1. Read and hash the authoritative DATE27 benchmark without modifying it.
2. Freeze all supplied original device placements and extract absolute ports.
3. Build coarse free-space/corridor guides used only for crossing prediction.
4. Build a parity-constrained crossing graph.  A pair has
   `count = inversion_bit + 2 * recrossing_motifs`.
5. Decompose permutation components into adjacent-swap wiring-diagram stages.
6. Enumerate exact PCell position, rotation, arm-mapping, and direction states.
7. Solve each crossing component with hard geometry/topology constraints and a
   lexicographic physical objective.
8. Materialize fixed crossing PCells and split parent nets in immutable order.
9. Invoke the frozen strict-preplaced router as a black box.
10. Convert failures only into crossing-frontend no-good or motif constraints.

## Forbidden adaptations

HCTP must never select behavior by case name, net name, saved coordinate,
historical placement, or benchmark-size identity.  It must not change native
route priority, A* costs/neighbors, crossing legality, DRC, post-processing, or
GDS geometry checks.

## Acceptance

A result is accepted only when source/tool hashes are recorded, original
instances are unchanged, every topology/parity audit passes, the strict router
reports no missing routes or implicit crossings, DB DRC is clean, and the final
GDS passes endpoint position/tangent/radius and route-cell continuity checks.

