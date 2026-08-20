"""Single source of truth for the crossing3 PDK angle contract."""

ANGLE_TOLERANCE_DEG = 0.5

# The final GDS database unit is 1 nm. Sub-DBU decimal serialization residue
# cannot produce a resolvable bend; any larger transverse offset is forbidden.
ACCESS_LATERAL_TOLERANCE_UM = 1e-3
