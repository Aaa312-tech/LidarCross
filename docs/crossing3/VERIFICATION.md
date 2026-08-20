# Verification record (2026-08-18)

## Frontend

The clean all-case run `frontend-all-hctp-v4` completed every authoritative
DATE27 benchmark under the final recorded frontend/config hashes.

| Case | Crossings | Materialized route nets | Frontend |
| --- | ---: | ---: | --- |
| clements_8x8 | 0 | 79 | PASS |
| clements_16x16 | 2 | 291 | PASS |
| mrr_weight_bank_4x4 | 3 | 36 | PASS |
| mrr_weight_bank_8x8 | 7 | 108 | PASS |
| mrr_weight_bank_16x16 | 15 | 348 | PASS |
| multiportmmi_8x8 | 33 | 177 | PASS |
| multiportmmi_16x16 | 63 | 349 | PASS |
| multiportmmi_32x32 | 125 | 697 | PASS |
| toy_example | 0 | 2 | PASS |

The channel guide reported fallback straight guides for 14 Clements16 nets
and 60 MMI32 nets.  This is recorded diagnostic state: crossing geometry was
still independently legalized and audited, but those fallback guides are not
evidence of detailed route feasibility.

## Frozen backend

- `strict-c8-hctp-v3`: **ACCEPTED** on the baseline attempt under the final
  recorded source/config hashes.  The frozen
  strict router routed 79/79 nets, DB DRC was clean, the strict renderer
  produced 79 route cells with 158 valid access waveguides, and continuity
  reported zero disconnected cells and zero endpoint tangent violations.
- `strict-mrr4-hctp-v3`: **STRICT_BACKEND_FAIL** after the configured 10
  attempts.  Failure-driven center/orientation no-goods reduced missing routes
  from 8 to 5, but did not reach acceptance.  No MRR4 result was published.

These two statuses must remain distinct.  The current implementation proves
the crossing topology/placement frontend runs across all cases and that the
frozen backend interface can produce an independently accepted result; it
does not claim strict detailed-route closure for every crossing-bearing case.
