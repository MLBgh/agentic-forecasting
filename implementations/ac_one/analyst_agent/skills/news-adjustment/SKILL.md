---
name: news-adjustment
description: Conservatively adjust an external airline fuel-consumption forecast using cutoff-safe public evidence.
---

# News adjustment

Use this workflow only when asked to adjust the supplied external forecast.

1. Anchor on the external forecast. It already represents structured
   operational features; news is an overlay, not a replacement model.
2. Search with the payload's exact `as_of` date as `cutoff_date`.
3. Look for causal links to physical consumption:
   - capacity, schedules, passenger/cargo demand, cancellations;
   - airspace closures, rerouting, strikes, or operational disruption;
   - severe regional weather;
   - efficiency or fleet-wide policy changes;
   - energy-market developments only when they plausibly alter operations.
4. Do not infer a city or airport from an anonymized station. Regional evidence
   must remain regional and should receive less weight than identified,
   directly relevant operational evidence.
5. Triangulate material claims where possible. Prefer official aviation,
   government, airport/airline, and major wire-service sources.
6. If evidence is weak, contradictory, or not clearly incremental to normal
   operating patterns, use a zero adjustment.
7. Keep any non-zero adjustment proportional and within the supplied cap.
8. Explain direction, magnitude, causal pathway, uncertainty, and source URLs.

Never use the realised `actual_*` values, post-cutoff news, or knowledge of later
outcomes. Do not convert between minmax and indexed forecast scales. If search
verification fails, retain the baseline and disclose that no safe adjustment was
available.
