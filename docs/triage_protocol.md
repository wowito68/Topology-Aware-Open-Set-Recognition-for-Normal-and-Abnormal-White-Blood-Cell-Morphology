# Delivery 1 Triage Protocol

Each detected cell receives exactly one primary status:

- `AUTO_ACCEPT`;
- `HUMAN_REVIEW`;
- `POSSIBLE_UNKNOWN`;
- `LOW_QUALITY`;
- `INVALID_DETECTION`.

The rule priority is fixed:

1. invalid detection or invalid crop -> `INVALID_DETECTION`;
2. deterministic QC failure -> `LOW_QUALITY`;
3. MSP unknown score above threshold -> `POSSIBLE_UNKNOWN`;
4. low classifier confidence -> `HUMAN_REVIEW`;
5. low top1-top2 margin -> `HUMAN_REVIEW`;
6. otherwise -> `AUTO_ACCEPT`.

The primary open-set score is MSP unknownness:

`unknown_score = 1 - max softmax probability`.

Larger values always mean more unknown-like.

Final triage thresholds are pending calibration on allowed validation data only.
The final test split must not be used for threshold tuning.
