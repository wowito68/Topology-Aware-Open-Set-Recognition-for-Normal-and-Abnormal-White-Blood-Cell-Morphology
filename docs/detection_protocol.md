# Delivery 1 Detection Protocol

The Delivery 1 detector is a single-class `LEUKOCYTE_CANDIDATE` detector.
It localizes candidate WBC regions and intentionally does not perform
morphology classification.

Primary dataset: Raabin-WBC, pending local archive verification.

Initial detector family: torchvision Faster R-CNN. This choice avoids adding a
new detector dependency before the dataset/split/leakage gate is complete.

Seed: 37.

Primary validation objective: WBC candidate recall. Missing a leukocyte prevents
all downstream classifier and review logic from seeing that cell.

Test data must not be used for threshold selection, detector selection,
calibration, or preprocessing decisions.

Before detector training:

- complete dataset archive verification;
- build `data/manifests/delivery1_detection_manifest.csv`;
- freeze train/validation/test splits;
- audit exact duplicate leakage;
- run a one-epoch detector smoke test.
