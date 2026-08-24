# Delivery 1.1 Detector Status

DETECTOR TRAINING STATUS:
INCOMPLETE — NUMERICAL FAILURE IN EPOCH 3

VALID CHECKPOINTS:
EPOCH 1 AND EPOCH 2

FINAL DETECTOR EVALUATION:
NOT PERFORMED

END-TO-END SYSTEM:
NOT EVALUATED

The Raabin-WBC Faster R-CNN detector training run did not complete the frozen
30-epoch protocol. It produced valid finite checkpoints for epochs 1 and 2, then
aborted in epoch 3 batch 298 with non-finite forward losses.

The epoch 1 and epoch 2 validation metrics are useful only as validation-run
diagnostics from an incomplete training protocol. They must not be reported as
detector test results, final detector performance, crop-domain performance, or
end-to-end system performance.

The best available checkpoint by validation recall is epoch 1, but it is not a
final detector checkpoint because the planned 30-epoch protocol was incomplete.

See `artifacts/diagnostics/detector_final_run/failure_report.md` for the
preserved failure report and reproducibility inventory.
