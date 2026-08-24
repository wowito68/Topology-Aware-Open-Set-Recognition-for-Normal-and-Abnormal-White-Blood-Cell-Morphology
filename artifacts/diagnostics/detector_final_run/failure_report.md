# Delivery 1.1 Detector Final Run Failure Report

DETECTOR TRAINING STATUS:
INCOMPLETE — NUMERICAL FAILURE IN EPOCH 3

VALID CHECKPOINTS:
EPOCH 1 AND EPOCH 2

FINAL DETECTOR EVALUATION:
NOT PERFORMED

END-TO-END SYSTEM:
NOT EVALUATED

## Summary

The final Raabin-WBC Faster R-CNN detector run started on EC2 at
`2026-08-23T04:31:37Z` under the frozen Delivery 1.1 protocol. It completed
epochs 1 and 2 with finite checkpoints, then aborted during epoch 3 at batch
298 due to non-finite forward losses.

The saved checkpoints are numerically valid, but the 30-epoch protocol is
incomplete. These validation metrics must not be reported as detector test
performance and must not be used as final end-to-end evidence.

## Failure

- phase: `FORWARD_LOSS`
- epoch: `3`
- batch index: `298`
- loss_classifier: `NaN`
- loss_box_reg: `NaN`
- loss_objectness: `NaN`
- loss_rpn_box_reg: `Infinity`
- total loss: `NaN`
- root cause: `NO AISLADA`

## Valid Epochs

Epoch 1:

- recall: `0.9867409`
- mAP50: `0.9483276`
- mAP50:95: `0.8735602`
- checkpoint non-finite tensors: `0`
- checkpoint non-finite values: `0`

Epoch 2:

- recall: `0.985680`
- mAP50: `0.945240`
- mAP50:95: `0.745603`
- checkpoint non-finite tensors: `0`
- checkpoint non-finite values: `0`

Best available checkpoint by validation recall:

- `best_checkpoint.pt`
- epoch: `1`
- validation recall: `0.986740917528507`

This is not a final detector checkpoint under the declared 30-epoch protocol.

## Reproducibility

- run code commit: `7478155`
- seed: `37`
- architecture: `torchvision Faster R-CNN`
- task: single foreground class `LEUKOCYTE_CANDIDATE`
- foreground label: `1`
- background label: `0`
- optimizer: `SGD + momentum`
- learning rate: `0.005`
- batch size: `2`
- planned epochs: `30`
- AMP: `false`
- split: Delivery 1.1 Raabin Film-ID split in `data/manifests/delivery11_raabin_detection.csv`
- selection metric: validation WBC candidate recall
- Python: `3.13.15`
- PyTorch: `2.13.0+cu130`
- torchvision: `0.28.0+cu130`
- CUDA: `13.0`
- GPU: `NVIDIA A10G`
- EC2: `g5.xlarge`, `i-0efafee8f1fda6af7`, `us-east-1c`
- remote repo: `/home/ubuntu/hemato-osr`

Dataset archive:

- remote path: `/opt/dlami/nvme/datasets/raabin_wbc/raw/WBCData.rar`
- size: `55781603692`
- SHA256: `23473a5444d1265777805c4fd395ef3dabcd6c4fdfbe9c6898146ae41543376e`

## Preservation Policy

The following large artifacts were intentionally not copied locally:

- Raabin-WBC raw archive
- extracted Raabin-WBC images
- generated crops or caches
- detector checkpoints

The local repository preserves compact metrics, diagnostics, manifests, audits,
configuration files, logs, and hashes needed to reproduce or audit the run.
