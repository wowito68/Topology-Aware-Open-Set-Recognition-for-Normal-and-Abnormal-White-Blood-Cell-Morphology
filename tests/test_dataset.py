from __future__ import annotations

from hemato_osr.data.dataset import ManifestImageDataset, make_label_mapping
from hemato_osr.data.taxonomy import DEFAULT_KNOWN_CLASSES
from hemato_osr.data.transforms import TransformConfig, build_transforms


def test_manifest_dataset_loads_tensor(synthetic_manifest) -> None:  # type: ignore[no-untyped-def]
    dataset = ManifestImageDataset(
        synthetic_manifest,
        label_to_index=make_label_mapping(DEFAULT_KNOWN_CLASSES),
        transform=build_transforms("test", TransformConfig(image_size=32)),
        split="train",
        include_unknown=False,
    )

    sample = dataset[0]

    assert tuple(sample.image.shape) == (3, 32, 32)
    assert sample.label >= 0
    assert sample.known_status == "known"
