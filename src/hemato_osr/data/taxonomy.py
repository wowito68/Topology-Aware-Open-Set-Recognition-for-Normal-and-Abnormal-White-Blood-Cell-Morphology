"""Central taxonomy for known and unknown white blood cell morphologies."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

KNOWN = "known"
UNKNOWN = "unknown"

DEFAULT_KNOWN_CLASSES: tuple[str, ...] = (
    "basophil",
    "eosinophil",
    "lymphocyte",
    "monocyte",
    "neutrophil_segmented",
)

DEFAULT_UNKNOWN_CLASSES: tuple[str, ...] = (
    "myeloblast",
    "promyelocyte",
    "promyelocyte_atypical",
    "myelocyte",
    "metamyelocyte",
    "hairy_cell",
    "lymphocyte_neoplastic",
    "lymphocyte_large_granular",
    "lymphocyte_reactive",
    "plasma_cell",
    "smudge_cell",
    "normoblast",
    "neutrophil_band",
)

DEFAULT_ALIASES: dict[str, str] = {
    "segmented_neutrophil": "neutrophil_segmented",
    "neutrophil": "neutrophil_segmented",
    "band_neutrophil": "neutrophil_band",
    "large_granular_lymphocyte": "lymphocyte_large_granular",
    "reactive_lymphocyte": "lymphocyte_reactive",
    "neoplastic_lymphocyte": "lymphocyte_neoplastic",
}


def normalize_label(label: str) -> str:
    """Normalize raw class labels into stable lowercase snake-case keys."""

    normalized = label.strip().lower()
    for char in (" ", "-", "/", "\\", "."):
        normalized = normalized.replace(char, "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


@dataclass(frozen=True)
class Taxonomy:
    """Map raw dataset labels to canonical labels and known/unknown status."""

    known_classes: tuple[str, ...] = DEFAULT_KNOWN_CLASSES
    unknown_classes: tuple[str, ...] = DEFAULT_UNKNOWN_CLASSES
    aliases: Mapping[str, str] = field(default_factory=lambda: DEFAULT_ALIASES)

    def __post_init__(self) -> None:
        known = tuple(normalize_label(label) for label in self.known_classes)
        unknown = tuple(normalize_label(label) for label in self.unknown_classes)
        overlap = set(known).intersection(unknown)
        if overlap:
            msg = f"Classes cannot be both known and unknown: {sorted(overlap)}"
            raise ValueError(msg)
        object.__setattr__(self, "known_classes", known)
        object.__setattr__(self, "unknown_classes", unknown)

    @property
    def canonical_classes(self) -> tuple[str, ...]:
        """All labels explicitly represented by this taxonomy."""

        return self.known_classes + self.unknown_classes

    def canonicalize(self, raw_label: str) -> str:
        """Return the canonical morphology for a raw dataset label."""

        normalized = normalize_label(raw_label)
        return normalize_label(self.aliases.get(normalized, normalized))

    def known_status(self, raw_label: str) -> str:
        """Return ``known`` or ``unknown`` for a raw or canonical label."""

        canonical = self.canonicalize(raw_label)
        if canonical in self.known_classes:
            return KNOWN
        return UNKNOWN

    def validate_known_label(self, label: str) -> None:
        """Raise if ``label`` is not one of the configured known classes."""

        canonical = self.canonicalize(label)
        if canonical not in self.known_classes:
            msg = f"{label!r} is not configured as a known class"
            raise ValueError(msg)

    @classmethod
    def from_config(cls, config: Mapping[str, object] | None) -> Taxonomy:
        """Create a taxonomy from an OmegaConf/plain mapping."""

        if config is None:
            return cls()
        known = _string_tuple(config.get("known_classes"), DEFAULT_KNOWN_CLASSES)
        unknown = _string_tuple(config.get("unknown_classes"), DEFAULT_UNKNOWN_CLASSES)
        aliases_obj = config.get("aliases", DEFAULT_ALIASES)
        aliases = (
            {str(k): str(v) for k, v in aliases_obj.items()}
            if isinstance(aliases_obj, Mapping)
            else DEFAULT_ALIASES
        )
        return cls(known_classes=known, unknown_classes=unknown, aliases=aliases)


def _string_tuple(value: object, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value)
    return default
