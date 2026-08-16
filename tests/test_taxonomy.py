from __future__ import annotations

import pytest

from hemato_osr.data.taxonomy import KNOWN, UNKNOWN, Taxonomy


def test_taxonomy_maps_known_and_unknown() -> None:
    taxonomy = Taxonomy()

    assert taxonomy.canonicalize("Segmented Neutrophil") == "neutrophil_segmented"
    assert taxonomy.known_status("basophil") == KNOWN
    assert taxonomy.known_status("myeloblast") == UNKNOWN


def test_taxonomy_rejects_known_unknown_overlap() -> None:
    with pytest.raises(ValueError, match="both known and unknown"):
        Taxonomy(known_classes=("basophil",), unknown_classes=("basophil",))
