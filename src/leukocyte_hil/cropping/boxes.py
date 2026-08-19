"""Bounding-box clipping, expansion, and crop checksums."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO

from PIL import Image


@dataclass(frozen=True)
class BoundingBox:
    """Pixel-space box in ``x_min, y_min, x_max, y_max`` convention."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def is_valid(self) -> bool:
        return self.width > 0.0 and self.height > 0.0

    def to_xyxy(self) -> list[float]:
        return [self.x_min, self.y_min, self.x_max, self.y_max]

    def to_int_tuple(self) -> tuple[int, int, int, int]:
        return (
            int(round(self.x_min)),
            int(round(self.y_min)),
            int(round(self.x_max)),
            int(round(self.y_max)),
        )


def clip_bbox(bbox: BoundingBox, image_size: tuple[int, int]) -> BoundingBox:
    """Clip a box to image bounds."""

    width, height = image_size
    return BoundingBox(
        x_min=min(max(0.0, bbox.x_min), float(width)),
        y_min=min(max(0.0, bbox.y_min), float(height)),
        x_max=min(max(0.0, bbox.x_max), float(width)),
        y_max=min(max(0.0, bbox.y_max), float(height)),
    )


def expand_bbox(
    bbox: BoundingBox, image_size: tuple[int, int], margin_fraction: float
) -> BoundingBox:
    """Expand a box by a fixed context margin and clip it to the image."""

    if margin_fraction < 0.0:
        msg = "margin_fraction must be non-negative"
        raise ValueError(msg)
    delta_x = bbox.width * margin_fraction
    delta_y = bbox.height * margin_fraction
    expanded = BoundingBox(
        x_min=bbox.x_min - delta_x,
        y_min=bbox.y_min - delta_y,
        x_max=bbox.x_max + delta_x,
        y_max=bbox.y_max + delta_y,
    )
    return clip_bbox(expanded, image_size)


def crop_image(image: Image.Image, bbox: BoundingBox) -> Image.Image:
    """Crop an image using a valid pixel-space bounding box."""

    if not bbox.is_valid:
        msg = f"Cannot crop invalid bounding box: {bbox}"
        raise ValueError(msg)
    return image.crop(bbox.to_int_tuple()).convert("RGB")


def crop_sha256(crop: Image.Image) -> str:
    """Return a stable SHA256 checksum for a crop encoded as PNG."""

    buffer = BytesIO()
    crop.save(buffer, format="PNG")
    return sha256(buffer.getvalue()).hexdigest()


def touches_image_border(bbox: BoundingBox, image_size: tuple[int, int]) -> bool:
    """Return whether a box touches any image edge after clipping."""

    width, height = image_size
    return bbox.x_min <= 0 or bbox.y_min <= 0 or bbox.x_max >= width or bbox.y_max >= height
