"""PyTorch dataset for the EMOTIC CVPR and PAMI MATLAB annotations.

The dataset unit is one target person. An image containing multiple annotated
people therefore produces multiple samples. Images are resolved from one or
more roots, each of which must contain folders such as ``mscoco/images`` and
``framesdb/images``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from torch import Tensor
from torch.utils.data import Dataset


EMOTION_LABELS: tuple[str, ...] = (
    "Affection",
    "Anger",
    "Annoyance",
    "Anticipation",
    "Aversion",
    "Confidence",
    "Disapproval",
    "Disconnection",
    "Disquietment",
    "Doubt/Confusion",
    "Embarrassment",
    "Engagement",
    "Esteem",
    "Excitement",
    "Fatigue",
    "Fear",
    "Happiness",
    "Pain",
    "Peace",
    "Pleasure",
    "Sadness",
    "Sensitivity",
    "Suffering",
    "Surprise",
    "Sympathy",
    "Yearning",
)

LABEL_TO_INDEX = {label: index for index, label in enumerate(EMOTION_LABELS)}
VAD_KEYS: tuple[str, str, str] = ("valence", "arousal", "dominance")

CategoryPolicy = Literal["union", "majority"]
ImageTransform = Callable[[Image.Image], Any]


@dataclass(frozen=True)
class _PersonSample:
    sample_id: str
    image_path: Path
    relative_image_path: str
    original_database: str
    bbox: tuple[float, float, float, float]
    category_annotations: tuple[tuple[str, ...], ...]
    vad_annotations: tuple[tuple[float, float, float], ...]
    gender: str
    age: str


def _as_list(value: Any) -> list[Any]:
    """Normalize scalar MATLAB cells and NumPy arrays to a Python list."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _as_text(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


class EmoticDataset(Dataset[dict[str, Any]]):
    """Load EMOTIC samples from an official ``.mat`` annotation file.

    Args:
        annotation_path: Path to ``Annotations.mat`` (PAMI) or
            ``CVPR17_Annotations.mat``.
        image_roots: One or more directories containing EMOTIC source folders.
            For example, a root may contain ``mscoco/images`` and
            ``ade20k/images``.
        split: ``train``, ``val``/``validation``, or ``test``.
        context_transform: Transform applied to the full RGB image.
        person_transform: Transform applied to the target-person crop.
        category_policy: ``union`` marks a class positive if any annotator
            selected it. ``majority`` requires more than half of categorical
            annotators to select it.
        crop_padding: Fractional padding added around each side of the body box.
            For example, ``0.05`` adds five percent of box width and height.
        strict_images: Raise an error when any referenced image is unavailable.
            When false, samples whose images are missing are skipped.

    The returned dictionary contains fixed-size targets suitable for PyTorch's
    default collate function. Without transforms, the two image values are PIL
    images and callers should use a custom collate function or inspect samples
    one at a time.
    """

    class_names = EMOTION_LABELS
    num_classes = len(EMOTION_LABELS)

    def __init__(
        self,
        annotation_path: str | Path,
        image_roots: str | Path | Sequence[str | Path],
        split: str = "train",
        context_transform: ImageTransform | None = None,
        person_transform: ImageTransform | None = None,
        category_policy: CategoryPolicy = "union",
        crop_padding: float = 0.0,
        strict_images: bool = True,
    ) -> None:
        self.annotation_path = Path(annotation_path).expanduser().resolve()
        self.image_roots = self._normalize_roots(image_roots)
        self.split = self._normalize_split(split)
        self.context_transform = context_transform
        self.person_transform = person_transform
        self.category_policy = category_policy
        self.crop_padding = float(crop_padding)
        self.strict_images = strict_images

        if category_policy not in {"union", "majority"}:
            raise ValueError("category_policy must be 'union' or 'majority'")
        if self.crop_padding < 0:
            raise ValueError("crop_padding must be non-negative")
        if not self.annotation_path.is_file():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_path}")
        missing_roots = [root for root in self.image_roots if not root.is_dir()]
        if missing_roots:
            formatted = "\n".join(f"  - {path}" for path in missing_roots)
            raise FileNotFoundError(f"Image root(s) not found:\n{formatted}")

        self._image_cache: dict[str, Path | None] = {}
        self.samples, self.missing_images = self._build_index()

        if self.missing_images and self.strict_images:
            examples = "\n".join(
                f"  - {path}" for path in self.missing_images[:10]
            )
            suffix = "" if len(self.missing_images) <= 10 else "\n  - ..."
            raise FileNotFoundError(
                f"{len(self.missing_images)} unique image(s) referenced by the "
                f"'{self.split}' split were not found under any image root:\n"
                f"{examples}{suffix}"
            )

        if not self.samples:
            raise RuntimeError(
                f"No usable person samples found in split '{self.split}' of "
                f"{self.annotation_path}"
            )

    @staticmethod
    def _normalize_split(split: str) -> str:
        normalized = split.strip().lower()
        aliases = {"validation": "val", "valid": "val"}
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val'/'validation', or 'test'")
        return normalized

    @staticmethod
    def _normalize_roots(
        image_roots: str | Path | Sequence[str | Path],
    ) -> tuple[Path, ...]:
        if isinstance(image_roots, (str, Path)):
            roots = [image_roots]
        else:
            roots = list(image_roots)
        if not roots:
            raise ValueError("image_roots must contain at least one directory")
        return tuple(Path(root).expanduser().resolve() for root in roots)

    def _resolve_image(self, relative_path: Path) -> Path | None:
        key = relative_path.as_posix()
        if key not in self._image_cache:
            self._image_cache[key] = next(
                (
                    candidate
                    for root in self.image_roots
                    if (candidate := root / relative_path).is_file()
                ),
                None,
            )
        return self._image_cache[key]

    def _build_index(self) -> tuple[list[_PersonSample], list[str]]:
        annotations = loadmat(self.annotation_path, simplify_cells=True)
        if self.split not in annotations:
            available = sorted(key for key in annotations if not key.startswith("__"))
            raise KeyError(
                f"Split '{self.split}' is absent from {self.annotation_path}. "
                f"Available variables: {available}"
            )

        records = _as_list(annotations[self.split])
        samples: list[_PersonSample] = []
        missing: set[str] = set()

        for image_index, record in enumerate(records):
            if not isinstance(record, dict):
                raise TypeError(
                    f"Unexpected annotation record at index {image_index}: "
                    f"{type(record).__name__}"
                )

            relative_path = Path(_as_text(record.get("folder"), "")) / _as_text(
                record.get("filename"), ""
            )
            image_path = self._resolve_image(relative_path)
            if image_path is None:
                missing.add(relative_path.as_posix())
                if not self.strict_images:
                    continue

            people = _as_list(record.get("person"))
            for person_index, person in enumerate(people):
                if not isinstance(person, dict):
                    raise TypeError(
                        f"Unexpected person record at image {image_index}, "
                        f"person {person_index}: {type(person).__name__}"
                    )

                bbox_values = np.asarray(person.get("body_bbox"), dtype=np.float32)
                bbox_values = bbox_values.reshape(-1)
                if bbox_values.size != 4 or not np.isfinite(bbox_values).all():
                    raise ValueError(
                        f"Invalid body_bbox at image {image_index}, person "
                        f"{person_index}: {bbox_values!r}"
                    )

                category_annotations = self._read_category_annotations(person)
                vad_annotations = self._read_vad_annotations(person)
                relative_text = relative_path.as_posix()
                samples.append(
                    _PersonSample(
                        sample_id=f"{self.split}:{image_index}:{person_index}",
                        image_path=image_path or Path(relative_text),
                        relative_image_path=relative_text,
                        original_database=_as_text(record.get("original_database")),
                        bbox=tuple(float(value) for value in bbox_values),
                        category_annotations=category_annotations,
                        vad_annotations=vad_annotations,
                        gender=_as_text(person.get("gender")),
                        age=_as_text(person.get("age")),
                    )
                )

        return samples, sorted(missing)

    @staticmethod
    def _read_category_annotations(
        person: dict[str, Any],
    ) -> tuple[tuple[str, ...], ...]:
        output: list[tuple[str, ...]] = []
        for annotation in _as_list(person.get("annotations_categories")):
            if not isinstance(annotation, dict):
                continue
            labels = tuple(
                _as_text(label)
                for label in _as_list(annotation.get("categories"))
                if _as_text(label) != "unknown"
            )
            unknown = sorted(set(labels) - LABEL_TO_INDEX.keys())
            if unknown:
                raise ValueError(f"Unknown EMOTIC emotion label(s): {unknown}")
            output.append(labels)
        return tuple(output)

    @staticmethod
    def _read_vad_annotations(
        person: dict[str, Any],
    ) -> tuple[tuple[float, float, float], ...]:
        output: list[tuple[float, float, float]] = []
        for annotation in _as_list(person.get("annotations_continuous")):
            if not isinstance(annotation, dict):
                continue
            values = tuple(float(annotation[key]) for key in VAD_KEYS)
            if not np.isfinite(values).all():
                continue
            output.append(values)
        return tuple(output)

    def _category_target(self, sample: _PersonSample) -> Tensor:
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        if not sample.category_annotations:
            return target

        counts = Counter(
            label
            for annotation in sample.category_annotations
            for label in set(annotation)
        )
        if self.category_policy == "union":
            selected = counts.keys()
        else:
            threshold = len(sample.category_annotations) // 2 + 1
            selected = (label for label, count in counts.items() if count >= threshold)

        for label in selected:
            target[LABEL_TO_INDEX[label]] = 1.0
        return target

    @staticmethod
    def _vad_target(sample: _PersonSample) -> tuple[Tensor, Tensor]:
        if not sample.vad_annotations:
            return (
                torch.zeros(3, dtype=torch.float32),
                torch.zeros(3, dtype=torch.bool),
            )
        values = torch.tensor(sample.vad_annotations, dtype=torch.float32)
        return values.mean(dim=0), torch.ones(3, dtype=torch.bool)

    def _clamp_bbox(
        self,
        bbox: tuple[float, float, float, float],
        image_width: int,
        image_height: int,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        if self.crop_padding:
            pad_x = (x2 - x1) * self.crop_padding
            pad_y = (y2 - y1) * self.crop_padding
            x1, y1, x2, y2 = x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y

        x1 = max(0, min(int(np.floor(x1)), image_width))
        y1 = max(0, min(int(np.floor(y1)), image_height))
        x2 = max(0, min(int(np.ceil(x2)), image_width))
        y2 = max(0, min(int(np.ceil(y2)), image_height))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Bounding box becomes empty after clamping: "
                f"{(x1, y1, x2, y2)} for image size "
                f"{(image_width, image_height)}"
            )
        return x1, y1, x2, y2

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as source:
            context_image = source.convert("RGB")

        bbox = self._clamp_bbox(
            sample.bbox,
            image_width=context_image.width,
            image_height=context_image.height,
        )
        person_image = context_image.crop(bbox)

        if self.context_transform is not None:
            context_image = self.context_transform(context_image)
        if self.person_transform is not None:
            person_image = self.person_transform(person_image)

        vad, vad_mask = self._vad_target(sample)
        return {
            "sample_id": sample.sample_id,
            "split": self.split,
            "image_path": str(sample.image_path),
            "relative_image_path": sample.relative_image_path,
            "original_database": sample.original_database,
            "context_image": context_image,
            "person_image": person_image,
            "bbox": torch.tensor(bbox, dtype=torch.float32),
            "labels": self._category_target(sample),
            "vad": vad,
            "vad_mask": vad_mask,
            "gender": sample.gender,
            "age": sample.age,
            "num_category_annotators": len(sample.category_annotations),
            "num_vad_annotators": len(sample.vad_annotations),
        }


__all__ = ["EMOTION_LABELS", "LABEL_TO_INDEX", "EmoticDataset"]
