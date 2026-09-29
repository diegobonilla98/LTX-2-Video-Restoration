from dataclasses import asdict, dataclass, field
from typing import Any, Literal

DegradationKind = Literal[
    "gaussian",
    "pixelation",
    "combined",
    "jpeg",
    "low_resolution",
    "mixed_restoration",
]
TrajectoryKind = Literal["progressive", "oneshot", "shuffled"]


@dataclass(frozen=True)
class DegradationSpec:
    kind: DegradationKind
    sigma0: float | None
    pixel_factor0: int | None
    seed: int
    jpeg_quality0: int | None = None
    lowres_factor0: float | None = None
    operator_order: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RestorationRecord:
    sample_id: str
    split: str
    source_root: str
    source_relpath: str
    source_sha256: str
    prompt: str
    trajectory: TrajectoryKind
    degradation: DegradationSpec
    clean_kind: Literal["natural", "text"]
    text: str | None = None
    font_id: str | None = None
    background_root: str | None = None
    background_relpath: str | None = None
    render: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["degradation"] = self.degradation.to_dict()
        return payload
