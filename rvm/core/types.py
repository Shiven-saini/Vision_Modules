from dataclasses import dataclass
from typing import List, Tuple
from dataclasses import dataclass, asdict


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int

    def to_dict(self):
        """Convert Box object to dictionary for JSON serialization."""
        return asdict(self)

@dataclass
class Mask:
    segmentation: List[List[int]]   # polygon points
    confidence: float
    class_id: int

    def to_dict(self):
        return asdict(self)


@dataclass
class Marker:
    id: int
    corners: List[Tuple[int, int]]


@dataclass
class QRCode:
    data: str
    corners: List[Tuple[int, int]]