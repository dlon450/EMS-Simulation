from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import math


@dataclass
class Location:
    x: Optional[float] = None
    y: Optional[float] = None

    def distance_to(self, other: "Location", x_scale: float = 1.0, y_scale: float = 1.0) -> float:
        """Euclidean distance using (optionally) scaled axes."""
        if self.x is None or self.y is None or other.x is None or other.y is None:
            raise ValueError("Cannot compute distance with unset coordinates")
        dx = (self.x - other.x) * x_scale
        dy = (self.y - other.y) * y_scale
        return math.hypot(dx, dy)


@dataclass
class Point:
    index: Optional[int] = None
    location: Location = field(default_factory=Location)
    value: Any = None
    nearest_node_index: Optional[int] = None
    nearest_node_dist: Optional[float] = None


@dataclass
class Node:
    index: Optional[int] = None
    location: Location = field(default_factory=Location)
    off_road_access: bool = True
    fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Arc:
    index: Optional[int] = None
    from_node_index: Optional[int] = None
    to_node_index: Optional[int] = None
    distance: Optional[float] = None  # None = not set
    fields: Dict[str, Any] = field(default_factory=dict)