from dataclasses import dataclass, field
from typing import List, Optional, Any
from geo import Location
from defs import Priority, RouteStatus
from misc import DistrRng


@dataclass
class Map:
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    x_scale: float = 1.0
    y_scale: float = 1.0

    @property
    def x_range(self) -> Optional[float]:
        if self.x_min is None or self.x_max is None:
            return None
        return self.x_max - self.x_min

    @property
    def y_range(self) -> Optional[float]:
        if self.y_min is None or self.y_max is None:
            return None
        return self.y_max - self.y_min

    def distance(self, a: Location, b: Location) -> float:
        """Off-road distance between two locations using this map's scaling."""
        return a.distance_to(b, x_scale=self.x_scale, y_scale=self.y_scale)


@dataclass
class GridSearchRect:
    x_dist: List[float] = field(default_factory=lambda: [0.0, 0.0])
    y_dist: List[float] = field(default_factory=lambda: [0.0, 0.0])
    ix_search: List[int] = field(default_factory=lambda: [0, 0])
    iy_search: List[int] = field(default_factory=lambda: [0, 0])
    ix_searched: List[int] = field(default_factory=lambda: [0, 0])
    iy_searched: List[int] = field(default_factory=lambda: [0, 0])


@dataclass
class GridRect:
    node_indices: List[int] = field(default_factory=list)


@dataclass
class Grid:
    nx: int = 0
    ny: int = 0
    x_range: float = 0.0
    y_range: float = 0.0
    x_range_dist: float = 0.0
    y_range_dist: float = 0.0
    rects: List[List[GridRect]] = field(default_factory=list)
    search_rect: GridSearchRect = field(default_factory=GridSearchRect)

    @classmethod
    def from_map(cls, map_: Map, nx: int, ny: int) -> "Grid":
        grid = cls(nx=nx, ny=ny)
        grid.rects = [[GridRect() for _ in range(ny)] for _ in range(nx)]
        grid.x_range = (map_.x_range or 0.0) / nx if nx > 0 else 0.0
        grid.y_range = (map_.y_range or 0.0) / ny if ny > 0 else 0.0
        grid.x_range_dist = grid.x_range * map_.x_scale
        grid.y_range_dist = grid.y_range * map_.y_scale
        return grid


@dataclass
class Raster:
    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    z: List[List[float]] = field(default_factory=list)

    @property
    def nx(self) -> int:
        return len(self.x)

    @property
    def ny(self) -> int:
        return len(self.y)

    @property
    def dx(self) -> float:
        if self.nx <= 1:
            return 0.0
        return (max(self.x) - min(self.x)) / (self.nx - 1)

    @property
    def dy(self) -> float:
        if self.ny <= 1:
            return 0.0
        return (max(self.y) - min(self.y)) / (self.ny - 1)


@dataclass
class RasterSampler:
    raster: Raster
    cell_distr_rng: DistrRng
    cell_loc_rng: Any  # random state for within-cell sampling


@dataclass
class Route:
    priority: Optional[Priority] = None
    travel_mode_index: Optional[int] = None

    start_loc: Location = field(default_factory=Location)
    start_time: Optional[float] = None
    end_loc: Location = field(default_factory=Location)
    end_time: Optional[float] = None

    start_fnode: Optional[int] = None
    start_fnode_time: Optional[float] = None
    start_fnode_dist: Optional[float] = None
    start_rnode: Optional[int] = None
    start_rnode_time: Optional[float] = None
    end_rnode: Optional[int] = None
    end_rnode_time: Optional[float] = None
    end_fnode: Optional[int] = None
    end_fnode_time: Optional[float] = None

    first_rarc: Optional[int] = None

    status: RouteStatus = RouteStatus.NULL
    recent_update_time: Optional[float] = None

    recent_rarc: Optional[int] = None
    recent_rarc_start_time: Optional[float] = None
    recent_rarc_end_time: Optional[float] = None

    recent_rarc_recent_fnode: Optional[int] = None
    recent_fnode: Optional[int] = None
    recent_fnode_time: Optional[float] = None

    recent_rarc_next_fnode: Optional[int] = None
    next_fnode: Optional[int] = None
    next_fnode_time: Optional[float] = None
    next_fnode_dist: Optional[float] = None

    def progress_fraction(self, t: float) -> float:
        """Return fraction of route completed at time t (0..1)."""
        if self.start_time is None or self.end_time is None or self.end_time <= self.start_time:
            return 0.0
        return max(0.0, min(1.0, (t - self.start_time) / (self.end_time - self.start_time)))
