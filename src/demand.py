from dataclasses import dataclass, field
from typing import List, Optional, Dict
from defs import Priority, TravelMode
from geo import Point
from map import Raster


@dataclass
class DemandMode:
    index: Optional[int] = None
    raster_index: Optional[int] = None
    priority: Optional[Priority] = None
    arrival_rate: float = 0.0
    raster: Raster = field(default_factory=Raster)
    raster_multiplier: float = 0.0


@dataclass
class Demand:
    initialised: bool = False
    num_rasters: int = 0
    num_modes: int = 0
    num_sets: int = 0

    rasters: List[Raster] = field(default_factory=list)
    raster_filenames: List[str] = field(default_factory=list)
    modes: List[DemandMode] = field(default_factory=list)
    # mode_lookup[set_index][priority_index] -> demand mode index
    mode_lookup: List[List[int]] = field(default_factory=list)
    sets_start_times: List[float] = field(default_factory=list)
    sets_time_order: List[int] = field(default_factory=list)
    recent_sets_start_times_index: int = 0

    def current_set_index(self, t: float) -> Optional[int]:
        if not self.sets_start_times:
            return None
        idx = 0
        for i, start in enumerate(self.sets_start_times):
            if t >= start:
                idx = i
            else:
                break
        self.recent_sets_start_times_index = idx
        if idx < len(self.sets_time_order):
            return self.sets_time_order[idx]
        return None


@dataclass
class PointsCoverageMode:
    index: Optional[int] = None
    points: List[Point] = field(default_factory=list)
    cover_time: float = 0.0
    travel_mode: TravelMode = field(default_factory=TravelMode)
    point_sets: List[List[int]] = field(default_factory=list)
    station_sets: List[List[int]] = field(default_factory=list)
    stations_cover_point_sets: List[List[int]] = field(default_factory=list)


@dataclass
class DemandCoverage:
    cover_times: Dict[Priority, float] = field(default_factory=dict)
    raster_cell_num_rows: int = 0
    raster_cell_num_cols: int = 0

    initialised: bool = False
    points: List[Point] = field(default_factory=list)
    nodes_points: List[List[int]] = field(default_factory=list)
    rasters_point_demands: List[List[float]] = field(default_factory=list)

    points_coverage_modes: List[PointsCoverageMode] = field(default_factory=list)
    # lookup[travel_mode_index][cover_time] -> PointsCoverageMode index
    points_coverage_mode_lookup: Dict[int, Dict[float, int]] = field(default_factory=dict)
    # [points_coverage_mode_index][raster_index] -> demand per point-set
    point_sets_demands: List[List[List[float]]] = field(default_factory=list)
