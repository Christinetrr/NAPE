#Schema for per-frame metrics
from dataclasses import dataclass
from typing import Any

@dataclass
class MouseDrag: 
    drag: bool
    start_pos: float
    end_pos: float
@dataclass
class MouseClick:
    click: bool
    timestamp: float
    x_pos: float
    y_pos: float
@dataclass
class FrameData:
    frame: int
    timestamp: float
    cursor_x: float 
    cursor_y: float
    cursor_match_score: float
    mouse_click_event: MouseClick  
    mouse_drag_event:  MouseDrag
    vel_x: float  
    vel_y: float 
    speed: float 
    acceleration: float 
    scene_change_score: float
    mag_pixel_change: float 
    nearest_target_objects: list[Any] 
    dist_cursor_to_target: float
    in_target_zone: bool 
    ui_change_score: float 



