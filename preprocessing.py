#Ingest mp4 instructional video files frame by frame

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from schema import FrameData, MouseClick, MouseDrag
from transcription_processing import extract_transcript
import pandas as pd

video = "resize_character.mp4"

# Opens video and returns cv2 VideoCapture 
def load_video(path: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Could not open video.")
    return cap


# Returns list of frames as numpy arrays
# Each frame : (height, width, 3), BGR, dtype uint8 — use frames[i] for frame index i.
def get_frames(path: str, show: bool = False):
    cap = load_video(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            if show:
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()
    return frames


def _estimate_blue_halo_hsv_bounds(reference_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate HSV bounds for the blue click halo from a reference image.
    Falls back to conservative defaults if estimation is unavailable.
    """
    # Default blue range (OpenCV HSV: H in [0,179]).
    default_lower = np.array([85, 50, 50], dtype=np.uint8)
    default_upper = np.array([135, 255, 255], dtype=np.uint8)

    ref = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    if ref is None or ref.size == 0:
        return default_lower, default_upper

    hsv = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)

    # Bootstrap a "blue-ish" mask, then tighten with robust percentiles.
    seed = cv2.inRange(hsv, np.array([80, 40, 40], dtype=np.uint8), np.array([140, 255, 255], dtype=np.uint8))
    ys, xs = np.where(seed > 0)
    if len(xs) < 20:
        return default_lower, default_upper

    sample = hsv[ys, xs]  # N x 3
    h = sample[:, 0].astype(np.float32)
    s = sample[:, 1].astype(np.float32)
    v = sample[:, 2].astype(np.float32)

    h_lo = int(np.clip(np.percentile(h, 5) - 5, 0, 179))
    h_hi = int(np.clip(np.percentile(h, 95) + 5, 0, 179))
    s_lo = int(np.clip(np.percentile(s, 20) - 15, 20, 255))
    v_lo = int(np.clip(np.percentile(v, 20) - 15, 20, 255))

    lower = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
    upper = np.array([h_hi, 255, 255], dtype=np.uint8)
    return lower, upper


#matches halo click event template to a radius around cursor position
#if match, record click event at the frame setting it to True, else False
def detect_click_events(
    frames: list[np.ndarray],
    positions: list[Tuple[Optional[int], Optional[int]]],
    *,
    ui_change_scores: Optional[list[float]] = None,
    halo_reference_path: str = "click_event.png",
    inner_radius_px: int = 10,
    outer_radius_px: int = 24,
    patch_radius_px: int = 30,
    halo_search_radius_px: int = 70,
    candidate_jitter_radius_px: int = 36,
    candidate_step_px: int = 10,
    z_threshold: float = 1.25,
    weak_z_threshold: float = 0.15,
    min_absolute_score: float = 0.04,
    min_separation_frames: int = 4,
    fps: Optional[float] = None,
    min_separation_seconds: float = 1.00,
    return_halo_centers: bool = False,
) -> list[bool] | Tuple[list[bool], list[Optional[Tuple[int, int]]]]:
    n = min(len(frames), len(positions))
    if n == 0:
        if return_halo_centers:
            return [], []
        return []

    lower_hsv, upper_hsv = _estimate_blue_halo_hsv_bounds(halo_reference_path)
    scores = np.full(n, np.nan, dtype=np.float32)
    halo_centers: list[Optional[Tuple[int, int]]] = [None] * n

    inner_r2 = float(inner_radius_px * inner_radius_px)
    outer_r2 = float(outer_radius_px * outer_radius_px)
    last_detected_pos: Optional[Tuple[int, int]] = None

    for i in range(n):
        x, y = positions[i]
        if x is not None and y is not None:
            last_detected_pos = (int(x), int(y))

        if last_detected_pos is None and (x is None or y is None):
            continue

        frame = frames[i]
        if frame is None or frame.size == 0:
            continue

        h, w = frame.shape[:2]
        if last_detected_pos is not None:
            anchor_x, anchor_y = last_detected_pos
        else:
            anchor_x, anchor_y = int(x), int(y)  # for type checker
        anchor_x = int(np.clip(anchor_x, 0, w - 1))
        anchor_y = int(np.clip(anchor_y, 0, h - 1))

        # Search in a larger window around last detected cursor position.
        sx0 = max(0, anchor_x - int(halo_search_radius_px))
        sx1 = min(w, anchor_x + int(halo_search_radius_px) + 1)
        sy0 = max(0, anchor_y - int(halo_search_radius_px))
        sy1 = min(h, anchor_y + int(halo_search_radius_px) + 1)
        patch = frame[sy0:sy1, sx0:sx1]
        if patch.size == 0:
            continue

        patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(patch_hsv, lower_hsv, upper_hsv)  # bit mask for blue halo

        # Candidate centers around anchor (plus current tracked point if available).
        candidates: list[Tuple[int, int]] = []
        jitter = max(0, int(candidate_jitter_radius_px))
        step = max(1, int(candidate_step_px))
        for dy in range(-jitter, jitter + 1, step):
            for dx in range(-jitter, jitter + 1, step):
                cx = int(np.clip(anchor_x + dx, 0, w - 1))
                cy = int(np.clip(anchor_y + dy, 0, h - 1))
                candidates.append((cx, cy))
        if x is not None and y is not None:
            candidates.append((int(np.clip(int(x), 0, w - 1)), int(np.clip(int(y), 0, h - 1))))

        yy, xx = np.ogrid[sy0:sy1, sx0:sx1]
        best_score = -1.0
        best_center: Optional[Tuple[int, int]] = None

        for cx, cy in candidates:
            dx = xx.astype(np.float32) - float(cx)
            dy = yy.astype(np.float32) - float(cy)
            d2 = dx * dx + dy * dy

            ring_mask = (d2 > inner_r2) & (d2 <= outer_r2)
            center_mask = d2 <= inner_r2
            if not np.any(ring_mask):
                continue

            ring_u8 = (ring_mask.astype(np.uint8) * 255)
            center_u8 = (center_mask.astype(np.uint8) * 255)
            ring_blue = cv2.bitwise_and(blue_mask, blue_mask, mask=ring_u8)
            center_blue = cv2.bitwise_and(blue_mask, blue_mask, mask=center_u8)

            ring_ratio = float(np.count_nonzero(ring_blue)) / float(np.count_nonzero(ring_mask))
            center_ratio = (
                float(np.count_nonzero(center_blue)) / float(np.count_nonzero(center_mask))
                if np.any(center_mask)
                else 0.0
            )
            s = ring_ratio - 0.5 * center_ratio
            if s > best_score:
                best_score = s
                best_center = (cx, cy)

        if best_score >= 0.0:
            scores[i] = float(best_score)
            if best_center is not None:
                halo_centers[i] = (int(best_center[0]), int(best_center[1]))
                # Track best halo center to stabilize next frame's search anchor.
                last_detected_pos = best_center

    valid = np.isfinite(scores)
    if not np.any(valid):
        out = [False] * n
        if return_halo_centers:
            return out, halo_centers
        return out

    valid_scores = scores[valid]
    baseline = float(np.median(valid_scores))
    spread = float(np.std(valid_scores))
    threshold = max(
        min_absolute_score,
        baseline + float(z_threshold) * spread,
    )
    weak_threshold = max(
        min_absolute_score,
        baseline + float(weak_z_threshold) * spread,
    )

    ui_arr: Optional[np.ndarray] = None
    ui_high_threshold = 0.0
    if ui_change_scores is not None and len(ui_change_scores) >= n:
        ui_arr = np.asarray(ui_change_scores[:n], dtype=np.float32)
        ui_high_threshold = float(np.percentile(ui_arr, 90))

    # Peak-picking: frame i is a click if it's a local maximum over threshold.
    peaks = np.zeros(n, dtype=bool)
    for i in range(1, n - 1):
        if not valid[i]:
            continue
        left = float(scores[i - 1]) if valid[i - 1] else -np.inf
        right = float(scores[i + 1]) if valid[i + 1] else -np.inf
        s = float(scores[i])
        strong_halo = s >= threshold
        weak_halo_with_ui = (
            (ui_arr is not None)
            and (s >= weak_threshold)
            and (float(ui_arr[i]) >= ui_high_threshold)
        )
        if (strong_halo or weak_halo_with_ui) and s >= left and s > right:
            peaks[i] = True

    # Edge frames: allow them to be peaks too.
    if n >= 1 and valid[0] and float(scores[0]) >= threshold:
        right0 = float(scores[1]) if n > 1 and valid[1] else -np.inf
        if float(scores[0]) > right0:
            peaks[0] = True
    if n >= 2 and valid[n - 1] and float(scores[n - 1]) >= threshold:
        leftn = float(scores[n - 2]) if valid[n - 2] else -np.inf
        if float(scores[n - 1]) >= leftn:
            peaks[n - 1] = True

    # Enforce minimum spacing between click frames (time-based preferred).
    sep_frames = int(min_separation_frames)
    if fps is not None and fps > 0 and min_separation_seconds > 0:
        sep_frames = max(sep_frames, int(round(float(min_separation_seconds) * float(fps))))

    click_flags = np.zeros(n, dtype=bool)
    peak_idxs = np.where(peaks)[0]
    if sep_frames <= 0 or len(peak_idxs) == 0:
        click_flags = peaks
        out = click_flags.tolist()
        if return_halo_centers:
            return out, halo_centers
        return out

    # Group nearby peaks and keep the strongest score in each group.
    cluster_start = int(peak_idxs[0])
    cluster_best = int(peak_idxs[0])
    for p in peak_idxs[1:]:
        p_int = int(p)
        if (p_int - cluster_start) <= sep_frames:
            if float(scores[p_int]) > float(scores[cluster_best]):
                cluster_best = p_int
        else:
            click_flags[cluster_best] = True
            cluster_start = p_int
            cluster_best = p_int
    click_flags[cluster_best] = True

    out = click_flags.tolist()
    if return_halo_centers:
        return out, halo_centers
    return out


def compute_ui_change_scores(
    frames: list[np.ndarray],
    positions: list[Tuple[Optional[int], Optional[int]]],
    *,
    patch_radius_px: int = 36,
    blur_ksize: int = 3,
) -> list[float]:
    """
    Compute per-frame UI change near cursor by comparing to previous frame.

    Score definition:
      mean(abs(curr_patch - prev_patch)) / 255.0  -> in [0, 1]
    """
    n = min(len(frames), len(positions))
    if n == 0:
        return []

    scores = [0.0] * n
    if n == 1:
        return scores

    k = max(1, int(blur_ksize))
    if k % 2 == 0:
        k += 1

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
    if k >= 3:
        prev_gray = cv2.GaussianBlur(prev_gray, (k, k), 0)

    last_valid_pos: Optional[Tuple[int, int]] = None
    x0, y0 = positions[0]
    if x0 is not None and y0 is not None:
        last_valid_pos = (int(x0), int(y0))

    for i in range(1, n):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        if k >= 3:
            curr_gray = cv2.GaussianBlur(curr_gray, (k, k), 0)

        h, w = curr_gray.shape[:2]
        x, y = positions[i]
        if x is not None and y is not None:
            cx, cy = int(x), int(y)
            last_valid_pos = (cx, cy)
        elif last_valid_pos is not None:
            cx, cy = last_valid_pos
        else:
            cx, cy = w // 2, h // 2

        cx = int(np.clip(cx, 0, w - 1))
        cy = int(np.clip(cy, 0, h - 1))

        px0 = max(0, cx - patch_radius_px)
        px1 = min(w, cx + patch_radius_px + 1)
        py0 = max(0, cy - patch_radius_px)
        py1 = min(h, cy + patch_radius_px + 1)

        curr_patch = curr_gray[py0:py1, px0:px1]
        prev_patch = prev_gray[py0:py1, px0:px1]
        if curr_patch.size == 0 or prev_patch.size == 0:
            scores[i] = 0.0
        else:
            mad = float(np.mean(np.abs(curr_patch - prev_patch)))
            scores[i] = mad / 255.0

        prev_gray = curr_gray

    return scores


def compute_cursor_kinematics(
    positions: list[Tuple[Optional[int], Optional[int]]],
    fps: float,
) -> Tuple[list[float], list[float], list[float], list[float]]:
    """
    Compute per-frame cursor kinematics:
      vel_x = (x_t - x_prev) / dt
      vel_y = (y_t - y_prev) / dt
      speed = sqrt(vel_x^2 + vel_y^2)
      acceleration = (speed_t - speed_prev) / dt
    """
    n = len(positions)
    vel_x = [0.0] * n
    vel_y = [0.0] * n
    speed = [0.0] * n
    acceleration = [0.0] * n

    if n == 0:
        return vel_x, vel_y, speed, acceleration

    dt = 1.0 / float(fps) if fps and fps > 0 else 1.0 / 30.0
    prev_speed = 0.0
    prev_valid = False

    for i in range(1, n):
        x_prev, y_prev = positions[i - 1]
        x_t, y_t = positions[i]
        if x_prev is None or y_prev is None or x_t is None or y_t is None:
            vel_x[i] = 0.0
            vel_y[i] = 0.0
            speed[i] = 0.0
            acceleration[i] = 0.0
            prev_speed = 0.0
            prev_valid = False
            continue

        vx = (float(x_t) - float(x_prev)) / dt
        vy = (float(y_t) - float(y_prev)) / dt
        sp = float(np.sqrt(vx * vx + vy * vy))
        if prev_valid:
            acc = (sp - prev_speed) / dt
        else:
            acc = 0.0

        vel_x[i] = vx
        vel_y[i] = vy
        speed[i] = sp
        acceleration[i] = acc

        prev_speed = sp
        prev_valid = True

    return vel_x, vel_y, speed, acceleration


def _smooth_1d(values: list[float], window: int = 7) -> list[float]:
    n = len(values)
    if n == 0:
        return []
    w = max(1, int(window))
    if w == 1:
        return [float(v) for v in values]
    if w % 2 == 0:
        w += 1
    pad = w // 2
    arr = np.asarray(values, dtype=np.float32)
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=np.float32) / float(w)
    out = np.convolve(padded, kernel, mode="valid")
    return out.astype(np.float32).tolist()


def detect_stop_events(
    speed_series: list[float],
    fps: float,
    *,
    stable_stop_seconds: float = 1.0,
    low_speed_threshold: Optional[float] = None,
    min_separation_seconds: float = 1.0,
) -> list[bool]:
    """
    Detect stop events where smoothed speed stays low continuously for >= stable_stop_seconds.
    """
    n = len(speed_series)
    if n == 0:
        return []

    dt = 1.0 / float(fps) if fps and fps > 0 else 1.0 / 30.0
    stable_frames = max(1, int(round(float(stable_stop_seconds) / dt)))
    sep_frames = max(1, int(round(float(min_separation_seconds) / dt)))

    speed_smooth = _smooth_1d(speed_series, window=7)
    if len(speed_smooth) < n:
        speed_smooth = speed_smooth + [0.0] * (n - len(speed_smooth))
    arr = np.asarray(speed_smooth[:n], dtype=np.float32)

    if low_speed_threshold is None:
        nonzero = arr[arr > 1e-6]
        if nonzero.size > 0:
            low_speed_threshold = float(np.percentile(nonzero, 25))
        else:
            low_speed_threshold = 40.0

    low = arr <= float(low_speed_threshold)
    raw_stops = np.zeros(n, dtype=bool)
    run = 0
    for i in range(n):
        if low[i]:
            run += 1
            if run >= stable_frames:
                raw_stops[i] = True
        else:
            run = 0

    # De-duplicate nearby stop detections.
    out = np.zeros(n, dtype=bool)
    last = -10_000
    for i in range(n):
        if raw_stops[i] and (i - last) >= sep_frames:
            out[i] = True
            last = i
    return out.tolist()


def build_smart_zoom_controls(
    positions: list[Tuple[Optional[int], Optional[int]]],
    click_events: list[bool],
    speed_series: list[float],
    ui_change_scores: list[float],
    *,
    fps: float,
    frame_width: int,
    frame_height: int,
    max_zoom: float = 1.6,
    lookback_seconds: float = 0.25,
    hold_after_click_seconds: float = 0.5,
    idle_zoom_out_seconds: float = 0.75,
    zoom_in_min_seconds: float = 0.80,
    reanchor_seconds: float = 0.55,
    stable_stop_seconds: float = 1.0,
    low_speed_threshold: Optional[float] = None,
    interaction_ui_threshold: Optional[float] = None,
) -> Tuple[list[float], list[Tuple[int, int]]]:
    """
    Build per-frame zoom level [0..1] and zoom anchor that satisfy smart click zoom:
    pre-click settle start, hold, inactivity zoom-out, and smooth re-anchor/pan.
    """
    n = len(positions)
    if n == 0:
        return [], []

    speed_smooth = _smooth_1d(speed_series, window=7)
    if len(speed_smooth) < n:
        speed_smooth = speed_smooth + [0.0] * (n - len(speed_smooth))
    ui_smooth = _smooth_1d(ui_change_scores, window=5)
    if len(ui_smooth) < n:
        ui_smooth = ui_smooth + [0.0] * (n - len(ui_smooth))

    speed_arr = np.asarray(speed_smooth[:n], dtype=np.float32)
    ui_arr = np.asarray(ui_smooth[:n], dtype=np.float32)

    if low_speed_threshold is None:
        nonzero = speed_arr[speed_arr > 1e-6]
        if nonzero.size > 0:
            low_speed_threshold = float(np.percentile(nonzero, 25))
        else:
            low_speed_threshold = 40.0
    if interaction_ui_threshold is None:
        interaction_ui_threshold = float(max(0.01, np.percentile(ui_arr, 70)))

    dt = 1.0 / float(fps) if fps and fps > 0 else 1.0 / 30.0
    back_frames = max(1, int(round(lookback_seconds / dt)))
    hold_frames = max(1, int(round(hold_after_click_seconds / dt)))
    idle_frames = max(1, int(round(idle_zoom_out_seconds / dt)))
    reanchor_frames = max(2, int(round(reanchor_seconds / dt)))
    zoom_in_min_frames = max(2, int(round(zoom_in_min_seconds / dt)))
    stable_stop_frames = max(1, int(round(max(0.0, float(stable_stop_seconds)) / dt)))

    def _valid_pos(i: int) -> Optional[Tuple[int, int]]:
        if i < 0 or i >= n:
            return None
        x, y = positions[i]
        if x is None or y is None:
            return None
        return (int(np.clip(int(x), 0, frame_width - 1)), int(np.clip(int(y), 0, frame_height - 1)))

    def _last_valid_before(i: int) -> Optional[Tuple[int, int]]:
        j = i
        while j >= 0:
            p = _valid_pos(j)
            if p is not None:
                return p
            j -= 1
        return None

    def _stable_before(i: int) -> Optional[Tuple[int, int]]:
        # A valid ending position requires continuous low-speed stop for >= stable_stop_seconds.
        search_back = max(back_frames, stable_stop_frames + 2)
        lo = max(0, i - search_back)
        for j in range(i, lo - 1, -1):
            start = j - stable_stop_frames + 1
            if start < 0:
                continue
            if np.all(speed_arr[start : j + 1] <= float(low_speed_threshold)):
                p = _valid_pos(j)
                if p is not None:
                    return p
        return None

    click_idxs = [i for i, c in enumerate(click_events[:n]) if bool(c)]
    zoom_start_map: dict[int, int] = {}
    for c in click_idxs:
        lo = max(0, c - back_frames)
        settle_idx: Optional[int] = None
        for j in range(c, lo - 1, -1):
            if speed_arr[j] <= float(low_speed_threshold):
                settle_idx = j
                break
        start_idx = settle_idx if settle_idx is not None else lo
        zoom_start_map[start_idx] = c

    level = np.zeros(n, dtype=np.float32)
    anchors: list[Tuple[int, int]] = [(frame_width // 2, frame_height // 2) for _ in range(n)]

    zoom_active = False
    hold_until = -1
    inactivity_count = 0
    transition_start = -1
    transition_end = -1
    transition_from = 0.0
    transition_to = 0.0

    anchor_curr = (frame_width // 2, frame_height // 2)
    pan_start = -1
    pan_end = -1
    pan_from = anchor_curr
    pan_to = anchor_curr
    click_ptr = 0

    def _begin_zoom_transition(i: int, to_level: float, duration_frames: int) -> None:
        nonlocal transition_start, transition_end, transition_from, transition_to
        transition_start = i
        transition_end = i + max(1, int(duration_frames))
        transition_from = float(level[i - 1]) if i > 0 else float(level[i])
        transition_to = float(np.clip(to_level, 0.0, 1.0))

    def _begin_pan(i: int, to_anchor: Tuple[int, int], duration_frames: int) -> None:
        nonlocal pan_start, pan_end, pan_from, pan_to
        pan_start = i
        pan_end = i + max(1, int(duration_frames))
        pan_from = anchor_curr
        pan_to = (
            int(np.clip(int(to_anchor[0]), 0, frame_width - 1)),
            int(np.clip(int(to_anchor[1]), 0, frame_height - 1)),
        )

    for i in range(n):
        while click_ptr < len(click_idxs) and click_idxs[click_ptr] <= i:
            click_ptr += 1
        next_click_idx: Optional[int] = click_idxs[click_ptr] if click_ptr < len(click_idxs) else None

        if i in zoom_start_map:
            c = zoom_start_map[i]
            c_anchor = _stable_before(c)
            if c_anchor is None:
                c_anchor = _last_valid_before(c)
            if c_anchor is None:
                c_anchor = (frame_width // 2, frame_height // 2)
            anchor_curr = c_anchor
            _begin_zoom_transition(i, 1.0, max(zoom_in_min_frames, c - i + 1))
            zoom_active = True

        if i in click_idxs:
            click_anchor = _stable_before(i)
            if click_anchor is None:
                click_anchor = _last_valid_before(i)
            if click_anchor is None:
                click_anchor = anchor_curr

            # Determine if click lies within central 60% of current zoom region.
            prev_level = float(level[i - 1]) if i > 0 else float(level[i])
            p = np.clip(prev_level, 0.0, 1.0)
            p_s = p * p * (3.0 - 2.0 * p)
            roi_w = float(frame_width) + (float(frame_width) / max(1.0, max_zoom) - float(frame_width)) * p_s
            roi_h = float(frame_height) + (float(frame_height) / max(1.0, max_zoom) - float(frame_height)) * p_s
            in_inner = (
                abs(float(click_anchor[0]) - float(anchor_curr[0])) <= 0.30 * roi_w
                and abs(float(click_anchor[1]) - float(anchor_curr[1])) <= 0.30 * roi_h
            )

            # Keep zoomed; smoothly pan/re-anchor without snapping to full frame.
            pan_frames = reanchor_frames if in_inner else int(round(1.25 * reanchor_frames))
            _begin_pan(i, click_anchor, pan_frames)
            _begin_zoom_transition(i, 1.0, max(2, int(round(0.45 / dt))))
            zoom_active = True
            hold_until = max(hold_until, i + hold_frames)
            inactivity_count = 0

        if pan_start >= 0 and pan_end > pan_start and i <= pan_end:
            tp = float(i - pan_start) / float(max(1, pan_end - pan_start))
            tp = float(np.clip(tp, 0.0, 1.0))
            ts = tp * tp * (3.0 - 2.0 * tp)
            ax = int(round(float(pan_from[0]) + (float(pan_to[0]) - float(pan_from[0])) * ts))
            ay = int(round(float(pan_from[1]) + (float(pan_to[1]) - float(pan_from[1])) * ts))
            anchor_curr = (
                int(np.clip(ax, 0, frame_width - 1)),
                int(np.clip(ay, 0, frame_height - 1)),
            )
        elif pan_end > 0 and i > pan_end:
            anchor_curr = pan_to

        if transition_start >= 0 and transition_end > transition_start and i <= transition_end:
            tz = float(i - transition_start) / float(max(1, transition_end - transition_start))
            tz = float(np.clip(tz, 0.0, 1.0))
            ts = tz * tz * (3.0 - 2.0 * tz)
            level[i] = float(transition_from + (transition_to - transition_from) * ts)
        else:
            level[i] = float(level[i - 1]) if i > 0 else 0.0

        meaningful_interaction = (ui_arr[i] >= float(interaction_ui_threshold)) or (
            speed_arr[i] >= float(low_speed_threshold) * 1.25
        )
        if zoom_active:
            if i <= hold_until:
                inactivity_count = 0
            elif meaningful_interaction:
                inactivity_count = 0
            else:
                inactivity_count += 1
                if inactivity_count >= idle_frames and float(level[i]) > 1e-3:
                    # If the next action is already within the current inner bounds,
                    # keep zoomed state and do not zoom back out to full frame.
                    keep_zoom_for_next_action = False
                    if next_click_idx is not None:
                        next_anchor = _stable_before(next_click_idx)
                        if next_anchor is None:
                            next_anchor = _last_valid_before(next_click_idx)
                        if next_anchor is not None:
                            p_now = float(np.clip(level[i], 0.0, 1.0))
                            p_now_s = p_now * p_now * (3.0 - 2.0 * p_now)
                            roi_w_now = float(frame_width) + (
                                float(frame_width) / max(1.0, max_zoom) - float(frame_width)
                            ) * p_now_s
                            roi_h_now = float(frame_height) + (
                                float(frame_height) / max(1.0, max_zoom) - float(frame_height)
                            ) * p_now_s
                            keep_zoom_for_next_action = (
                                abs(float(next_anchor[0]) - float(anchor_curr[0])) <= 0.30 * roi_w_now
                                and abs(float(next_anchor[1]) - float(anchor_curr[1])) <= 0.30 * roi_h_now
                            )
                    if keep_zoom_for_next_action:
                        inactivity_count = idle_frames
                        anchors[i] = anchor_curr
                        continue
                    _begin_zoom_transition(i, 0.0, max(zoom_in_min_frames, int(round(1.25 * idle_frames))))
                    zoom_active = False
                    inactivity_count = 0

        anchors[i] = anchor_curr

    return level.astype(np.float32).tolist(), anchors

'''CURSOR METRICS EXTRACTION: template matching + iterate over all 
   frames + compute match confidence score to template + record cursor
   tip position as x and y only if confidence above threshold. 
   
   If position extracted is too far from previous location, interpret 
   as noisy and reject. '''
def find_cursor_positions(
    video_path,
    template_path,
    threshold=0.5,
    search_radius=None,
    mask_path=None,
    mask_bg_threshold=200,
    mask_morph_open_ksize=3,
    mask_pad=2,
    alpha_threshold=1,
    max_jump_px=600,
    return_scores=False,
    *,
    tip_offset_from_top_left: Optional[Tuple[int, int]] = None,
    auto_tip_from_mask: bool = True,
):
    #applies bitwise masking to match actual cursor object
    def load_template_and_mask(t_path, m_path):
        raw = cv2.imread(t_path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Cursor template not found: {t_path}")

        tmpl_gray = None
        mask = None

        if raw.ndim == 3 and raw.shape[2] == 4:
            b, g, r, a = cv2.split(raw)
            bgr = cv2.merge([b, g, r])
            mask = cv2.threshold(a, alpha_threshold, 255, cv2.THRESH_BINARY)[1]
            tmpl_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        else:
            tmpl_gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if raw.ndim == 3 else raw

            _, mask = cv2.threshold(
                tmpl_gray,
                mask_bg_threshold,
                255,
                cv2.THRESH_BINARY_INV,
            )

            k = max(1, int(mask_morph_open_ksize))
            if k % 2 == 0:
                k += 1
            if k >= 3:
                kernel = np.ones((k, k), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if num_labels > 1:
                largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                clean_mask = np.zeros_like(mask)
                clean_mask[labels == largest] = 255
                mask = clean_mask

            if m_path is None:
                ys, xs = np.where(mask > 0)
                if len(xs) > 0 and len(ys) > 0:
                    x0, x1 = int(xs.min()), int(xs.max())
                    y0, y1 = int(ys.min()), int(ys.max())
                    pad = max(0, int(mask_pad))
                    x0 = max(0, x0 - pad)
                    y0 = max(0, y0 - pad)
                    x1 = min(tmpl_gray.shape[1] - 1, x1 + pad)
                    y1 = min(tmpl_gray.shape[0] - 1, y1 + pad)
                    tmpl_gray = tmpl_gray[y0 : y1 + 1, x0 : x1 + 1]
                    mask = mask[y0 : y1 + 1, x0 : x1 + 1]

        if m_path is not None:
            m = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise FileNotFoundError(f"Cursor mask not found: {m_path}")
            mask = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY)[1]

        return tmpl_gray, mask

    tmpl, mask = load_template_and_mask(template_path, mask_path)
    th, tw = tmpl.shape[:2]

    if tip_offset_from_top_left is not None:
        pos_ox, pos_oy = (
            int(tip_offset_from_top_left[0]),
            int(tip_offset_from_top_left[1]),
        )
    elif auto_tip_from_mask and mask is not None:
        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            # Top-left of mask bbox — usual hot-spot for arrow cursors
            pos_ox, pos_oy = int(xs.min()), int(ys.min())
        else:
            pos_ox, pos_oy = 0, 0
    else:
        pos_ox, pos_oy = 0, 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    positions = []
    scores = []
    prev_pos = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if search_radius is not None and prev_pos is not None:
                prev_cx, prev_cy = prev_pos
                x0 = max(0, prev_cx - search_radius)
                y0 = max(0, prev_cy - search_radius)
                x1 = min(gray.shape[1], prev_cx + search_radius)
                y1 = min(gray.shape[0], prev_cy + search_radius)
                roi = gray[y0:y1, x0:x1]
                if roi.shape[1] >= tw and roi.shape[0] >= th:
                    if mask is not None:
                        try:
                            res = cv2.matchTemplate(roi, tmpl, cv2.TM_CCORR_NORMED, mask=mask)
                        except TypeError:
                            res = cv2.matchTemplate(roi, tmpl, cv2.TM_CCORR_NORMED)
                    else:
                        res = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    x, y = max_loc
                    x_global, y_global = x0 + x, y0 + y
                else:
                    if mask is not None:
                        try:
                            res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCORR_NORMED, mask=mask)
                        except TypeError:
                            res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCORR_NORMED)
                    else:
                        res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    x_global, y_global = max_loc
            else:
                if mask is not None:
                    try:
                        res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCORR_NORMED, mask=mask)
                    except TypeError:
                        res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCORR_NORMED)
                else:
                    res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                x_global, y_global = max_loc

            scores.append(float(max_val))

            if max_val >= threshold:
                px = x_global + pos_ox
                py = y_global + pos_oy
                if max_jump_px is not None and prev_pos is not None:
                    prev_x, prev_y = prev_pos
                    dx = px - prev_x
                    dy = py - prev_y
                    if (dx * dx + dy * dy) > (max_jump_px * max_jump_px):
                        positions.append((None, None))
                    else:
                        positions.append((int(px), int(py)))
                        prev_pos = (int(px), int(py))
                else:
                    positions.append((int(px), int(py)))
                    prev_pos = (int(px), int(py))
            else:
                positions.append((None, None))
                prev_pos = None
    finally:
        cap.release()

    if return_scores:
        return positions, scores
    return positions

#overlays enlarged cursor on original cursor 
def visualize_cursor_positions(
    video_path,
    positions,
    out_path = "cursor_debug.mp4",
    *,
    early_seconds = 2.0,
    normal_radius = 8,
    early_radius = 16,
    overlay_path = "mac_cursor.png",
    overlay_hotspot_offset = (0, 0),
    overlay_hotspot_xy: Tuple[int, int] = (0, 0),
    click_events: Optional[list[bool]] = None,
    click_halo_centers: Optional[list[Optional[Tuple[int, int]]]] = None,
    click_overlay_gate_radius_px: int = 18,
    speed_series: Optional[list[float]] = None,
    ui_change_scores: Optional[list[float]] = None,
    enable_smart_zoom: bool = True,
    smart_zoom_max: float = 1.6,
):

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open for out_path={out_path}")

    early_end_idx = None

    try:
        overlay_raw = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
        if overlay_raw is None:
            raise FileNotFoundError(f"Overlay cursor not found: {overlay_path}")
        if overlay_raw.ndim == 3 and overlay_raw.shape[2] == 4:
            ob, og, orr, oa = cv2.split(overlay_raw)
            overlay_bgr = cv2.merge([ob, og, orr])
            overlay_alpha = oa  # 0..255
        else:
            overlay_bgr = overlay_raw[:, :, :3] if overlay_raw.ndim == 3 else overlay_raw
            overlay_gray = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2GRAY)
            _, overlay_alpha = cv2.threshold(overlay_gray, 250, 255, cv2.THRESH_BINARY_INV)

        ox_off, oy_off = overlay_hotspot_offset
        hx, hy = int(overlay_hotspot_xy[0]), int(overlay_hotspot_xy[1])

        normal_radius = max(1, int(normal_radius))
        early_radius = max(1, int(early_radius))

        zoom_levels: Optional[list[float]] = None
        zoom_anchors: Optional[list[Tuple[int, int]]] = None
        if enable_smart_zoom and click_events is not None and speed_series is not None and ui_change_scores is not None:
            zoom_levels, zoom_anchors = build_smart_zoom_controls(
                positions=positions,
                click_events=click_events,
                speed_series=speed_series,
                ui_change_scores=ui_change_scores,
                fps=float(fps),
                frame_width=width,
                frame_height=height,
                max_zoom=float(smart_zoom_max),
            )

        idx = 0
        if early_seconds is not None:
            early_end_idx = int(early_seconds * fps)
        while True:
            ret, frame = cap.read()
            if not ret or idx >= len(positions):
                break

            x, y = positions[idx]
            has_cursor_detection = x is not None and y is not None
            draw_x, draw_y = x, y

            show_overlay = has_cursor_detection
            if (
                show_overlay
                and click_events is not None
                and idx < len(click_events)
                and click_events[idx]
            ):
                # On click frames, require cursor to stay near detected blue halo center.
                if (
                    click_halo_centers is None
                    or idx >= len(click_halo_centers)
                    or click_halo_centers[idx] is None
                ):
                    show_overlay = False
                else:
                    hx_c, hy_c = click_halo_centers[idx]  # type: ignore[misc]
                    dx_c = float(x) - float(hx_c)  # x is not None because show_overlay is True
                    dy_c = float(y) - float(hy_c)
                    if (dx_c * dx_c + dy_c * dy_c) > float(click_overlay_gate_radius_px * click_overlay_gate_radius_px):
                        show_overlay = False

            # Smart zoom around click episodes.
            if (
                zoom_levels is not None
                and zoom_anchors is not None
                and idx < len(zoom_levels)
                and idx < len(zoom_anchors)
            ):
                z = float(np.clip(zoom_levels[idx], 0.0, 1.0))
                if z > 1e-6:
                    anchor = zoom_anchors[idx]
                    frame, mapped_cursor, _, _ = zoom_roi(
                        frame,
                        (x, y),
                        z,
                        max_zoom=float(smart_zoom_max),
                        fallback_center=anchor,
                        zoom_anchor=anchor,
                    )
                    if mapped_cursor is not None:
                        draw_x, draw_y = mapped_cursor

            # Only draw overlay when this frame passes visibility gating.
            if show_overlay and draw_x is not None and draw_y is not None:
                h0, w0 = overlay_bgr.shape[:2]
                target_r = early_radius
                if early_end_idx is not None and early_end_idx > 0 and idx < early_end_idx:
                    t = idx / float(early_end_idx)  # 0..1
                    t_s = t * t * (3.0 - 2.0 * t)
                    target_r = normal_radius + (early_radius - normal_radius) * t_s

                base_target_h = max(1, int(round(2.0 * target_r)))
                base_scale = float(base_target_h) / float(h0)
                base_target_w = max(1, int(round(w0 * base_scale)))

                # Keep cursor overlay size driven only by early-radius animation,
                # not by the camera zoom, so the overlay doesn't "grow" during zoom.
                target_h = base_target_h
                target_w = base_target_w

                ov_bgr = cv2.resize(overlay_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
                ov_alpha = cv2.resize(
                    overlay_alpha,
                    (target_w, target_h),
                    interpolation=cv2.INTER_NEAREST,
                )

                w, h = target_w, target_h
                cx, cy = int(draw_x) + ox_off, int(draw_y) + oy_off
                # Align overlay hotspot to recorded tip; scale with resize
                hx_s = hx * (target_w / float(w0)) if w0 else 0.0
                hy_s = hy * (target_h / float(h0)) if h0 else 0.0
                x0 = int(round(cx - hx_s))
                y0 = int(round(cy - hy_s))

                fx0 = max(0, x0)
                fy0 = max(0, y0)
                fx1 = min(frame.shape[1], x0 + w)
                fy1 = min(frame.shape[0], y0 + h)

                if fx1 > fx0 and fy1 > fy0:
                    ox0 = fx0 - x0
                    oy0 = fy0 - y0
                    ox1 = ox0 + (fx1 - fx0)
                    oy1 = oy0 + (fy1 - fy0)

                    roi = frame[fy0:fy1, fx0:fx1]
                    ov_roi = ov_bgr[oy0:oy1, ox0:ox1]
                    a_roi = ov_alpha[oy0:oy1, ox0:ox1].astype(np.float32) / 255.0
                    a3 = a_roi[:, :, None]
                    frame[fy0:fy1, fx0:fx1] = (roi * (1.0 - a3) + ov_roi * a3).astype(
                        roi.dtype
                    )

            writer.write(frame)
            idx += 1
    finally:
        cap.release()
        writer.release()

"""ZOOMING LOGIC"""
def zoom_roi(
    frame: np.ndarray,
    cursor_pos: Optional[Tuple[Optional[int], Optional[int]]],
    progress: float,
    *,
    max_zoom: float = 2.5,
    fallback_center: Optional[Tuple[int, int]] = None,
    zoom_anchor: Optional[Tuple[int, int]] = None,   # NEW
) -> Tuple[np.ndarray, Optional[Tuple[int, int]], float, float]:
    if frame is None or frame.size == 0:
        raise ValueError("zoom_roi received an empty frame.")

    h, w = frame.shape[:2]
    if h <= 1 or w <= 1:
        return frame.copy(), None, 1.0, 1.0

    def smoothstep(t: float) -> float:
        t = float(np.clip(t, 0.0, 1.0))
        return t * t * (3.0 - 2.0 * t)

    def interpolate(a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    p = smoothstep(progress)

    start_w, start_h = float(w), float(h)
    target_w = float(w) / max(1.0, float(max_zoom))
    target_h = float(h) / max(1.0, float(max_zoom))

    roi_w = int(round(interpolate(start_w, target_w, p)))
    roi_h = int(round(interpolate(start_h, target_h, p)))

    roi_w = max(2, min(w, roi_w))
    roi_h = max(2, min(h, roi_h))

    # FIXED zoom center
    if zoom_anchor is not None:
        cx, cy = int(zoom_anchor[0]), int(zoom_anchor[1])
    elif (
        cursor_pos is not None
        and len(cursor_pos) == 2
        and cursor_pos[0] is not None
        and cursor_pos[1] is not None
    ):
        cx, cy = int(cursor_pos[0]), int(cursor_pos[1])
    elif fallback_center is not None:
        cx, cy = int(fallback_center[0]), int(fallback_center[1])
    else:
        cx, cy = w // 2, h // 2

    cx = int(np.clip(cx, 0, w - 1))
    cy = int(np.clip(cy, 0, h - 1))

    half_w = roi_w // 2
    half_h = roi_h // 2

    x0 = max(0, min(w - roi_w, cx - half_w))
    y0 = max(0, min(h - roi_h, cy - half_h))
    x1 = x0 + roi_w
    y1 = y0 + roi_h

    crop = frame[y0:y1, x0:x1]
    if crop.shape[0] != roi_h or crop.shape[1] != roi_w:
        return frame.copy(), None, 1.0, 1.0

    zoomed = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

    mapped_cursor = None
    if (
        cursor_pos is not None
        and len(cursor_pos) == 2
        and cursor_pos[0] is not None
        and cursor_pos[1] is not None
    ):
        sx = w / float(roi_w)
        sy = h / float(roi_h)
        mx = int(round((float(cursor_pos[0]) - float(x0)) * sx))
        my = int(round((float(cursor_pos[1]) - float(y0)) * sy))
        mapped_cursor = (int(np.clip(mx, 0, w - 1)), int(np.clip(my, 0, h - 1)))

    scale_x = w / float(roi_w)
    scale_y = h / float(roi_h)

    return zoomed, mapped_cursor, scale_x, scale_y
if __name__ == "__main__":
    cursor_template = "cursor.png"
    frames = get_frames(video)

    #CURSOR METRICS
    #track and retrieve cursor positional data
    #returns list of (x, y) positions of cursor tip. on error returns (NaN, NaN)
    cursor_positions, scores = find_cursor_positions(
        video,
        cursor_template,
        threshold=0.7,
        search_radius=140,  # constrain matching near last known cursor location
        mask_bg_threshold=170,
        mask_morph_open_ksize=3,
        mask_pad=1,
        max_jump_px=120,  # reject moderate jumps to avoid locking onto wrong regions
        return_scores=True,
        auto_tip_from_mask=True,  # tip = top-left of mask bbox; or set tip_offset_from_top_left
    )

    #retrieve fps from video
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    fps = float(fps) if fps and fps > 0 else 30.0

    
    #BUILD DATAFRAME OF FRAME METRICS
    ui_change_scores = compute_ui_change_scores(frames, cursor_positions)
    click_events, click_halo_centers = detect_click_events(
        frames,
        cursor_positions,
        ui_change_scores=ui_change_scores,
        halo_reference_path="click_event.png",
        fps=fps,
        min_separation_seconds=1.00,
        return_halo_centers=True,
    )
    vel_x, vel_y, speed, acceleration = compute_cursor_kinematics(cursor_positions, fps)
    stop_events = detect_stop_events(speed, fps, stable_stop_seconds=1.0)

    #do the cursor overlay (with smart click-centric zoom)
    visualize_cursor_positions(
        video,
        cursor_positions,
        out_path="cursor_debug.mp4",
        early_seconds=1,
        normal_radius=14,
        early_radius=35,
        overlay_path="mac_cursor.png",
        overlay_hotspot_xy=(0, 0),
        click_events=click_events,
        click_halo_centers=click_halo_centers,
        click_overlay_gate_radius_px=18,
        speed_series=speed,
        ui_change_scores=ui_change_scores,
        enable_smart_zoom=True,
        smart_zoom_max=1.6,
    )

    #TRANSCRIPTION METRICS
    try:
        #extract transcription and writes text to file
        #NTD: record transcription metrics and send to LLM for evaluation
        transcript_text = extract_transcript(video)
        transcript_out = Path("transcript.txt")
        transcript_out.write_text(transcript_text, encoding="utf-8")
        print(f"Wrote transcript to {transcript_out}")
    except Exception as e:
        print(f"Transcript extraction failed: {e}")

    for i, is_click in enumerate(click_events):
        if is_click:
            print(f"Click detected at timestamp: {i / fps:.3f}s")
    for i, is_stop in enumerate(stop_events):
        if is_stop:
            print(f"Stop detected at timestamp: {i / fps:.3f}s")
    rows: list[FrameData] = []
    for i in range(len(frames)):
        x, y = cursor_positions[i]
        #if click event if detected, record {timestamp, x position, y position}
        #otherwise record False as default
        click_metric: bool | MouseClick = False
        if i < len(click_events) and click_events[i]:
            click_metric = MouseClick(
                timestamp=i / fps,
                x_pos=float("nan") if x is None else float(x),
                y_pos=float("nan") if y is None else float(y),
            )
        row = FrameData(
            frame=i,
            timestamp=i / fps,
            cursor_x=float("nan") if x is None else float(x),
            cursor_y=float("nan") if y is None else float(y),
            cursor_match_score=float(scores[i]),
            mouse_click_event=click_metric,
            mouse_drag_event=MouseDrag(drag=False, start_pos=0.0, end_pos=0.0),
            vel_x=float(vel_x[i]) if i < len(vel_x) else 0.0,
            vel_y=float(vel_y[i]) if i < len(vel_y) else 0.0,
            speed=float(speed[i]) if i < len(speed) else 0.0,
            acceleration=float(acceleration[i]) if i < len(acceleration) else 0.0,
            scene_change_score=0.0,
            mag_pixel_change=0.0,
            nearest_target_objects=[],
            dist_cursor_to_target=0.0,
            in_target_zone=False,
            ui_change_score=float(ui_change_scores[i]) if i < len(ui_change_scores) else 0.0,
        )
        rows.append(row)

    df = pd.DataFrame(rows)

    