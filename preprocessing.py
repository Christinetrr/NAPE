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
    halo_reference_path: str = "click_event.png",
    inner_radius_px: int = 10,
    outer_radius_px: int = 24,
    patch_radius_px: int = 30,
    z_threshold: float = 1.25,
    min_absolute_score: float = 0.04,
    min_separation_frames: int = 4,
) -> list[bool]:
    n = min(len(frames), len(positions))
    if n == 0:
        return []

    lower_hsv, upper_hsv = _estimate_blue_halo_hsv_bounds(halo_reference_path)
    scores = np.full(n, np.nan, dtype=np.float32)

    inner_r2 = float(inner_radius_px * inner_radius_px)
    outer_r2 = float(outer_radius_px * outer_radius_px)

    for i in range(n):
        x, y = positions[i]
        if x is None or y is None:
            continue

        frame = frames[i]
        if frame is None or frame.size == 0:
            continue

        h, w = frame.shape[:2]
        cx = int(np.clip(int(x), 0, w - 1))
        cy = int(np.clip(int(y), 0, h - 1))

        x0 = max(0, cx - patch_radius_px)
        x1 = min(w, cx + patch_radius_px + 1)
        y0 = max(0, cy - patch_radius_px)
        y1 = min(h, cy + patch_radius_px + 1)
        patch = frame[y0:y1, x0:x1]
        if patch.size == 0:
            continue

        patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(patch_hsv, lower_hsv, upper_hsv)  # bit mask for blue halo

        # Build geometric masks centered on cursor for ring vs center.
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dx = xx.astype(np.float32) - float(cx)
        dy = yy.astype(np.float32) - float(cy)
        d2 = dx * dx + dy * dy

        ring_mask = (d2 > inner_r2) & (d2 <= outer_r2)
        center_mask = d2 <= inner_r2
        if not np.any(ring_mask):
            continue

        # Convert bool masks to uint8 for bitwise operations.
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

        # Favor frames where blue is concentrated in annulus (halo) not center (cursor body).
        scores[i] = ring_ratio - 0.5 * center_ratio

    valid = np.isfinite(scores)
    if not np.any(valid):
        return [False] * n

    valid_scores = scores[valid]
    baseline = float(np.median(valid_scores))
    spread = float(np.std(valid_scores))
    threshold = max(
        min_absolute_score,
        baseline + float(z_threshold) * spread,
    )

    # Peak-picking: frame i is a click if it's a local maximum over threshold.
    peaks = np.zeros(n, dtype=bool)
    for i in range(1, n - 1):
        if not valid[i]:
            continue
        left = float(scores[i - 1]) if valid[i - 1] else -np.inf
        right = float(scores[i + 1]) if valid[i + 1] else -np.inf
        s = float(scores[i])
        if s >= threshold and s >= left and s > right:
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

    # Enforce minimum spacing between click frames.
    click_flags = np.zeros(n, dtype=bool)
    if min_separation_frames > 0:
        last_kept = -10_000
        for i in range(n):
            if peaks[i] and (i - last_kept) >= int(min_separation_frames):
                click_flags[i] = True
                last_kept = i
    else:
        click_flags = peaks

    return click_flags.tolist()


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

        idx = 0
        if early_seconds is not None:
            early_end_idx = int(early_seconds * fps)
        while True:
            ret, frame = cap.read()
            if not ret or idx >= len(positions):
                break

            x, y = positions[idx]

            if x is not None and y is not None:
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
                cx, cy = int(x) + ox_off, int(y) + oy_off
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

    #do the cursor overlay
    visualize_cursor_positions(
        video,
        cursor_positions,
        out_path="cursor_debug.mp4",
        early_seconds=1,
        normal_radius=14,
        early_radius=35,
        overlay_path="mac_cursor.png",
        overlay_hotspot_xy=(0, 0),
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

    #retrieve fps from video
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    fps = float(fps) if fps and fps > 0 else 30.0

    
    #BUILD DATAFRAME OF FRAME METRICS
    ui_change_scores = compute_ui_change_scores(frames, cursor_positions)
    click_events = detect_click_events(frames, cursor_positions, halo_reference_path="click_event.png")
    for i, is_click in enumerate(click_events):
        if is_click:
            print(f"Click detected at timestamp: {i / fps:.3f}s")
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
            vel_x=0.0,
            vel_y=0.0,
            speed=0.0,
            acceleration=0.0,
            scene_change_score=0.0,
            mag_pixel_change=0.0,
            nearest_target_objects=[],
            dist_cursor_to_target=0.0,
            in_target_zone=False,
            ui_change_score=float(ui_change_scores[i]) if i < len(ui_change_scores) else 0.0,
        )
        rows.append(row)

    df = pd.DataFrame(rows)

    