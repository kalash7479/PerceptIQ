#!/usr/bin/env python3
import sys
import os
import subprocess
import platform

def _pip(*args, show=False):
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", *args]
    if show:
        cmd = [sys.executable, "-m", "pip", "install", *args]
        subprocess.check_call(cmd)
    else:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _ensure_packages():
    print("=" * 60)
    print("  REAL-TIME 3D DASHCAM PERCEPTION SYSTEM")
    print("  Auto-checking dependencies...")
    print("=" * 60)

    for mod, pkg in [("numpy", "numpy==1.24.4"), ("PIL", "Pillow"),
                     ("tqdm", "tqdm"), ("scipy", "scipy")]:
        try:
            __import__(mod)
        except ImportError:
            print(f"[AUTO-INSTALL] {pkg}")
            _pip(pkg)

  
    _TORCH_VER = "2.2.0"
    _TV_VER    = "0.17.0"
    _IDX       = "https://download.pytorch.org/whl/cpu"

    needs_torch = False
    try:
        import torch as _t
        import torchvision as _tv
        tv = tuple(int(x) for x in _tv.__version__.split("+")[0].split(".")[:3])
        tk = tuple(int(x) for x in  _t.__version__.split("+")[0].split(".")[:3])
        # torchvision 0.17.x requires torch 2.2.x
        if tk[:2] != (2, 2) or tv[:2] != (0, 17):
            needs_torch = True
            print(f"[AUTO-INSTALL] torch/torchvision version mismatch "
                  f"(torch={_t.__version__}, tv={_tv.__version__}) — fixing...")
    except Exception:
        needs_torch = True
        print("[AUTO-INSTALL] torch + torchvision not found — installing...")

    if needs_torch:
        subprocess.call(
            [sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print(f"[AUTO-INSTALL] Installing torch=={_TORCH_VER} + torchvision=={_TV_VER} "
              f"(CPU, ~200 MB) — please wait...")
        _pip(f"torch=={_TORCH_VER}", f"torchvision=={_TV_VER}",
             "--index-url", _IDX, show=True)
        print("[AUTO-INSTALL] torch + torchvision installed ✓")

    try:
        import transformers as _tr
        ver = tuple(int(x) for x in _tr.__version__.split(".")[:2])
        if ver >= (5, 0) or ver < (4, 0):
            raise ImportError("incompatible version")
    except Exception:
        print("[AUTO-INSTALL] transformers==4.40.2 (compatible with torch 2.12.0)")
        _pip("transformers==4.40.2")

    for mod, pkg in [("timm", "timm"), ("accelerate", "accelerate")]:
        try:
            __import__(mod)
        except ImportError:
            print(f"[AUTO-INSTALL] {pkg}")
            _pip(pkg)

    # ── 4. CV / detection libs ────────────────────────────────────────────
    for mod, pkg in [("cv2", "opencv-python"),
                     ("ultralytics", "ultralytics"),
                     ("supervision", "supervision"),
                     ("ffmpeg", "ffmpeg-python")]:
        try:
            __import__(mod)
        except ImportError:
            print(f"[AUTO-INSTALL] {pkg}")
            _pip(pkg)

    print("[AUTO-INSTALL] All Python dependencies satisfied ✓")


def _ensure_ffmpeg():
    """Ensure the ffmpeg *binary* is on PATH."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        pass

    sys_name = platform.system().lower()
    print("[AUTO-INSTALL] ffmpeg binary not found — attempting install...")

    if sys_name == "windows":
        # Try winget first
        try:
            subprocess.check_call(
                ["winget", "install", "--id", "Gyan.FFmpeg", "-e", "--silent"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            print("[AUTO-INSTALL] ffmpeg installed via winget ✓  "
                  "(restart terminal if ffmpeg isn't found next run)")
            return True
        except Exception:
            pass
        # Try static download
        try:
            import urllib.request, zipfile, shutil, tempfile
            url = ("https://github.com/BtbN/FFmpeg-Builds/releases/download/"
                   "latest/ffmpeg-master-latest-win64-gpl.zip")
            print("[AUTO-INSTALL] Downloading ffmpeg static build (~90 MB)…")
            tmp = tempfile.mkdtemp()
            zp  = os.path.join(tmp, "ffmpeg.zip")
            urllib.request.urlretrieve(url, zp)
            with zipfile.ZipFile(zp) as z:
                z.extractall(tmp)
            for root, _, files in os.walk(tmp):
                if "ffmpeg.exe" in files:
                    dest = os.path.join(os.path.dirname(sys.executable), "ffmpeg.exe")
                    shutil.copy(os.path.join(root, "ffmpeg.exe"), dest)
                    print(f"[AUTO-INSTALL] ffmpeg.exe → {dest} ✓")
                    return True
        except Exception as e:
            print(f"[WARN] Could not auto-install ffmpeg: {e}")
        print("[WARN] Install ffmpeg manually: https://ffmpeg.org/download.html")
        return False

    elif sys_name == "linux":
        try:
            subprocess.check_call(["apt-get", "install", "-y", "ffmpeg"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            print("[WARN] Run: sudo apt-get install ffmpeg")
            return False

    elif sys_name == "darwin":
        try:
            subprocess.check_call(["brew", "install", "ffmpeg"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            print("[WARN] Run: brew install ffmpeg")
            return False

    return False


_ensure_packages()
_FFMPEG_OK = _ensure_ffmpeg()


import argparse
import time
import math
import collections
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from tqdm import tqdm
import torch
from ultralytics import YOLO
import supervision as sv


VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 1: "bicycle"}
RISK_COLORS = {
    "SAFE":    (0, 255, 80),
    "CAUTION": (0, 220, 255),
    "WARNING": (0, 140, 255),
    "DANGER":  (0, 0, 255),
}

OUTPUT_W = 1920
OUTPUT_H = 540
PANEL_W  = OUTPUT_W // 2
PANEL_H  = OUTPUT_H

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║          Real-Time 3D Dashcam Perception System v1.2             ║
║          YOLO · DepthAnything V2 · BEV · Collision Risk          ║
╚══════════════════════════════════════════════════════════════════╝
"""


def _build_turbo_lut() -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        if t < 0.25:
            r, g, b = 0, int(t * 4 * 255), 255
        elif t < 0.5:
            r, g, b = 0, 255, int((1 - (t - 0.25) * 4) * 255)
        elif t < 0.75:
            r, g, b = int((t - 0.5) * 4 * 255), 255, 0
        else:
            r, g, b = 255, int((1 - (t - 0.75) * 4) * 255), 0
        lut[i] = [b, g, r]  # BGR
    return lut

_TURBO_LUT = _build_turbo_lut()


def depth_to_color(depth_norm: np.ndarray) -> np.ndarray:
    idx = (np.clip(depth_norm, 0, 1) * 255).astype(np.uint8)
    return _TURBO_LUT[idx]


class DepthEstimator:
    def __init__(self, device: str):
        self.device      = device
        self._prev       = None
        self._alpha      = 0.6
        self._pipe       = None

    def load(self):
        print("[DEPTH] Loading Depth Anything V2 Small …")
        from transformers import pipeline as hf_pipeline
        hf_dev = 0 if (self.device == "cuda" and torch.cuda.is_available()) else -1
        self._pipe = hf_pipeline(
            task="depth-estimation",
            model=DEPTH_MODEL,
            device=hf_dev,
        )
        print("[DEPTH] Model ready ✓")

    def estimate(self, bgr: np.ndarray) -> np.ndarray:
        from PIL import Image as PILImage
        rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = self._pipe(PILImage.fromarray(rgb))
        depth  = np.array(result["depth"], dtype=np.float32)
        depth  = cv2.resize(depth, (bgr.shape[1], bgr.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
        mn, mx = depth.min(), depth.max()
        if mx - mn > 1e-5:
            depth = (depth - mn) / (mx - mn)
        if self._prev is not None and self._prev.shape == depth.shape:
            depth = self._alpha * self._prev + (1 - self._alpha) * depth
        self._prev = depth.copy()
        return depth

class ObjectTracker:
    def __init__(self, max_history=30):
        self.tracker       = sv.ByteTrack()
        self.trajectories  = collections.defaultdict(
            lambda: collections.deque(maxlen=max_history))
        self.velocities    = collections.defaultdict(lambda: (0.0, 0.0))
        self._prev_centers = {}

    def update(self, detections: sv.Detections) -> sv.Detections:
        tracked = self.tracker.update_with_detections(detections)
        if tracked.tracker_id is None:
            return tracked
        for i, tid in enumerate(tracked.tracker_id):
            box = tracked.xyxy[i]
            cx  = int((box[0] + box[2]) / 2)
            cy  = int((box[1] + box[3]) / 2)
            self.trajectories[tid].append((cx, cy))
            if tid in self._prev_centers:
                px, py = self._prev_centers[tid]
                self.velocities[tid] = (cx - px, cy - py)
            self._prev_centers[tid] = (cx, cy)
        return tracked


FOCAL_LEN    = 700.0
REAL_CAR_W   = 2.0
BEV_Z_FAR    = 60.0

def estimate_distance(depth_map, box, fw, fh) -> float:
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(fw - 1, x2), min(fh - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return 50.0
    patch   = depth_map[y1:y2, x1:x2]
    med     = float(np.median(patch))
    d_depth = 1.0 + med * 79.0
    px_w    = max(x2 - x1, 1)
    d_proj  = (FOCAL_LEN * REAL_CAR_W) / px_w
    return round(float(np.clip(0.5 * d_depth + 0.5 * d_proj, 1.0, 100.0)), 1)

def get_risk(dist: float, box, fw: int) -> str:
    cx        = (box[0] + box[2]) / 2
    lat_off   = abs(cx - fw / 2) / (fw / 2)
    if dist < 8 and lat_off < 0.4:  return "DANGER"
    if dist < 15:                    return "WARNING"
    if dist < 25:                    return "CAUTION"
    return "SAFE"

def _draw_lane(frame):
    h, w = frame.shape[:2]
    pts  = np.array([
        [int(w * 0.35), int(h * 0.55)],
        [int(w * 0.65), int(h * 0.55)],
        [int(w * 0.85), h - 1],
        [int(w * 0.15), h - 1],
    ], dtype=np.int32)
    ov = frame.copy()
    cv2.fillPoly(ov, [pts], (0, 80, 0))
    cv2.addWeighted(ov, 0.18, frame, 0.82, 0, frame)
    cv2.polylines(frame, [pts], True, (0, 255, 60), 1, cv2.LINE_AA)


def _glow_box(frame, x1, y1, x2, y2, color, thickness=2):
    for t, a in [(thickness + 4, 0.15), (thickness + 2, 0.30), (thickness, 1.0)]:
        c = tuple(int(v * a) for v in color)
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, t, cv2.LINE_AA)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def draw_left_panel(frame, tracked, depth_map, tracker: ObjectTracker):
    out = frame.copy()
    _draw_lane(out)
    h, w = out.shape[:2]

    if tracked is not None and len(tracked) > 0:
        for i in range(len(tracked.xyxy)):
            box   = tracked.xyxy[i].astype(int)
            cls   = int(tracked.class_id[i])   if tracked.class_id   is not None else 2
            tid   = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else 0
            label = VEHICLE_CLASSES.get(cls, "vehicle")
            dist  = estimate_distance(depth_map, box, w, h)
            risk  = get_risk(dist, box, w)
            color = RISK_COLORS[risk]

            _glow_box(out, box[0], box[1], box[2], box[3], color)

            txt = f"{label} {dist:.0f}m {risk}"
            fs  = 0.48
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, fs, 1)
            lx, ly = box[0], max(box[1] - 6, th + 4)
            cv2.rectangle(out, (lx - 2, ly - th - 4), (lx + tw + 4, ly + 2), (0,0,0), -1)
            cv2.putText(out, txt, (lx, ly), cv2.FONT_HERSHEY_DUPLEX, fs, color, 1, cv2.LINE_AA)

            traj = list(tracker.trajectories[tid])
            for k in range(1, len(traj)):
                cv2.line(out, traj[k - 1], traj[k], color, 1, cv2.LINE_AA)

            vx, vy = tracker.velocities.get(tid, (0, 0))
            cx_b   = (box[0] + box[2]) // 2
            cy_b   = (box[1] + box[3]) // 2
            cv2.arrowedLine(out, (cx_b, cy_b),
                            (cx_b + int(vx * 8), cy_b + int(vy * 8)),
                            (255, 255, 0), 1, cv2.LINE_AA, tipLength=0.4)

    cv2.putText(out, "DASHCAM · ADAS PERCEPTION", (10, 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 80), 1, cv2.LINE_AA)
    return out


def _draw_grid(canvas, pw, ph):
    for frac in np.linspace(0.3, 1.0, 10):
        y      = int(frac * ph)
        xl     = int(pw * 0.5 - pw * 0.4 * frac)
        xr     = int(pw * 0.5 + pw * 0.4 * frac)
        alpha  = int(80 + 120 * frac)
        cv2.line(canvas, (xl, y), (xr, y), (alpha, 0, alpha // 2), 1, cv2.LINE_AA)
    vp_x, vp_y = pw // 2, int(ph * 0.28)
    for k in np.linspace(-0.42, 0.42, 16):
        bx    = int(pw * 0.5 + k * pw)
        alpha = int(60 + 80 * (1 - abs(k) / 0.45))
        cv2.line(canvas, (vp_x, vp_y), (bx, ph), (0, alpha, alpha // 2), 1, cv2.LINE_AA)


def _project_pointcloud(depth_norm, canvas, step=3):
    h, w   = depth_norm.shape
    ch, cw = canvas.shape[:2]
    ys     = np.arange(h // 2, h, step)
    xs     = np.arange(0, w, step)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    d      = depth_norm[yy, xx].astype(np.float32)
    u      = xx.astype(np.float32) / w
    bev_y  = (ch * 0.28 + (1.0 - d) * ch * 0.70).astype(np.int32)
    spread = 0.38 + 0.52 * d
    bev_x  = (cw * 0.5 + (u - 0.5) * cw * spread).astype(np.int32)
    mask   = (bev_x >= 0) & (bev_x < cw) & (bev_y >= 0) & (bev_y < ch)
    canvas[bev_y[mask].ravel(), bev_x[mask].ravel()] = depth_to_color(d[mask].ravel())


def _bev_box(canvas, cx_norm, dist_m, label, color, pw, ph):
    d_norm = float(np.clip(1.0 - dist_m / BEV_Z_FAR, 0, 1))
    bev_y  = int(ph * 0.28 + (1.0 - d_norm) * ph * 0.70)
    spread = 0.38 + 0.52 * d_norm
    bev_x  = int(pw * 0.5 + (cx_norm - 0.5) * pw * spread)
    scale  = 0.5 + 1.5 * (1.0 - d_norm)
    bw     = int(55 * scale); bh = int(35 * scale); bd = int(20 * scale)
    fl, fr = bev_x - bw // 2, bev_x + bw // 2
    ft, fb = bev_y - bh // 2, bev_y + bh // 2
    bl, br = fl + bd, fr + bd
    bt, bb = ft - bd // 2, fb - bd // 2

    def gl(p1, p2):
        for t, a in [(3, 0.2), (2, 0.5), (1, 1.0)]:
            cv2.line(canvas, p1, p2, tuple(int(v * a) for v in color), t, cv2.LINE_AA)

    gl((fl,ft),(fr,ft)); gl((fr,ft),(fr,fb)); gl((fr,fb),(fl,fb)); gl((fl,fb),(fl,ft))
    gl((bl,bt),(br,bt)); gl((br,bt),(br,bb)); gl((br,bb),(bl,bb)); gl((bl,bb),(bl,bt))
    for p1, p2 in [((fl,ft),(bl,bt)),((fr,ft),(br,bt)),((fr,fb),(br,bb)),((fl,fb),(bl,bb))]:
        gl(p1, p2)

    fs = 0.38 * (0.6 + 0.8 * scale)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fs, 1)
    tx, ty = bev_x - tw // 2, ft - 6
    if 0 < ty < ph:
        cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, fs, color, 1, cv2.LINE_AA)


def draw_right_panel(depth_norm, tracked, frame_w, frame_h):
    canvas  = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    d_small = cv2.resize(depth_norm, (PANEL_W, PANEL_H))
    _project_pointcloud(d_small, canvas)

    # Horizon fade
    horizon = int(PANEL_H * 0.3)
    grad    = np.linspace(200, 0, horizon, dtype=np.uint8)
    canvas[:horizon, :, 0] = np.minimum(
        canvas[:horizon, :, 0].astype(np.int16) + grad[:, None] // 4, 255
    ).astype(np.uint8)

    _draw_grid(canvas, PANEL_W, PANEL_H)

    if tracked is not None and len(tracked) > 0 and tracked.tracker_id is not None:
        d_full = cv2.resize(depth_norm, (frame_w, frame_h))
        for i in range(len(tracked.xyxy)):
            box      = tracked.xyxy[i]
            cls      = int(tracked.class_id[i]) if tracked.class_id is not None else 2
            name     = VEHICLE_CLASSES.get(cls, "vehicle")
            dist     = estimate_distance(d_full, box, frame_w, frame_h)
            risk     = get_risk(dist, box, frame_w)
            color    = RISK_COLORS[risk]
            cx_norm  = ((box[0] + box[2]) / 2) / frame_w
            _bev_box(canvas, cx_norm, dist, f"{name} {dist:.0f}m {risk}",
                     color, PANEL_W, PANEL_H)

    cv2.putText(canvas, "3D BEV · DEPTH PERCEPTION", (10, 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)
    return canvas


def compose(left, right):
    l   = cv2.resize(left,  (PANEL_W, PANEL_H))
    r   = cv2.resize(right, (PANEL_W, PANEL_H))
    div = np.full((PANEL_H, 4, 3), (0, 255, 80), dtype=np.uint8)
    return np.hstack([l, div, r])


def add_hud(frame, fps, frame_no, total, device_label):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 28), (w, h), (5, 5, 5), -1)
    pct = frame_no / max(total, 1) * 100
    txt = (f"  FPS: {fps:5.1f}  |  Frame: {frame_no}/{total} ({pct:.1f}%)  "
           f"|  Device: {device_label}  |  3D-PERC v1.2")
    cv2.putText(frame, txt, (10, h - 8), cv2.FONT_HERSHEY_DUPLEX, 0.42,
                (0, 220, 80), 1, cv2.LINE_AA)


def process_video(input_path: str, args):
    print(BANNER)

    # Device
    if args.device == "cuda" and torch.cuda.is_available():
        device    = "cuda"
        dev_label = f"CUDA · {torch.cuda.get_device_name(0)}"
    else:
        device    = "cpu"
        dev_label = "CPU"
    print(f"[INFO] Device: {dev_label}")

    # Paths
    os.makedirs("./output", exist_ok=True)
    tmp_out   = "./output/_tmp_raw.mp4"
    final_out = "./output/final_output.mp4"

    # Open input
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {input_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Input:  {input_path}  {src_w}×{src_h}  {src_fps:.1f}fps  {total_frames} frames")
    print(f"[INFO] Output: {final_out}")

    # Load models
    print("[YOLO] Loading YOLOv8n …")
    yolo = YOLO("yolov8n.pt")
    yolo.to(device)

    depth_est = DepthEstimator(device)
    depth_est.load()

    tracker = ObjectTracker()

    # Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_out, fourcc, src_fps, (OUTPUT_W + 4, OUTPUT_H))

    # Processing loop
    pbar     = tqdm(total=total_frames, desc="Processing", unit="fr")
    t0       = time.time()
    frame_no = 0
    fps_avg  = 0.0

    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        frame_no += 1

        # Resize to working resolution
        work_w = min(src_w, 640)
        work_h = int(src_h * (work_w / src_w))
        work   = cv2.resize(bgr, (work_w, work_h))

        # YOLO detection
        results = yolo(work, conf=args.conf,
                       classes=list(VEHICLE_CLASSES.keys()),
                       verbose=False, device=device)
        sv_det  = sv.Detections.from_ultralytics(results[0])
        tracked = tracker.update(sv_det)

        # Depth estimation (on smaller input for speed)
        depth_input = cv2.resize(work, (320, 192))
        depth_norm  = depth_est.estimate(depth_input)
        depth_norm  = cv2.resize(depth_norm, (work_w, work_h))

        # Build panels
        left_panel  = draw_left_panel(work, tracked, depth_norm, tracker)
        right_panel = draw_right_panel(depth_norm, tracked, work_w, work_h)
        composed    = compose(left_panel, right_panel)

        # FPS
        elapsed = time.time() - t0
        fps_now = frame_no / max(elapsed, 1e-6)
        fps_avg = 0.9 * fps_avg + 0.1 * fps_now if fps_avg > 0 else fps_now

        add_hud(composed, fps_avg, frame_no, total_frames, dev_label)
        writer.write(composed)

        if args.show:
            preview = cv2.resize(composed, (1280, int(PANEL_H * 1280 / OUTPUT_W)))
            cv2.imshow("3D Dashcam Perception", preview)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] User quit.")
                break

        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    total_elapsed = time.time() - t0
    print(f"\n[INFO] Done: {frame_no} frames in {total_elapsed:.1f}s  "
          f"({frame_no / max(total_elapsed, 1e-6):.1f} fps avg)")

    if _FFMPEG_OK:
        # Re-check ffmpeg is actually on PATH after possible winget install
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            ffmpeg_available = True
        except Exception:
            ffmpeg_available = False

        if ffmpeg_available:
            print(f"[FFMPEG] Compressing → {final_out} …")
            try:
                cmd = [
                    "ffmpeg", "-y", "-i", tmp_out,
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    final_out,
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                if os.path.exists(tmp_out):
                    os.remove(tmp_out)
                size_mb = os.path.getsize(final_out) / (1024 ** 2)
                print(f"\n✅  OUTPUT SAVED: {final_out}  ({size_mb:.1f} MB)")
                return
            except subprocess.CalledProcessError as e:
                print(f"[WARN] FFmpeg failed: {e.stderr.decode()[:300]}")

    import shutil
    shutil.move(tmp_out, final_out)
    size_mb = os.path.getsize(final_out) / (1024 ** 2)
    print(f"\n✅  OUTPUT SAVED (uncompressed): {final_out}  ({size_mb:.1f} MB)")
    print("[INFO] Install ffmpeg for smaller files: https://ffmpeg.org/download.html")

def parse_args():
    p = argparse.ArgumentParser(
        description="Real-Time 3D Dashcam Perception System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input",    type=str,   help="Path to input dashcam MP4")
    p.add_argument("--conf",   type=float, default=0.35, help="YOLO confidence threshold")
    p.add_argument("--device", type=str,   default="cpu", choices=["cuda", "cpu"])
    p.add_argument("--show",   action="store_true", help="Live preview window")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.isfile(args.input):
        print(f"[ERROR] File not found: {args.input}")
        sys.exit(1)
    process_video(args.input, args)
