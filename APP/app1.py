"""
Streamlit Web App: AI Smart Traffic Management System (Authority Dashboard)
- Replaces Tkinter UI from `f.py` with a modern Streamlit dashboard
- Preserves features: adaptive green timing, yellow phase, round-robin fairness,
  manual overrides (force green per lane), skip next, emergency stop/resume,
  Peak Hours mode, YOLOv8-based vehicle detection, congestion metrics, logging.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import time
import threading
from datetime import datetime
from typing import List, Dict, Tuple

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO


# ----------------------------- CONFIG ---------------------------------
MIN_GREEN_TIME_DEFAULT = 5
MAX_GREEN_TIME_DEFAULT = 20
YELLOW_TIME_DEFAULT = 3

PEAK_MIN_GREEN = 8
PEAK_MAX_GREEN = 30

VIDEO_RESIZE: Tuple[int, int] = (380, 214)
MODEL_PATH = "yolov8n.pt"
LOG_FILE = "traffic_log.txt"

DIRECTIONS = ["North", "East", "South", "West"]
VEHICLE_COCO_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# UI colors (align roughly with f.py)
PRIMARY_BG = "#2C3E50"
SECONDARY_BG = "#34495E"
ACCENT_COLOR = "#2980B9"
TEXT_COLOR = "#ECF0F1"
DANGER_COLOR = "#E74C3C"
WARNING_COLOR = "#F39C12"
SUCCESS_COLOR = "#2ECC71"
INFO_COLOR = "#3498DB"


# ---------------------------- UTILITIES -------------------------------
def log_event(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
    except Exception as exc:
        # For Streamlit debugging
        print(f"Log write failed: {exc}")


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    try:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception:
        return image


# ------------------------- BACKGROUND WORKER ---------------------------
class TrafficEngine(threading.Thread):
    """
    Background worker that:
    - Reads 4 video feeds
    - Runs YOLO detections and stores processed frames + counts
    - Manages the traffic signal state machine and timers
    """

    def __init__(self, video_files: List[str], state: Dict):
        super().__init__(daemon=True)
        self.video_files = video_files
        self.state = state
        self._running = threading.Event()
        self._running.set()

        # Internal helpers
        self._lock = state["lock"]
        self._last_timer_tick = time.time()

        # Initialize video captures
        self.caps = []
        for path in self.video_files:
            full_path = os.path.join(os.getcwd(), path)
            cap = cv2.VideoCapture(full_path)
            if not cap.isOpened():
                log_event(f"Video open failed: {full_path}")
            self.caps.append(cap)

        # Load model once
        try:
            self.model = YOLO(MODEL_PATH)
        except Exception as exc:
            log_event(f"Failed to load YOLO model: {exc}")
            raise

    def stop(self):
        self._running.clear()

    def _read_and_detect(self) -> Tuple[List[np.ndarray], List[int]]:
        frames_out: List[np.ndarray] = []
        counts: List[int] = []
        for cap in self.caps:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            if not ret:
                # Build a NO SIGNAL frame
                err = np.zeros((VIDEO_RESIZE[1], VIDEO_RESIZE[0], 3), np.uint8)
                cv2.putText(err, "NO SIGNAL", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                frames_out.append(err)
                counts.append(0)
                continue

            try:
                results = self.model(frame, verbose=False, classes=VEHICLE_COCO_CLASSES)
                rendered = results[0].plot(labels=False, conf=False, boxes=True)
                frames_out.append(rendered)
                counts.append(len(results[0].boxes))
            except Exception as exc:
                log_event(f"YOLO processing error: {exc}")
                frames_out.append(frame)
                counts.append(0)

        return frames_out, counts

    def _tick_timer(self):
        now = time.time()
        if now - self._last_timer_tick < 1.0:  # Tick every 1 second for proper countdown
            return
        self._last_timer_tick = now

        with self._lock:
            if self.state.get("control_mode") == "EMERGENCY":
                return

            if int(self.state.get("remaining_time", 0)) > 0:
                self.state["remaining_time"] = int(self.state.get("remaining_time", 0)) - 1

            if int(self.state.get("remaining_time", 0)) <= 0:
                if bool(self.state.get("is_yellow_phase")):
                    # Transition out of yellow
                    self.state["is_yellow_phase"] = False

                    if self.state.get("control_mode") == "AUTO":
                        next_lane = (int(self.state.get("current_green_lane", -1)) + 1) % 4 if int(
                            self.state.get("current_green_lane", -1)
                        ) != -1 else 0
                        if bool(self.state.get("skip_request")):
                            log_event(f"AUTO: Skipping lane {next_lane}.")
                            next_lane = (next_lane + 1) % 4
                            self.state["skip_request"] = False
                        self.state["current_green_lane"] = next_lane

                    elif self.state.get("control_mode") == "MANUAL" and self.state.get("manual_override_request") is not None:
                        self.state["current_green_lane"] = int(self.state.get("manual_override_request"))  # consume
                        self.state["manual_override_request"] = None

                    # Set green time based on vehicle count
                    self._calculate_and_set_green_time()

                else:
                    # End of green -> start yellow or boot system
                    if int(self.state.get("current_green_lane", -1)) != -1:
                        self.state["is_yellow_phase"] = True
                        self.state["remaining_time"] = int(self.state.get("yellow_time", YELLOW_TIME_DEFAULT))

                        if self.state.get("control_mode") == "MANUAL":
                            # Revert to AUTO after manual force cycle completes
                            log_event("Manual override complete. Reverting to AUTO mode.")
                            self.state["control_mode"] = "AUTO"
                    else:
                        # Initialization: set first lane
                        self.state["control_mode"] = "AUTO"
                        self.state["current_green_lane"] = 0
                        self.state["is_yellow_phase"] = False
                        self._calculate_and_set_green_time()
                        log_event("System initialized: Starting with North lane (0)")

    def _calculate_and_set_green_time(self):
        lane = int(self.state.get("current_green_lane", -1))
        v = int(self.state.get("vehicle_counts", [0, 0, 0, 0])[lane]) if 0 <= lane < 4 else 0
        min_g = int(self.state.get("min_green_time", MIN_GREEN_TIME_DEFAULT))
        max_g = int(self.state.get("max_green_time", MAX_GREEN_TIME_DEFAULT))
        computed = int(max(min_g, min(max_g, min_g + v * 1.5)))
        self.state["remaining_time"] = computed

    def _update_congestion(self):
        total = int(sum(self.state.get("vehicle_counts", [0, 0, 0, 0])))
        if total > 25:
            level, color = "High", DANGER_COLOR
        elif total > 15:
            level, color = "Medium", WARNING_COLOR
        else:
            level, color = "Low", SUCCESS_COLOR
        self.state["total_vehicles_str"] = str(total)
        self.state["congestion_level"] = level
        self.state["congestion_color"] = color

    def run(self):
        # Initialize system on first run
        with self._lock:
            if self.state.get("current_green_lane", -1) == -1:
                self.state["control_mode"] = "AUTO"
                self.state["current_green_lane"] = 0
                self.state["is_yellow_phase"] = False
                self._calculate_and_set_green_time()
                log_event("System initialized: Starting with North lane (0)")
        
        while self._running.is_set():
            # 1) Read feeds + detect
            frames, counts = self._read_and_detect()
            with self._lock:
                self.state["frames"] = [cv2.resize(f, VIDEO_RESIZE) for f in frames]
                self.state["vehicle_counts"] = counts
                self._update_congestion()

            # 2) Tick timer roughly once per second
            self._tick_timer()

            # 3) Throttle for UI
            time.sleep(0.1)  # Faster processing for more responsive timer

        # Cleanup
        for cap in self.caps:
            try:
                cap.release()
            except Exception:
                pass


# ------------------------------ APP -----------------------------------
def init_state():
    if "initialized" in st.session_state:
        return

    # UI flags (Streamlit session-only)
    st.session_state.is_peak_hours = False
    st.session_state.live_update = True
    st.session_state.last_refresh = time.time()

    # Engine handles its own state dict separate from Streamlit
    st.session_state.engine = None
    st.session_state.engine_state = None
    st.session_state.initialized = True

    log_event("System Initialized.")


def start_engine():
    if st.session_state.engine is not None:
        return
    # Use simple filenames, TrafficEngine will convert to absolute paths
    video_files = ["north.mp4", "east.mp4", "south.mp4", "west.mp4"]
    # Ensure files exist; Streamlit should still run, frames will show NO SIGNAL if missing
    missing = [p for p in video_files if not os.path.exists(p)]
    if missing:
        log_event(f"Missing video files: {missing}")
    # Build independent engine state (plain dict + lock)
    engine_state = {
        "lock": threading.Lock(),
        "min_green_time": MIN_GREEN_TIME_DEFAULT,
        "max_green_time": MAX_GREEN_TIME_DEFAULT,
        "yellow_time": YELLOW_TIME_DEFAULT,
        "off_peak_min": MIN_GREEN_TIME_DEFAULT,
        "off_peak_max": MAX_GREEN_TIME_DEFAULT,
        "vehicle_counts": [0, 0, 0, 0],
        "current_green_lane": 0,
        "remaining_time": 5,
        "is_yellow_phase": False,
        "control_mode": "AUTO",
        "manual_override_request": None,
        "skip_request": False,
        "frames": [np.zeros((VIDEO_RESIZE[1], VIDEO_RESIZE[0], 3), np.uint8) for _ in range(4)],
        "total_vehicles_str": "0",
        "congestion_level": "Low",
        "congestion_color": SUCCESS_COLOR,
    }
    try:
        engine = TrafficEngine(video_files, engine_state)
        engine.start()
        st.session_state.engine = engine
        st.session_state.engine_state = engine_state
    except Exception as exc:
        st.error(f"Failed to start engine: {exc}")
        log_event(f"Engine start failed: {exc}")


def stop_engine():
    eng = st.session_state.engine
    if eng is not None:
        try:
            eng.stop()
        except Exception:
            pass
        st.session_state.engine = None
        log_event("System Shutdown.")


def apply_force_green(lane_index: int):
    es = st.session_state.engine_state
    if es is None:
        return
    with es["lock"]:
        if es.get("control_mode") == "EMERGENCY":
            st.warning("System in EMERGENCY STOP. Resume first.")
            return
        log_event(f"MANUAL OVERRIDE: Forcing green for lane {lane_index}. Will revert to AUTO.")
        es["control_mode"] = "MANUAL"

        if int(es.get("current_green_lane", -1)) == lane_index and not bool(es.get("is_yellow_phase")):
            # Extend current lane using same compute rule
            lane = lane_index
            v = int(es.get("vehicle_counts", [0, 0, 0, 0])[lane])
            min_g = int(es.get("min_green_time", MIN_GREEN_TIME_DEFAULT))
            max_g = int(es.get("max_green_time", MAX_GREEN_TIME_DEFAULT))
            es["remaining_time"] = int(max(min_g, min(max_g, min_g + v * 1.5)))
            return

        es["manual_override_request"] = int(lane_index)
        if int(es.get("current_green_lane", -1)) != -1 and not bool(es.get("is_yellow_phase")):
            es["is_yellow_phase"] = True
            es["remaining_time"] = int(es.get("yellow_time", YELLOW_TIME_DEFAULT))
        else:
            es["is_yellow_phase"] = False
            es["remaining_time"] = 0


def apply_skip_next():
    es = st.session_state.engine_state
    if es is None:
        return
    with es["lock"]:
        if es.get("control_mode") == "EMERGENCY":
            st.warning("System in EMERGENCY STOP. Resume first.")
            return
        if es.get("control_mode") != "AUTO":
            st.info("Skip is only available in AUTO mode.")
            return
        if bool(es.get("is_yellow_phase")) or int(es.get("remaining_time", 0)) < 3:
            st.info("Cannot skip, lane change already in progress.")
            return
        log_event("MANUAL: Skip next green lane requested.")
        es["skip_request"] = True
        es["remaining_time"] = 0


def apply_emergency_stop():
    es = st.session_state.engine_state
    if es is None:
        return
    with es["lock"]:
        log_event("EMERGENCY STOP ACTIVATED.")
        es["control_mode"] = "EMERGENCY"
        es["current_green_lane"] = -1
        es["is_yellow_phase"] = False
        es["remaining_time"] = 0


def apply_resume_operation():
    es = st.session_state.engine_state
    if es is None:
        return
    with es["lock"]:
        log_event("EMERGENCY state cleared. Resuming normal operation.")
        es["control_mode"] = "AUTO"
        if int(es.get("current_green_lane", -1)) == -1:
            es["remaining_time"] = 0


def toggle_peak_hours(enabled: bool):
    es = st.session_state.engine_state
    if es is None:
        return
    with es["lock"]:
        if enabled:
            log_event("Peak Hours Mode ENABLED.")
            es["off_peak_min"] = int(es.get("min_green_time", MIN_GREEN_TIME_DEFAULT))
            es["off_peak_max"] = int(es.get("max_green_time", MAX_GREEN_TIME_DEFAULT))
            es["min_green_time"] = PEAK_MIN_GREEN
            es["max_green_time"] = PEAK_MAX_GREEN
        else:
            log_event("Peak Hours Mode DISABLED.")
            es["min_green_time"] = int(es.get("off_peak_min", MIN_GREEN_TIME_DEFAULT))
            es["max_green_time"] = int(es.get("off_peak_max", MAX_GREEN_TIME_DEFAULT))


def status_badge(text: str, color_hex: str) -> str:
    return f"""
<div style='background:{SECONDARY_BG}; padding:6px 10px; border-radius:6px; display:inline-block; color:{color_hex}; font-weight:700;'>
  {text}
</div>
"""


def render_signals(es: Dict) -> str:
    lane_names = DIRECTIONS
    cur_lane = int(es.get("current_green_lane", -1))
    is_yellow = bool(es.get("is_yellow_phase"))
    mode = es.get("control_mode", "AUTO")

    def color_for(idx: int) -> str:
        if mode == "EMERGENCY":
            return DANGER_COLOR
        if cur_lane == -1:
            return DANGER_COLOR
        if idx == cur_lane:
            return WARNING_COLOR if is_yellow else SUCCESS_COLOR
        return DANGER_COLOR

    cells = []
    for i, name in enumerate(lane_names):
        col = color_for(i)
        cells.append(
            f"<div style='flex:1; min-width:140px; margin:6px; padding:12px; background:{SECONDARY_BG}; border:1px solid {ACCENT_COLOR}; border-radius:10px; text-align:center;'>"
            f"<div style='width:26px; height:26px; border-radius:50%; background:{col}; margin:0 auto 8px auto; border:2px solid #111'></div>"
            f"<div style='color:{TEXT_COLOR}; font-weight:700'>{name}</div>"
            "</div>"
        )

    return "<div style='display:flex; flex-wrap:wrap; gap:0px;'>" + "".join(cells) + "</div>"


def main():
    st.set_page_config(page_title="AI Smart Traffic Management", layout="wide")
    init_state()
    start_engine()

    # ---- Sidebar (Authority Dashboard controls) ----
    with st.sidebar:
        st.markdown("<h2 style='color:%s;'>AUTHORITY DASHBOARD</h2>" % TEXT_COLOR, unsafe_allow_html=True)

        # System Status
        st.markdown("<div style='color:%s; font-weight:700; font-size:18px;'>System Status</div>" % TEXT_COLOR,
                    unsafe_allow_html=True)

        with st.container():
            es = st.session_state.engine_state or {}
            mode = es.get("control_mode", "AUTO")
            mode_color = {"AUTO": INFO_COLOR, "MANUAL": WARNING_COLOR, "EMERGENCY": DANGER_COLOR}.get(mode,
                                                                                                      TEXT_COLOR)
            st.markdown(status_badge(f"Mode: {mode}", mode_color), unsafe_allow_html=True)

            cur_lane = int(es.get("current_green_lane", -1))
            active_lane_txt = "ALL STOPPED" if mode == "EMERGENCY" else (DIRECTIONS[cur_lane] if cur_lane != -1 else "N/A")
            active_color = SUCCESS_COLOR if not bool(es.get("is_yellow_phase")) and mode != "EMERGENCY" else WARNING_COLOR
            st.markdown(status_badge(f"Active Lane: {active_lane_txt}", active_color), unsafe_allow_html=True)

            st.markdown("---")
            st.markdown(status_badge(f"Total Vehicles: {es.get('total_vehicles_str', '0')}", TEXT_COLOR),
                        unsafe_allow_html=True)
            st.markdown(status_badge(f"Congestion: {es.get('congestion_level', 'Low')}", es.get('congestion_color',
                                                                                                 SUCCESS_COLOR)),
                        unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Manual Intervention
        st.markdown("<div style='color:%s; font-weight:700; font-size:18px;'>Manual Intervention</div>" % TEXT_COLOR,
                    unsafe_allow_html=True)
        with st.container():
            cols = st.columns(2)
            if cols[0].button("FORCE GREEN NORTH"):
                apply_force_green(0)
            if cols[1].button("FORCE GREEN EAST"):
                apply_force_green(1)
            if cols[0].button("FORCE GREEN SOUTH"):
                apply_force_green(2)
            if cols[1].button("FORCE GREEN WEST"):
                apply_force_green(3)

            st.markdown("---")
            if st.button("SKIP NEXT GREEN"):
                apply_skip_next()

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # System Control
        st.markdown("<div style='color:%s; font-weight:700; font-size:18px;'>System Control</div>" % TEXT_COLOR,
                    unsafe_allow_html=True)
        with st.container():
            peak_enabled = st.checkbox("Enable Peak Hours Mode", value=bool(st.session_state.is_peak_hours))
            if peak_enabled != st.session_state.is_peak_hours:
                st.session_state.is_peak_hours = peak_enabled
                toggle_peak_hours(peak_enabled)

            cols = st.columns(2)
            if (st.session_state.engine_state or {}).get("control_mode", "AUTO") != "EMERGENCY":
                if cols[0].button("🚨 EMERGENCY STOP 🚨"):
                    apply_emergency_stop()
            else:
                if cols[0].button("✅ RESUME OPERATION"):
                    apply_resume_operation()

            # Live update toggle (use checkbox)
            live = st.checkbox("Live Update", value=bool(st.session_state.get("live_update", True)))
            st.session_state.live_update = bool(live)
            
            # Manual refresh button
            if st.button("🔄 Refresh Dashboard"):
                st.rerun()

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Timing Parameters
        st.markdown("<div style='color:%s; font-weight:700; font-size:18px;'>Timing Parameters</div>" % TEXT_COLOR,
                    unsafe_allow_html=True)
        with st.container():
            disabled = bool(st.session_state.is_peak_hours)
            es2 = st.session_state.engine_state or {}
            min_val = st.slider("Min Green (s)", 2, 10, int(es2.get("min_green_time", MIN_GREEN_TIME_DEFAULT)),
                                disabled=disabled)
            max_val = st.slider("Max Green (s)", 15, 45, int(es2.get("max_green_time", MAX_GREEN_TIME_DEFAULT)),
                                disabled=disabled)
            yellow_val = st.slider("Yellow Time (s)", 2, 5, int(es2.get("yellow_time", YELLOW_TIME_DEFAULT)))

            # Apply changes
            if es2 is not None:
                with es2["lock"]:
                    if not disabled:
                        es2["min_green_time"] = int(min_val)
                        es2["max_green_time"] = int(max_val)
                    es2["yellow_time"] = int(yellow_val)

    # ---- Main area: 4 video feeds with counts ----
    st.markdown(
        f"<div style='background:{PRIMARY_BG}; padding:8px 12px; border-radius:8px; color:{TEXT_COLOR}; font-size:22px; font-weight:800;'>"
        "AI Smart Traffic Management — Live Feeds"
        "</div>",
        unsafe_allow_html=True,
    )

    # Large status row (Active lane and Timer) + Signals
    es_top = st.session_state.engine_state or {}
    col_status, col_signals = st.columns([1, 2])
    with col_status:
        st.markdown(
            f"<div style='background:{SECONDARY_BG}; padding:16px; border:1px solid {ACCENT_COLOR}; border-radius:10px;'>"
            f"<div style='color:{TEXT_COLOR}; font-size:18px; font-weight:700; margin-bottom:6px;'>Controller</div>"
            f"<div style='color:{SUCCESS_COLOR if not bool(es_top.get('is_yellow_phase')) else WARNING_COLOR}; font-size:28px; font-weight:800;'>"
            f"{(DIRECTIONS[int(es_top.get('current_green_lane', -1))] if int(es_top.get('current_green_lane', -1)) != -1 else 'N/A')}"
            f"</div>"
            f"<div style='color:{WARNING_COLOR}; font-size:24px; font-weight:800;'>{int(es_top.get('remaining_time', 0))}s</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with col_signals:
        st.markdown(
            f"<div style='background:{SECONDARY_BG}; padding:12px; border:1px solid {ACCENT_COLOR}; border-radius:10px;'>"
            f"<div style='color:{TEXT_COLOR}; font-size:16px; font-weight:700; margin-bottom:8px;'>Signal Indicators</div>"
            f"{render_signals(es_top)}"
            "</div>",
            unsafe_allow_html=True,
        )

    row1 = st.columns(2)
    row2 = st.columns(2)
    containers = [row1[0], row1[1], row2[0], row2[1]]

    # Placeholders for labels and images
    label_ph = []
    image_ph = []
    for idx, cont in enumerate(containers):
        with cont:
            label_ph.append(st.empty())
            image_ph.append(st.empty())

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    st.caption("Dashboard updates in real time. All processing is done on the server.")

    # Get current state
    es = st.session_state.engine_state or {}
    frames = es.get("frames", [np.zeros((VIDEO_RESIZE[1], VIDEO_RESIZE[0], 3), np.uint8) for _ in range(4)])
    counts = es.get("vehicle_counts", [0, 0, 0, 0])

    # Update video feeds
    for idx in range(4):
        label = f"{DIRECTIONS[idx]} — Vehicles: {counts[idx] if idx < len(counts) else 0}"
        label_ph[idx].caption(label)
        img = frames[idx] if idx < len(frames) else np.zeros((VIDEO_RESIZE[1], VIDEO_RESIZE[0], 3), np.uint8)
        # Use width='stretch' instead of deprecated use_column_width
        image_ph[idx].image(bgr_to_rgb(img), channels="RGB", width='stretch')

    # Auto-refresh every 1 second to match timer tick rate
    if st.session_state.get("live_update", True):
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()
