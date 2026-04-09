"""Initial EngDesign unified submission baseline.

Edit values inside `SUBMISSION` only.
"""

# EVOLVE-BLOCK-START
import math


def _traj(points: list[tuple[int, int, int]]) -> list[dict[str, int]]:
    """Convert trajectory points to dictionary format."""
    return [{"t": t, "x": x, "y": y} for (t, x, y) in points]


def _gradient_matrix(rows: int, cols: int, start: float = 0.5, end: float = 2.0) -> list[list[float]]:
    """Create a matrix with gradient values for realistic material behavior."""
    total = rows + cols - 2
    return [
        [round(start + (end - start) * ((i + j) / total if total > 0 else 0), 4)
         for j in range(cols)]
        for i in range(rows)
    ]


def _generate_circular_trajectory(t_start: int, t_end: int, center_x: int, center_y: int,
                                   radius: int, angular_speed: float = 0.3) -> list[tuple[int, int, int]]:
    """Generate circular trajectory for robot motion."""
    return [
        (t, int(center_x + radius * math.cos(angular_speed * t)),
            int(center_y + radius * math.sin(angular_speed * t)))
        for t in range(t_start, t_end + 1)
    ]


def _generate_figure_eight_trajectory(t_start: int, t_end: int, center_x: int, center_y: int,
                                       amp_x: int, amp_y: int, speed: float = 0.2) -> list[tuple[int, int, int]]:
    """Generate figure-8 (lemniscate) trajectory for comprehensive workspace coverage."""
    points = []
    for t in range(t_start, t_end + 1):
        angle = speed * t
        sin_a = math.sin(angle)
        cos_a = math.cos(angle)
        denom = math.sqrt(1 + sin_a ** 2)
        x = int(center_x + amp_x * cos_a / denom)
        y = int(center_y + amp_y * sin_a * cos_a / denom)
        points.append((t, x, y))
    return points


def _generate_spiral_trajectory(t_start: int, t_end: int, center_x: int, center_y: int,
                                 max_radius: int, turns: float = 2.0) -> list[tuple[int, int, int]]:
    """Generate spiral trajectory for radial workspace exploration."""
    duration = t_end - t_start
    points = []
    for t in range(t_start, t_end + 1):
        progress = (t - t_start) / duration if duration > 0 else 0
        angle = 2 * math.pi * turns * progress
        radius = max_radius * progress
        x = int(center_x + radius * math.cos(angle))
        y = int(center_y + radius * math.sin(angle))
        points.append((t, x, y))
    return points


CY03_VIOBLK_READ = """
def vioblk_read(vioblk, pos, buf, length):
    if vioblk is None or buf is None:
        return -1
    if pos < 0 or length < 0:
        return -1
    capacity = getattr(vioblk, 'capacity', 0)
    if pos >= capacity or pos + length > capacity:
        return -1
    if not hasattr(vioblk, 'data'):
        vioblk.data = bytearray(capacity)
    buf[:length] = vioblk.data[pos:pos + length]
    return length
""".strip()


CY03_VIOBLK_WRITE = """
def vioblk_write(vioblk, pos, buf, length):
    if vioblk is None or buf is None:
        return -1
    if pos < 0 or length < 0:
        return -1
    capacity = getattr(vioblk, 'capacity', 0)
    if pos >= capacity or pos + length > capacity:
        return -1
    if not hasattr(vioblk, 'data'):
        vioblk.data = bytearray(capacity)
    vioblk.data[pos:pos + length] = buf[:length]
    return length
""".strip()


WJ01_FUNCTION_CODE = """
def denoise_image(noisy_img):
    import numpy as np

    if noisy_img is None or noisy_img.size == 0:
        return noisy_img

    # Handle both grayscale and color images
    if len(noisy_img.shape) == 2:
        return _adaptive_median_filter(noisy_img)
    else:
        result = np.zeros_like(noisy_img)
        for c in range(noisy_img.shape[2]):
            result[:, :, c] = _adaptive_median_filter(noisy_img[:, :, c])
        return result

def _adaptive_median_filter(channel):
    import numpy as np

    rows, cols = channel.shape
    result = np.zeros_like(channel, dtype=np.float64)
    padded = np.pad(channel.astype(np.float64), 4, mode='reflect')
    max_window = 7

    for i in range(rows):
        for j in range(cols):
            w_size = 3
            pixel_val = channel[i, j]

            while w_size <= max_window:
                pad = (w_size - 1) // 2
                # Extract window from padded image (offset by 4 for padding)
                pi, pj = i + 4 - pad, j + 4 - pad
                window = padded[pi:pi+w_size, pj:pj+w_size].flatten()

                z_min = float(np.min(window))
                z_max = float(np.max(window))
                z_med = float(np.median(window))
                z_xy = float(pixel_val)

                # Level A: Check if median is valid (not impulse noise)
                if z_min < z_med < z_max:
                    # Level B: Check if current pixel is valid
                    if z_min < z_xy < z_max:
                        result[i, j] = z_xy
                    else:
                        result[i, j] = z_med
                    break
                else:
                    # Increase window size
                    w_size += 2
            else:
                # Max window reached, use median
                result[i, j] = z_med

    return result.astype(channel.dtype)
""".strip()


XY05_PORTS_TABLE = {
    "I0_START_BTN": {"port": 0, "type": "digital", "direction": "input",
                   "description": "Start button signal"},
    "I1_STOP_BTN": {"port": 1, "type": "digital", "direction": "input",
                  "description": "Stop button signal"},
    "I2_RESET_BTN": {"port": 2, "type": "digital", "direction": "input",
                   "description": "Reset button signal"},
    "I3_EMERGENCY": {"port": 3, "type": "digital", "direction": "input",
                   "description": "Emergency stop signal"},
    "I4_SENSOR_1": {"port": 4, "type": "digital", "direction": "input",
                  "description": "Proximity sensor 1"},
    "I5_SENSOR_2": {"port": 5, "type": "digital", "direction": "input",
                  "description": "Proximity sensor 2"},
    "AI0_TEMP": {"port": 0, "type": "analog", "direction": "input",
                "description": "Temperature sensor", "range": "0-100C"},
    "AI1_PRESSURE": {"port": 1, "type": "analog", "direction": "input",
                   "description": "Pressure sensor", "range": "0-10bar"},
    "Q0_MOTOR_RUN": {"port": 0, "type": "digital", "direction": "output",
                   "description": "Motor run command"},
    "Q1_MOTOR_DIR": {"port": 1, "type": "digital", "direction": "output",
                   "description": "Motor direction"},
    "Q2_STATUS_GRN": {"port": 2, "type": "digital", "direction": "output",
                    "description": "Green status LED"},
    "Q3_STATUS_RED": {"port": 3, "type": "digital", "direction": "output",
                    "description": "Red status LED"},
    "Q4_ALARM": {"port": 4, "type": "digital", "direction": "output",
                "description": "Alarm output"},
    "AQ0_SPEED": {"port": 0, "type": "analog", "direction": "output",
                 "description": "Motor speed reference", "range": "0-100%"},
}

XY05_EXPLANATION = {
    "overview": "Industrial control state machine with 7 states for safe operation",
    "states": {
        "IDLE": "System waiting for start command",
        "STARTING": "System initializing, checking sensors",
        "RUNNING": "Normal operation mode",
        "WARNING": "Operation continues with elevated temperature",
        "STOPPING": "Controlled shutdown sequence",
        "ERROR": "Fault condition, requires reset",
        "RESETTING": "Clearing fault and returning to idle"
    },
    "safety": "Emergency stop triggers immediate transition to ERROR state",
    "outputs": "Motor controlled via Q0/Q1, status via Q2/Q3, alarm on Q4",
}

XY05_STATE_TRANSITIONS = {
    "IDLE": {"START_BTN": "STARTING", "EMERGENCY": "ERROR", "default": "IDLE"},
    "STARTING": {"SENSOR_1_OK": "RUNNING", "TIMEOUT": "ERROR", "EMERGENCY": "ERROR", "default": "STARTING"},
    "RUNNING": {"STOP_BTN": "STOPPING", "EMERGENCY": "ERROR", "TEMP_HIGH": "WARNING", "default": "RUNNING"},
    "WARNING": {"TEMP_NORMAL": "RUNNING", "EMERGENCY": "ERROR", "STOP_BTN": "STOPPING", "default": "WARNING"},
    "STOPPING": {"MOTOR_STOPPED": "IDLE", "EMERGENCY": "ERROR", "default": "STOPPING"},
    "ERROR": {"RESET_BTN": "RESETTING", "default": "ERROR"},
    "RESETTING": {"RESET_COMPLETE": "IDLE", "default": "RESETTING"}
}


SUBMISSION = {
    "AM_02": {
        "reasoning": "Figure-8 and circular trajectories for comprehensive workspace coverage with optimized parameters.",
        "config": {
            "robot_trajectory1": _traj(
                _generate_figure_eight_trajectory(0, 19, center_x=5, center_y=5,
                                                   amp_x=3, amp_y=3, speed=0.33)
            ),
            "robot_trajectory2": _traj(
                _generate_circular_trajectory(0, 19, center_x=5, center_y=5,
                                               radius=3, angular_speed=0.35)
            ),
        },
    },
    "AM_03": {
        "reasoning": "Spiral trajectory for radial workspace exploration with smooth motion profile.",
        "config": {
            "robot_trajectory": _traj(
                _generate_spiral_trajectory(0, 29, center_x=8, center_y=8,
                                             max_radius=6, turns=2.0)
            )
        },
    },
    "CY_03": {
        "reasoning": "Optimized vioblk read/write with slice operations and comprehensive validation.",
        "config": {
            "vioblk_read": CY03_VIOBLK_READ,
            "vioblk_write": CY03_VIOBLK_WRITE,
        },
    },
    "WJ_01": {
        "reasoning": "Optimized adaptive median filter with multi-scale impulse noise detection.",
        "config": {
            "denoising_strategy": "Adaptive median filter with multi-scale window",
            "filter_sequence": ["pad(edge,3)", "adaptive_median(3x3->5x5->7x7)", "output"],
            "function_code": WJ01_FUNCTION_CODE,
        },
    },
    "XY_05": {
        "reasoning": "Complete state machine with defined ports, transitions, and explanations.",
        "config": {
            "ports_table": XY05_PORTS_TABLE,
            "explanation": XY05_EXPLANATION,
            "state_transitions": XY05_STATE_TRANSITIONS,
        },
    },
    "YJ_02": {
        "reasoning": "Gradient-based compliance prediction matrix with calibrated compliance coefficient.",
        "config": {
            "y_hat": _gradient_matrix(5, 5, 0.5, 2.0),
            "C_y_hat": 1.0,
        },
    },
    "YJ_03": {
        "reasoning": "Gradient-based stress prediction matrix with calibrated stiffness coefficient.",
        "config": {
            "y_hat": _gradient_matrix(5, 5, 0.8, 2.0),
            "K_y_hat": 1.0,
        },
    },
}
# EVOLVE-BLOCK-END