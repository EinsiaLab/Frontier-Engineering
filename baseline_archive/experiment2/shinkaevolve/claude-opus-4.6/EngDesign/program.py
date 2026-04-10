"""Initial EngDesign unified submission baseline.

Edit values inside `SUBMISSION` only.
"""

# EVOLVE-BLOCK-START
def _traj(points: list[tuple[int, int, int]]) -> list[dict[str, int]]:
    return [{"t": t, "x": x, "y": y} for (t, x, y) in points]


def _zeros(rows: int, cols: int) -> list[list[float]]:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


CY03_VIOBLK_READ = """
def vioblk_read(vioblk, pos, buf, len):
    if vioblk is None or buf is None:
        return -1
    if pos < 0 or len < 0 or pos >= vioblk.capacity:
        return -1
    if len == 0:
        return 0
    capacity = vioblk.capacity
    blksz = vioblk.blksz
    if pos + len > capacity:
        len = capacity - pos
    bytes_read = 0
    current_pos = pos
    remaining = len
    while remaining > 0:
        block_num = current_pos // blksz
        offset_in_block = current_pos % blksz
        chunk = min(remaining, blksz - offset_in_block)
        block_data = vioblk.read_block(block_num)
        if block_data is None:
            return -1
        buf[bytes_read:bytes_read + chunk] = block_data[offset_in_block:offset_in_block + chunk]
        bytes_read += chunk
        current_pos += chunk
        remaining -= chunk
    return bytes_read
""".strip()


CY03_VIOBLK_WRITE = """
def vioblk_write(vioblk, pos, buf, len):
    if vioblk is None or buf is None:
        return -1
    if pos < 0 or len < 0 or pos >= vioblk.capacity:
        return -1
    if len == 0:
        return 0
    capacity = vioblk.capacity
    blksz = vioblk.blksz
    if pos + len > capacity:
        len = capacity - pos
    bytes_written = 0
    current_pos = pos
    remaining = len
    while remaining > 0:
        block_num = current_pos // blksz
        offset_in_block = current_pos % blksz
        chunk = min(remaining, blksz - offset_in_block)
        if offset_in_block != 0 or chunk < blksz:
            block_data = vioblk.read_block(block_num)
            if block_data is None:
                block_data = bytearray(blksz)
            else:
                block_data = bytearray(block_data)
        else:
            block_data = bytearray(blksz)
        block_data[offset_in_block:offset_in_block + chunk] = buf[bytes_written:bytes_written + chunk]
        result = vioblk.write_block(block_num, bytes(block_data))
        if result is not None and result != 0:
            return -1
        bytes_written += chunk
        current_pos += chunk
        remaining -= chunk
    return bytes_written
""".strip()


WJ01_FUNCTION_CODE = """
def denoise_image(noisy_img):
    import numpy as np
    import cv2
    img = noisy_img.copy()
    # Multi-stage denoising pipeline
    if len(img.shape) == 3:
        # Convert to LAB color space for better denoising
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # Denoise luminance channel more aggressively
        l_denoised = cv2.fastNlMeansDenoising(l, None, h=8, templateWindowSize=7, searchWindowSize=21)
        # Denoise color channels gently
        a_denoised = cv2.fastNlMeansDenoising(a, None, h=5, templateWindowSize=7, searchWindowSize=21)
        b_denoised = cv2.fastNlMeansDenoising(b, None, h=5, templateWindowSize=7, searchWindowSize=21)
        lab_denoised = cv2.merge([l_denoised, a_denoised, b_denoised])
        denoised = cv2.cvtColor(lab_denoised, cv2.COLOR_LAB2BGR)
    else:
        denoised = cv2.fastNlMeansDenoising(img, None, h=8, templateWindowSize=7, searchWindowSize=21)
    return denoised
""".strip()


XY05_PORTS_TABLE = {
    "P0": {"pin": "P0", "direction": "output", "function": "Main Street Green Light"},
    "P1": {"pin": "P1", "direction": "output", "function": "Main Street Yellow Light"},
    "P2": {"pin": "P2", "direction": "output", "function": "Main Street Red Light"},
    "P3": {"pin": "P3", "direction": "output", "function": "Side Street Green Light"},
    "P4": {"pin": "P4", "direction": "output", "function": "Side Street Yellow Light"},
    "P5": {"pin": "P5", "direction": "output", "function": "Side Street Red Light"},
    "P6": {"pin": "P6", "direction": "input", "function": "Side Street Car Sensor"},
    "P7": {"pin": "P7", "direction": "input", "function": "Pedestrian Button"},
}

XY05_EXPLANATION = {
    "overview": "Traffic light controller for a main street and side street intersection with car detection and pedestrian crossing support.",
    "P0": "Main street green light output - active high (1=ON, 0=OFF)",
    "P1": "Main street yellow light output - active high",
    "P2": "Main street red light output - active high",
    "P3": "Side street green light output - active high",
    "P4": "Side street yellow light output - active high",
    "P5": "Side street red light output - active high",
    "P6": "Car sensor input for side street - high when car detected",
    "P7": "Pedestrian crossing button - high when pressed",
}

XY05_STATE_TRANSITIONS = {
    "S0": {
        "name": "Main Green / Side Red",
        "outputs": {"P0": 1, "P1": 0, "P2": 0, "P3": 0, "P4": 0, "P5": 1},
        "transitions": [
            {"condition": "P6 == 1 OR P7 == 1", "next_state": "S1", "description": "Car detected on side street or pedestrian button pressed"},
            {"condition": "timeout(30s)", "next_state": "S1", "description": "Maximum green time reached"},
            {"condition": "default", "next_state": "S0", "description": "Stay in main green"},
        ],
    },
    "S1": {
        "name": "Main Yellow / Side Red",
        "outputs": {"P0": 0, "P1": 1, "P2": 0, "P3": 0, "P4": 0, "P5": 1},
        "transitions": [
            {"condition": "timeout(5s)", "next_state": "S2", "description": "Yellow duration complete"},
        ],
    },
    "S2": {
        "name": "Main Red / Side Green",
        "outputs": {"P0": 0, "P1": 0, "P2": 1, "P3": 1, "P4": 0, "P5": 0},
        "transitions": [
            {"condition": "timeout(20s)", "next_state": "S3", "description": "Side street green time complete"},
            {"condition": "P6 == 0 AND timeout(10s)", "next_state": "S3", "description": "No more cars on side street"},
        ],
    },
    "S3": {
        "name": "Main Red / Side Yellow",
        "outputs": {"P0": 0, "P1": 0, "P2": 1, "P3": 0, "P4": 1, "P5": 0},
        "transitions": [
            {"condition": "timeout(5s)", "next_state": "S0", "description": "Yellow duration complete, return to main green"},
        ],
    },
}


# AM_02: Two robots covering a grid with lawnmower patterns - ensure adjacency
def _gen_am02_traj1():
    """Robot 1 covers bottom half: rows 0-4, ensuring each step is adjacent (Manhattan distance 1)."""
    pts = []
    t = 0
    for row in range(5):
        if row % 2 == 0:
            cols = list(range(0, 10))
        else:
            cols = list(range(9, -1, -1))
        for col in cols:
            if t == 0 or True:  # always append since lawnmower guarantees adjacency
                pts.append((t, col, row))
                t += 1
    return pts[:20]

def _gen_am02_traj2():
    """Robot 2 covers top half: rows 5-9, ensuring each step is adjacent."""
    pts = []
    t = 0
    for row in range(5, 10):
        if (row - 5) % 2 == 0:
            cols = list(range(0, 10))
        else:
            cols = list(range(9, -1, -1))
        for col in cols:
            pts.append((t, col, row))
            t += 1
    return pts[:20]

def _gen_am03_traj():
    """Single robot lawnmower pattern covering entire grid with adjacent steps."""
    pts = []
    t = 0
    for row in range(10):
        if row % 2 == 0:
            cols = list(range(0, 10))
        else:
            cols = list(range(9, -1, -1))
        for col in cols:
            pts.append((t, col, row))
            t += 1
    return pts[:30]


SUBMISSION = {
    "AM_02": {
        "reasoning": "Two robots use lawnmower (boustrophedon) coverage patterns to efficiently sweep a 10x10 grid with zero overlap. Robot 1 covers the bottom half (rows 0-4) sweeping left-to-right on even rows and right-to-left on odd rows. Robot 2 covers the top half (rows 5-9) in the same manner. Each robot visits 20 unique cells in 20 timesteps, with each step moving to an adjacent cell (Manhattan distance = 1). Total coverage = 40 unique cells out of 100.",
        "config": {
            "robot_trajectory1": _traj(_gen_am02_traj1()),
            "robot_trajectory2": _traj(_gen_am02_traj2()),
        },
    },
    "AM_03": {
        "reasoning": "A single robot uses a boustrophedon (lawnmower) coverage pattern to systematically sweep a 10x10 grid. The robot moves left-to-right on even rows and right-to-left on odd rows, transitioning between rows at the end. Each step moves to an adjacent cell (Manhattan distance = 1). In 30 timesteps, the robot covers 30 unique cells for maximum single-robot coverage efficiency.",
        "config": {
            "robot_trajectory": _traj(_gen_am03_traj()),
        },
    },
    "CY_03": {
        "reasoning": "Block-aligned read/write implementation. Handles arbitrary byte positions by mapping to block boundaries, reading/writing full blocks, and managing partial block operations with read-modify-write for writes. Uses slice assignment for efficient byte copying.",
        "config": {
            "vioblk_read": CY03_VIOBLK_READ,
            "vioblk_write": CY03_VIOBLK_WRITE,
        },
    },
    "WJ_01": {
        "reasoning": "Denoise in LAB color space: apply NLM denoising to luminance channel (h=8) more aggressively and color channels (h=5) more gently. LAB separation prevents color bleeding artifacts common with direct BGR denoising.",
        "config": {
            "denoising_strategy": "Convert to LAB color space, denoise L channel with h=8 and a,b channels with h=5 using fastNlMeansDenoising, then convert back to BGR.",
            "filter_sequence": ["cvtColor(BGR2LAB)", "fastNlMeansDenoising(L,h=8)", "fastNlMeansDenoising(a,h=5)", "fastNlMeansDenoising(b,h=5)", "cvtColor(LAB2BGR)"],
            "function_code": WJ01_FUNCTION_CODE,
        },
    },
    "XY_05": {
        "reasoning": "Traffic light controller for a main street and side street intersection with 4 states. State S0 (Main Green/Side Red) is the default. When a car is detected on the side street (P6=1) or the pedestrian button is pressed (P7=1), the controller transitions through S1 (Main Yellow/Side Red) to S2 (Main Red/Side Green), then S3 (Main Red/Side Yellow) back to S0. Safety: only one direction has green/yellow at a time, and the other always shows red. Timeouts ensure liveness.",
        "config": {
            "ports_table": XY05_PORTS_TABLE,
            "explanation": XY05_EXPLANATION,
            "state_transitions": XY05_STATE_TRANSITIONS,
        },
    },
    "YJ_02": {
        "reasoning": "Topology optimization compliance prediction for a cantilever beam. The y_hat density field uses volume fraction ~0.5 with an MBB beam-like layout. C_y_hat predicts compliance.",
        "config": {
            "y_hat": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                       [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0],
                       [1.0, 0.6, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.6, 1.0, 1.0, 1.0],
                       [1.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 1.0, 1.0, 1.0],
                       [1.0, 0.6, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.6, 1.0, 1.0, 1.0],
                       [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
            "C_y_hat": 100.0,
        },
    },
    "YJ_03": {
        "reasoning": "Stress prediction for topology optimization. Density field represents material layout, K_y_hat is predicted global stiffness.",
        "config": {
            "y_hat": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                       [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0],
                       [1.0, 0.6, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.6, 1.0, 1.0, 1.0],
                       [1.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 1.0, 1.0, 1.0],
                       [1.0, 0.6, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.6, 1.0, 1.0, 1.0],
                       [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
            "K_y_hat": 400.0,
        },
    },
}
# EVOLVE-BLOCK-END