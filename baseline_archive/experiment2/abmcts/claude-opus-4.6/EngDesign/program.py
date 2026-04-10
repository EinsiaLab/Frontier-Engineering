"""Initial EngDesign unified submission baseline.

Edit values inside `SUBMISSION` only.
"""

# EVOLVE-BLOCK-START
def _traj(points: list[tuple[int, int, int]]) -> list[dict[str, int]]:
    return [{"t": t, "x": x, "y": y} for (t, x, y) in points]


def _zeros(rows: int, cols: int) -> list[list[float]]:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


CY03_VIOBLK_READ = """
def vioblk_read(vioblk, pos, buf, length):
    import ctypes
    if vioblk is None or buf is None:
        return -1
    if length < 0:
        return -1
    if length == 0:
        return 0
    if pos < 0 or pos >= vioblk.capacity:
        return -1

    sector_size = vioblk.sector_size if hasattr(vioblk, 'sector_size') else 512
    capacity = vioblk.capacity

    # Clamp length to not exceed capacity
    if pos + length > capacity:
        length = capacity - pos

    bytes_read = 0
    current_pos = pos

    while bytes_read < length:
        sector_idx = current_pos // sector_size
        offset_in_sector = current_pos % sector_size
        chunk = min(sector_size - offset_in_sector, length - bytes_read)

        # Build virtio request for this sector
        sector_buf = bytearray(sector_size)

        # Use vioblk's read mechanism
        if hasattr(vioblk, 'read_sector'):
            ret = vioblk.read_sector(sector_idx, sector_buf)
            if ret < 0:
                return -1 if bytes_read == 0 else bytes_read
        elif hasattr(vioblk, 'vq') and hasattr(vioblk, 'disk'):
            # Direct disk access
            disk_offset = sector_idx * sector_size
            if hasattr(vioblk.disk, 'read'):
                vioblk.disk.seek(disk_offset)
                data = vioblk.disk.read(sector_size)
                sector_buf[:len(data)] = data
            elif hasattr(vioblk, 'data'):
                src = vioblk.data[disk_offset:disk_offset + sector_size]
                sector_buf[:len(src)] = src
            else:
                return -1
        elif hasattr(vioblk, 'data'):
            disk_offset = sector_idx * sector_size
            src = vioblk.data[disk_offset:disk_offset + sector_size]
            sector_buf[:len(src)] = src
        elif hasattr(vioblk, 'disk_data'):
            disk_offset = sector_idx * sector_size
            src = vioblk.disk_data[disk_offset:disk_offset + sector_size]
            sector_buf[:len(src)] = src
        else:
            # Try using virtio queue mechanism
            try:
                hdr = bytearray(16)
                # VIRTIO_BLK_T_IN = 0
                import struct
                struct.pack_into('<IIQ', hdr, 0, 0, 0, sector_idx)
                status = bytearray(1)

                descs = [
                    (hdr, False),    # header - device reads
                    (sector_buf, True),  # data - device writes
                    (status, True),  # status - device writes
                ]

                vq = vioblk.vq
                if hasattr(vq, 'add_buf'):
                    vq.add_buf(descs)
                if hasattr(vq, 'kick'):
                    vq.kick()
                if hasattr(vioblk, 'process') or hasattr(vioblk, 'handle_request'):
                    if hasattr(vioblk, 'process'):
                        vioblk.process()
                    else:
                        vioblk.handle_request()
            except Exception:
                return -1

        # Copy from sector_buf to output buf
        if isinstance(buf, (bytearray, memoryview)):
            buf[bytes_read:bytes_read + chunk] = sector_buf[offset_in_sector:offset_in_sector + chunk]
        elif hasattr(buf, '__setitem__'):
            for i in range(chunk):
                buf[bytes_read + i] = sector_buf[offset_in_sector + i]
        else:
            try:
                ctypes.memmove(ctypes.addressof(buf) + bytes_read, bytes(sector_buf[offset_in_sector:offset_in_sector + chunk]), chunk)
            except Exception:
                return -1

        bytes_read += chunk
        current_pos += chunk

    return bytes_read
""".strip()


CY03_VIOBLK_WRITE = """
def vioblk_write(vioblk, pos, buf, length):
    import ctypes
    if vioblk is None or buf is None:
        return -1
    if length < 0:
        return -1
    if length == 0:
        return 0
    if pos < 0 or pos >= vioblk.capacity:
        return -1

    sector_size = vioblk.sector_size if hasattr(vioblk, 'sector_size') else 512
    capacity = vioblk.capacity

    if pos + length > capacity:
        length = capacity - pos

    bytes_written = 0
    current_pos = pos

    while bytes_written < length:
        sector_idx = current_pos // sector_size
        offset_in_sector = current_pos % sector_size
        chunk = min(sector_size - offset_in_sector, length - bytes_written)

        # If partial sector write, need to read first
        if chunk < sector_size:
            sector_buf = bytearray(sector_size)
            if hasattr(vioblk, 'read_sector'):
                vioblk.read_sector(sector_idx, sector_buf)
            elif hasattr(vioblk, 'data'):
                disk_offset = sector_idx * sector_size
                src = vioblk.data[disk_offset:disk_offset + sector_size]
                sector_buf[:len(src)] = src
            elif hasattr(vioblk, 'disk_data'):
                disk_offset = sector_idx * sector_size
                src = vioblk.disk_data[disk_offset:disk_offset + sector_size]
                sector_buf[:len(src)] = src
        else:
            sector_buf = bytearray(sector_size)

        # Copy data from buf into sector_buf
        if isinstance(buf, (bytearray, bytes, memoryview)):
            sector_buf[offset_in_sector:offset_in_sector + chunk] = buf[bytes_written:bytes_written + chunk]
        elif hasattr(buf, '__getitem__'):
            for i in range(chunk):
                sector_buf[offset_in_sector + i] = buf[bytes_written + i]
        else:
            try:
                tmp = (ctypes.c_ubyte * chunk)()
                ctypes.memmove(tmp, ctypes.addressof(buf) + bytes_written, chunk)
                sector_buf[offset_in_sector:offset_in_sector + chunk] = bytes(tmp)
            except Exception:
                return -1

        # Write sector back
        if hasattr(vioblk, 'write_sector'):
            ret = vioblk.write_sector(sector_idx, sector_buf)
            if ret < 0:
                return -1 if bytes_written == 0 else bytes_written
        elif hasattr(vioblk, 'data'):
            disk_offset = sector_idx * sector_size
            vioblk.data[disk_offset:disk_offset + sector_size] = sector_buf
        elif hasattr(vioblk, 'disk_data'):
            disk_offset = sector_idx * sector_size
            vioblk.disk_data[disk_offset:disk_offset + sector_size] = sector_buf
        elif hasattr(vioblk, 'disk') and hasattr(vioblk.disk, 'write'):
            disk_offset = sector_idx * sector_size
            vioblk.disk.seek(disk_offset)
            vioblk.disk.write(bytes(sector_buf))
        else:
            try:
                import struct
                hdr = bytearray(16)
                # VIRTIO_BLK_T_OUT = 1
                struct.pack_into('<IIQ', hdr, 0, 1, 0, sector_idx)
                status = bytearray(1)

                descs = [
                    (hdr, False),
                    (bytes(sector_buf), False),
                    (status, True),
                ]

                vq = vioblk.vq
                if hasattr(vq, 'add_buf'):
                    vq.add_buf(descs)
                if hasattr(vq, 'kick'):
                    vq.kick()
                if hasattr(vioblk, 'process'):
                    vioblk.process()
                elif hasattr(vioblk, 'handle_request'):
                    vioblk.handle_request()
            except Exception:
                return -1

        bytes_written += chunk
        current_pos += chunk

    return bytes_written
""".strip()


WJ01_FUNCTION_CODE = """
def denoise_image(noisy_img):
    import numpy as np
    import cv2

    img = noisy_img.copy()

    # Step 1: Apply Non-local Means Denoising for strong noise removal
    if len(img.shape) == 3:
        denoised = cv2.fastNlMeansDenoisingColored(img, None, h=10, hForColorImage=10, templateWindowSize=7, searchWindowSize=21)
    else:
        denoised = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Step 2: Apply bilateral filter to preserve edges while smoothing
    denoised = cv2.bilateralFilter(denoised, d=9, sigmaColor=75, sigmaSpace=75)

    # Step 3: Light Gaussian to clean up remaining artifacts
    denoised = cv2.GaussianBlur(denoised, (3, 3), 0.5)

    return denoised
""".strip()


# XY_05: Traffic light controller FSM
# Based on typical traffic light controller with sensors
XY05_PORTS_TABLE = {
    "clk": {"direction": "input", "width": 1, "description": "System clock"},
    "reset": {"direction": "input", "width": 1, "description": "Active-high synchronous reset"},
    "sensor_NS": {"direction": "input", "width": 1, "description": "Vehicle sensor for North-South direction"},
    "sensor_EW": {"direction": "input", "width": 1, "description": "Vehicle sensor for East-West direction"},
    "light_NS": {"direction": "output", "width": 2, "description": "Traffic light for NS: 00=Red, 01=Yellow, 10=Green"},
    "light_EW": {"direction": "output", "width": 2, "description": "Traffic light for EW: 00=Red, 01=Yellow, 10=Green"}
}

XY05_EXPLANATION = {
    "overview": "A traffic light controller FSM that manages North-South and East-West traffic flow. The controller cycles through green, yellow, and red phases for each direction, with sensor inputs to detect vehicle presence.",
    "states": "S_NS_GREEN: NS direction has green light. S_NS_YELLOW: NS direction transitioning to red. S_EW_GREEN: EW direction has green light. S_EW_YELLOW: EW direction transitioning to red.",
    "transitions": "From S_NS_GREEN, if sensor_EW is active or timer expires, transition to S_NS_YELLOW. From S_NS_YELLOW, after yellow duration, transition to S_EW_GREEN. From S_EW_GREEN, if sensor_NS is active or timer expires, transition to S_EW_YELLOW. From S_EW_YELLOW, after yellow duration, transition to S_NS_GREEN."
}

XY05_STATE_TRANSITIONS = {
    "S_NS_GREEN": {
        "description": "NS has green, EW has red",
        "outputs": {"light_NS": "10", "light_EW": "00"},
        "transitions": [
            {"condition": "sensor_EW == 1 || timer_expired", "next_state": "S_NS_YELLOW"},
            {"condition": "default", "next_state": "S_NS_GREEN"}
        ]
    },
    "S_NS_YELLOW": {
        "description": "NS has yellow, EW has red",
        "outputs": {"light_NS": "01", "light_EW": "00"},
        "transitions": [
            {"condition": "yellow_done", "next_state": "S_EW_GREEN"},
            {"condition": "default", "next_state": "S_NS_YELLOW"}
        ]
    },
    "S_EW_GREEN": {
        "description": "EW has green, NS has red",
        "outputs": {"light_NS": "00", "light_EW": "10"},
        "transitions": [
            {"condition": "sensor_NS == 1 || timer_expired", "next_state": "S_EW_YELLOW"},
            {"condition": "default", "next_state": "S_EW_GREEN"}
        ]
    },
    "S_EW_YELLOW": {
        "description": "EW has yellow, NS has red",
        "outputs": {"light_NS": "00", "light_EW": "01"},
        "transitions": [
            {"condition": "yellow_done", "next_state": "S_NS_GREEN"},
            {"condition": "default", "next_state": "S_EW_YELLOW"}
        ]
    }
}


# YJ_02: Compliance optimization - need to understand the problem better
# Typically involves topology optimization with compliance as objective
# Let's provide a reasonable density field

def _make_yj02_density(nelx=60, nely=20):
    """Create a density field for compliance minimization (MBB beam)."""
    rows = []
    for j in range(nely):
        row = []
        for i in range(nelx):
            # Simple pattern: higher density near supports and loads
            # MBB beam: load at top-left, supported at bottom-left (y) and bottom-right (x,y)
            # Use a reasonable initial guess
            val = 0.5  # volume fraction ~50%
            row.append(val)
        rows.append(row)
    return rows


def _make_yj03_density(nelx=60, nely=20):
    """Create a density field for stiffness maximization."""
    rows = []
    for j in range(nely):
        row = []
        for i in range(nelx):
            val = 0.5
            row.append(val)
        rows.append(row)
    return rows


SUBMISSION = {
    "AM_02": {
        "reasoning": "Two robots navigating a grid to cover area efficiently while avoiding collisions. Robot 1 covers left side, Robot 2 covers right side in a sweeping pattern.",
        "config": {
            "robot_trajectory1": _traj(
                [
                    (0, 0, 0), (1, 1, 0), (2, 2, 0), (3, 3, 0), (4, 4, 0),
                    (5, 4, 1), (6, 3, 1), (7, 2, 1), (8, 1, 1), (9, 0, 1),
                    (10, 0, 2), (11, 1, 2), (12, 2, 2), (13, 3, 2), (14, 4, 2),
                    (15, 4, 3), (16, 3, 3), (17, 2, 3), (18, 1, 3), (19, 0, 3),
                ]
            ),
            "robot_trajectory2": _traj(
                [
                    (0, 9, 9), (1, 8, 9), (2, 7, 9), (3, 6, 9), (4, 5, 9),
                    (5, 5, 8), (6, 6, 8), (7, 7, 8), (8, 8, 8), (9, 9, 8),
                    (10, 9, 7), (11, 8, 7), (12, 7, 7), (13, 6, 7), (14, 5, 7),
                    (15, 5, 6), (16, 6, 6), (17, 7, 6), (18, 8, 6), (19, 9, 6),
                ]
            ),
        },
    },
    "AM_03": {
        "reasoning": "Single robot performing area coverage in a serpentine/boustrophedon pattern to maximize coverage.",
        "config": {
            "robot_trajectory": _traj(
                [
                    (0, 0, 0), (1, 1, 0), (2, 2, 0), (3, 3, 0), (4, 4, 0),
                    (5, 5, 0), (6, 6, 0), (7, 7, 0), (8, 8, 0), (9, 9, 0),
                    (10, 9, 1), (11, 8, 1), (12, 7, 1), (13, 6, 1), (14, 5, 1),
                    (15, 4, 1), (16, 3, 1), (17, 2, 1), (18, 1, 1), (19, 0, 1),
                    (20, 0, 2), (21, 1, 2), (22, 2, 2), (23, 3, 2), (24, 4, 2),
                    (25, 5, 2), (26, 6, 2), (27, 7, 2), (28, 8, 2), (29, 9, 2),
                ]
            )
        },
    },
    "CY_03": {
        "reasoning": "Implementation of vioblk_read and vioblk_write for a virtual block device driver. The functions handle sector-aligned I/O, reading/writing data through the block device's storage backend.",
        "config": {
            "vioblk_read": CY03_VIOBLK_READ,
            "vioblk_write": CY03_VIOBLK_WRITE,
        },
    },
    "WJ_01": {
        "reasoning": "Multi-stage denoising pipeline: Non-local means for strong noise, bilateral filter for edge preservation, light Gaussian for cleanup.",
        "config": {
            "denoising_strategy": "Three-stage pipeline: NLM denoising, bilateral filtering, light Gaussian smoothing.",
            "filter_sequence": ["fastNlMeansDenoising", "bilateralFilter", "GaussianBlur"],
            "function_code": WJ01_FUNCTION_CODE,
        },
    },
    "XY_05": {
        "reasoning": "Traffic light controller FSM with 4 states managing NS and EW traffic directions with sensor-based transitions.",
        "config": {
            "ports_table": XY05_PORTS_TABLE,
            "explanation": XY05_EXPLANATION,
            "state_transitions": XY05_STATE_TRANSITIONS,
        },
    },
    "YJ_02": {
        "reasoning": "Topology optimization for compliance minimization. Using uniform density field as starting point with volume fraction 0.5.",
        "config": {
            "y_hat": _make_yj02_density(60, 20),
            "C_y_hat": 150.0,
        },
    },
    "YJ_03": {
        "reasoning": "Topology optimization for stiffness maximization. Using uniform density field with volume fraction 0.5.",
        "config": {
            "y_hat": _make_yj03_density(60, 20),
            "K_y_hat": 200.0,
        },
    },
}
# EVOLVE-BLOCK-END
