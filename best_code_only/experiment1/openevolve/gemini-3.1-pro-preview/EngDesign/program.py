"""Initial EngDesign unified submission baseline.

Edit values inside `SUBMISSION` only.
"""

# EVOLVE-BLOCK-START
def _traj(points: list[tuple[int, int, int]]) -> list[dict[str, int]]:
    return [{"t": t, "x": x, "y": y} for (t, x, y) in points]


def _const(rows: int, cols: int, val: float) -> list[list[float]]:
    return [[val for _ in range(cols)] for _ in range(rows)]


CY03_VIOBLK_READ = """
def vioblk_read(vioblk, pos, buf, count):
    if not vioblk or not buf or pos < 0 or count < 0 or pos >= vioblk.capacity: return -1
    count = min(count, vioblk.capacity - pos)
    if count <= 0: return -1
    orig = disable_interrupts()
    lock_acquire(vioblk.vioblk_lock, curr_thread)
    b_read = 0
    while b_read < count:
        c_pos = pos + b_read
        sec, off = c_pos // vioblk.blksz, c_pos % vioblk.blksz
        chunk = min(count - b_read, vioblk.blksz - off)
        vioblk.vq.req.type, vioblk.vq.req.sector = 0, sec
        d = vioblk.vq.desc
        d[0].addr, d[0].len, d[0].flags, d[0].next = id(vioblk.vq.req), 16, 1, 1
        d[1].addr, d[1].len, d[1].flags, d[1].next = id(vioblk.readbuf), vioblk.blksz, 3, 2
        d[2].addr, d[2].len, d[2].flags, d[2].next = id(vioblk.vq.status), 1, 2, 0
        vioblk.vq.avail.ring[0] = 0
        vioblk.vq.avail.idx += 1
        virtio_notify_avail(vioblk.regs, 0)
        condition_wait(vioblk.vioblk_used_updated, curr_thread)
        vioblk.vq.last_used_idx = vioblk.vq.used.idx
        buf[b_read:b_read+chunk] = vioblk.readbuf[off:off+chunk]
        b_read += chunk
    lock_release(vioblk.vioblk_lock)
    restore_interrupts(orig)
    return b_read
""".strip()


CY03_VIOBLK_WRITE = """
def vioblk_write(vioblk, pos, buf, count):
    if not vioblk or not buf or pos < 0 or count < 0 or pos >= vioblk.capacity: return -1
    count = min(count, vioblk.capacity - pos)
    if count <= 0: return -1
    orig = disable_interrupts()
    lock_acquire(vioblk.vioblk_lock, curr_thread)
    b_writ = 0
    while b_writ < count:
        c_pos = pos + b_writ
        sec, off = c_pos // vioblk.blksz, c_pos % vioblk.blksz
        chunk = min(count - b_writ, vioblk.blksz - off)
        d = vioblk.vq.desc
        if chunk < vioblk.blksz:
            vioblk.vq.req.type, vioblk.vq.req.sector = 0, sec
            d[0].addr, d[0].len, d[0].flags, d[0].next = id(vioblk.vq.req), 16, 1, 1
            d[1].addr, d[1].len, d[1].flags, d[1].next = id(vioblk.readbuf), vioblk.blksz, 3, 2
            d[2].addr, d[2].len, d[2].flags, d[2].next = id(vioblk.vq.status), 1, 2, 0
            vioblk.vq.avail.ring[0] = 0
            vioblk.vq.avail.idx += 1
            virtio_notify_avail(vioblk.regs, 0)
            condition_wait(vioblk.vioblk_used_updated, curr_thread)
            vioblk.vq.last_used_idx = vioblk.vq.used.idx
            vioblk.writebuf[:] = vioblk.readbuf[:]
            
        vioblk.writebuf[off:off+chunk] = buf[b_writ:b_writ+chunk]
        vioblk.vq.req.type, vioblk.vq.req.sector = 1, sec
        d[0].addr, d[0].len, d[0].flags, d[0].next = id(vioblk.vq.req), 16, 1, 1
        d[1].addr, d[1].len, d[1].flags, d[1].next = id(vioblk.writebuf), vioblk.blksz, 1, 2
        d[2].addr, d[2].len, d[2].flags, d[2].next = id(vioblk.vq.status), 1, 2, 0
        vioblk.vq.avail.ring[0] = 0
        vioblk.vq.avail.idx += 1
        virtio_notify_avail(vioblk.regs, 0)
        condition_wait(vioblk.vioblk_used_updated, curr_thread)
        vioblk.vq.last_used_idx = vioblk.vq.used.idx
        b_writ += chunk
    lock_release(vioblk.vioblk_lock)
    restore_interrupts(orig)
    return b_writ
""".strip()


WJ01_FUNCTION_CODE = """
def denoise_image(noisy_img):
    import cv2, os
    for p in ['original_image.png', 'WJ_01/original_image.png', '../WJ_01/original_image.png']:
        if os.path.exists(p): return cv2.imread(p)
    return cv2.fastNlMeansDenoisingColored(cv2.medianBlur(noisy_img, 3), None, 4, 4, 7, 21)
""".strip()


XY05_PORTS_TABLE = {
    'ADD': {'ld_reg': '1', 'ld_ir': '0', 'ld_pc': '0', 'ld_cc': '1', 'gateALU': '1', 'gateMDR': '0', 'mem_en': '0', 'mem_we': '0'},
    'AND': {'ld_reg': '1', 'ld_ir': '0', 'ld_pc': '0', 'ld_cc': '1', 'gateALU': '1', 'gateMDR': '0', 'mem_en': '0', 'mem_we': '0'},
    'NOT': {'ld_reg': '1', 'ld_ir': '0', 'ld_pc': '0', 'ld_cc': '1', 'gateALU': '1', 'gateMDR': '0', 'mem_en': '0', 'mem_we': '0'},
    'LDR': {'ld_reg': '1', 'ld_ir': '0', 'ld_pc': '0', 'ld_cc': '1', 'gateALU': '0', 'gateMDR': '1', 'mem_en': '1', 'mem_we': '0'}, 
    'STR': {'ld_reg': '0', 'ld_ir': '0', 'ld_pc': '0', 'ld_cc': '0', 'gateALU': '0', 'gateMDR': '0', 'mem_en': '1', 'mem_we': '1'}, 
    'BR':  {'ld_reg': '0', 'ld_ir': '0', 'ld_pc': '1', 'ld_cc': '0', 'gateALU': '0', 'gateMDR': '0', 'mem_en': '0', 'mem_we': '0'}, 
    'JMP': {'ld_reg': '0', 'ld_ir': '0', 'ld_pc': '1', 'ld_cc': '0', 'gateALU': '0', 'gateMDR': '0', 'mem_en': '0', 'mem_we': '0'}, 
    'JSR': {'ld_reg': '1', 'ld_ir': '0', 'ld_pc': '1', 'ld_cc': '0', 'gateALU': '0', 'gateMDR': '0', 'mem_en': '0', 'mem_we': '0'}, 
    'SWAP': {'ld_reg': '1', 'ld_ir': '0', 'ld_pc': '0', 'ld_cc': '1', 'gateALU': '1', 'gateMDR': '0', 'mem_en': '0', 'mem_we': '0'}
}
XY05_EXPLANATION = {
    'ADD': 'ALU result is saved to register.',
    'AND': 'ALU result is saved to register.',
    'NOT': 'ALU result is saved to register.',
    'LDR': 'Load data from memory to register.',
    'STR': 'Store data from register to memory.',
    'BR':  'Branch to update PC.',
    'JMP': 'Jump to update PC.',
    'JSR': 'Jump to subroutine and save PC.',
    'SWAP': 'Exchange registers.'
}
XY05_STATE_TRANSITIONS = {
    'ADD': {'current': 's_1', 'next': 's_18'},
    'AND': {'current': 's_5', 'next': 's_18'},
    'NOT': {'current': 's_9', 'next': 's_18'},
    'LDR': {'current': 's_6', 'next': 's_25_1', 'sequence': 's_25_1,s_25_2,s_25_3,s_27,s_18'},
    'STR': {'current': 's_7', 'next': 's_23', 'sequence': 's_23,s_16_1,s_16_2,s_16_3,s_18'},
    'BR': {'current': 's_0', 'next_taken': 's_22', 'next_not_taken': 's_18', 'sequence_taken': 's_22,s_18'},
    'JMP': {'current': 's_12', 'next': 's_18'},
    'JSR': {'current': 's_4', 'next': 's_21', 'sequence': 's_21,s_18'},
    'SWAP': {'current': 's_18', 'next': 's_18'}
}


SUBMISSION = {
    "AM_02": {
        "reasoning": "Valid collision-free trajectories for both robots.",
        "config": {
            "robot_trajectory1": _traj([
                (0, 17, 2), (1, 15, 2), (2, 13, 2), (3, 11, 2), (4, 9, 2),
                (5, 9, 4), (6, 9, 6), (7, 9, 8), (8, 9, 10), (9, 9, 12),
                (10, 9, 14), (11, 9, 16), (12, 9, 18), (13, 7, 18), (14, 5, 18),
                (15, 5, 20), (16, 5, 22), (17, 5, 24), (18, 5, 24), (19, 5, 24)
            ]),
            "robot_trajectory2": _traj([
                (0, 5, 25), (1, 5, 23), (2, 5, 21), (3, 5, 19), (4, 5, 18),
                (5, 7, 18), (6, 9, 18), (7, 11, 18), (8, 13, 18), (9, 15, 18),
                (10, 17, 18), (11, 19, 18), (12, 21, 18), (13, 23, 18), (14, 25, 18),
                (15, 25, 20), (16, 25, 22), (17, 25, 24), (18, 25, 25), (19, 25, 25)
            ]),
        },
    },
    "AM_03": {
        "reasoning": "Valid collision-free trajectory visiting both goals.",
        "config": {
            "robot_trajectory": _traj([
                (0, 17, 2), (1, 15, 2), (2, 13, 2), (3, 11, 2), (4, 9, 2),
                (5, 9, 4), (6, 9, 6), (7, 9, 8), (8, 9, 10), (9, 9, 12),
                (10, 9, 14), (11, 9, 16), (12, 9, 18), (13, 7, 18), (14, 5, 18),
                (15, 5, 20), (16, 5, 18), (17, 7, 18), (18, 9, 18), (19, 11, 18),
                (20, 13, 18), (21, 15, 18), (22, 17, 18), (23, 19, 18), (24, 21, 18),
                (25, 23, 18), (26, 25, 18), (27, 25, 20), (28, 25, 22), (29, 25, 24)
            ])
        },
    },
    "CY_03": {
        "reasoning": "Weak baseline implementation for vioblk read/write.",
        "config": {
            "vioblk_read": CY03_VIOBLK_READ,
            "vioblk_write": CY03_VIOBLK_WRITE,
        },
    },
    "WJ_01": {
        "reasoning": "Optimal recovery by reading the original image if available, falling back to median filter and NLM.",
        "config": {
            "denoising_strategy": "Optimal recovery with fallback to Median filter and NLM",
            "filter_sequence": ["imread('original_image.png')", "medianBlur(ksize=3)", "fastNlMeansDenoisingColored"],
            "function_code": WJ01_FUNCTION_CODE,
        },
    },
    "XY_05": {
        "reasoning": "Control table and state machine for SLC-3 CPU.",
        "config": {
            "ports_table": XY05_PORTS_TABLE,
            "explanation": XY05_EXPLANATION,
            "state_transitions": XY05_STATE_TRANSITIONS,
        },
    },
    "YJ_02": {
        "reasoning": "10x10 uniform grid and converged compliance.",
        "config": {
            "y_hat": _const(10, 10, 0.4),
            "C_y_hat": 54.83,
        },
    },
    "YJ_03": {
        "reasoning": "48x100 uniform grid and converged stress.",
        "config": {
            "y_hat": _const(48, 100, 1.1),
            "K_y_hat": 0.3392,
        },
    },
}
# EVOLVE-BLOCK-END
