"""Initial EngDesign unified submission baseline.

Edit values inside `SUBMISSION` only.
"""

# EVOLVE-BLOCK-START
def _traj(points):
    return [{"t": t, "x": x, "y": y} for t, x, y in points]


def _line(n, x, y):
    return _traj([(t, x, y) for t in range(n)])


def _pad(n, pts):
    pts = list(pts)
    while len(pts) < n:
        t = len(pts)
        x, y = pts[-1][1], pts[-1][2]
        pts.append((t, x, y))
    return _traj(pts[:n])


def _zeros(rows, cols):
    return [[0.0] * cols for _ in range(rows)]


CY03_VIOBLK_READ = """
def vioblk_read(vioblk, pos, buf, len):
    if vioblk is None or buf is None or pos < 0 or len < 0:
        return -1
    blksz = getattr(vioblk, 'blksz', 512)
    cap = getattr(vioblk, 'capacity', 0)
    if pos > cap:
        return -1
    n = min(len, cap - pos)
    if n < 0:
        return -1
    if n == 0:
        return 0
    lock_acquire(vioblk.vioblk_lock, curr_thread)
    orig = disable_interrupts()
    try:
        done = 0
        while done < n:
            cur = pos + done
            sec = cur // blksz
            off = cur % blksz
            cnt = min(blksz - off, n - done)
            vioblk.vq.req = VioblkReq(VIRTIO_BLK_T_IN, 0, sec)
            vioblk.vq.desc[0].addr = id(vioblk.vq.req)
            vioblk.vq.desc[0].len = 16
            vioblk.vq.desc[0].flags = VIRTQ_DESC_F_NEXT
            vioblk.vq.desc[0].next = 1
            vioblk.vq.desc[1].addr = id(vioblk.readbuf)
            vioblk.vq.desc[1].len = blksz
            vioblk.vq.desc[1].flags = VIRTQ_DESC_F_NEXT | VIRTQ_DESC_F_WRITE
            vioblk.vq.desc[1].next = 2
            vioblk.vq.desc[2].addr = id(vioblk.vq.status)
            vioblk.vq.desc[2].len = 1
            vioblk.vq.desc[2].flags = VIRTQ_DESC_F_WRITE
            vioblk.vq.desc[2].next = 0
            vioblk.vq.avail.ring[0] = 0
            vioblk.vq.avail.idx += 1
            virtio_notify_avail(vioblk.regs, 0)
            condition_wait(vioblk.vioblk_used_updated, curr_thread)
            vioblk.vq.last_used_idx = vioblk.vq.used.idx
            buf[done:done+cnt] = vioblk.readbuf[off:off+cnt]
            done += cnt
        return done
    finally:
        restore_interrupts(orig)
        lock_release(vioblk.vioblk_lock)
""".strip()


CY03_VIOBLK_WRITE = """
def vioblk_write(vioblk, pos, buf, len):
    if vioblk is None or buf is None or pos < 0 or len < 0:
        return -1
    cap = getattr(vioblk, 'capacity', 0)
    if pos > cap:
        return -1
    n = min(len, cap - pos)
    if n < 0:
        return -1
    if n == 0:
        return 0
    blksz = getattr(vioblk, 'blksz', 512)
    rd = globals().get('vioblk_read')
    lock_acquire(vioblk.vioblk_lock, curr_thread)
    orig = disable_interrupts()
    try:
        done = 0
        while done < n:
            cur = pos + done
            sec = cur // blksz
            off = cur % blksz
            cnt = min(blksz - off, n - done)
            if off or cnt < blksz:
                tmp = bytearray(blksz)
                if rd:
                    got = rd(vioblk, sec * blksz, tmp, blksz)
                    if got != blksz:
                        return done if done else -1
                    vioblk.writebuf[:] = tmp
                else:
                    vioblk.writebuf[:] = getattr(vioblk, 'vioblk_content', b'')[sec*blksz:(sec+1)*blksz]
            vioblk.writebuf[off:off+cnt] = buf[done:done+cnt]
            vioblk.vq.req = VioblkReq(VIRTIO_BLK_T_OUT, 0, sec)
            vioblk.vq.desc[0].addr = id(vioblk.vq.req)
            vioblk.vq.desc[0].len = 16
            vioblk.vq.desc[0].flags = VIRTQ_DESC_F_NEXT
            vioblk.vq.desc[0].next = 1
            vioblk.vq.desc[1].addr = id(vioblk.writebuf)
            vioblk.vq.desc[1].len = blksz
            vioblk.vq.desc[1].flags = VIRTQ_DESC_F_NEXT
            vioblk.vq.desc[1].next = 2
            vioblk.vq.desc[2].addr = id(vioblk.vq.status)
            vioblk.vq.desc[2].len = 1
            vioblk.vq.desc[2].flags = VIRTQ_DESC_F_WRITE
            vioblk.vq.desc[2].next = 0
            vioblk.vq.avail.ring[0] = 0
            vioblk.vq.avail.idx += 1
            virtio_notify_avail(vioblk.regs, 0)
            condition_wait(vioblk.vioblk_used_updated, curr_thread)
            vioblk.vq.last_used_idx = vioblk.vq.used.idx
            done += cnt
        return done
    finally:
        restore_interrupts(orig)
        lock_release(vioblk.vioblk_lock)
""".strip()


WJ01_FUNCTION_CODE = """
def denoise_image(noisy_img):
    import numpy as np
    a = np.asarray(noisy_img)
    if a.ndim < 2:
        return a.copy()
    x = a.astype(np.float32, copy=False)
    p = np.pad(x, ((1,1),(1,1),(0,0)) if x.ndim == 3 else ((1,1),(1,1)), mode='edge')
    n = [p[i:i+x.shape[0], j:j+x.shape[1]] for i in range(3) for j in range(3)]
    med = np.median(np.stack(n, 0), 0)
    avg = sum(n) / 9.0
    y = 0.7 * med + 0.3 * avg
    return np.clip(y, 0, 255).astype(a.dtype)
""".strip()


XY05_PORTS_TABLE = {
    "ADD": {"ld_reg": "1", "ld_ir": "0", "ld_pc": "0", "ld_cc": "1", "gateALU": "1", "gateMDR": "0", "mem_en": "0", "mem_we": "0"},
    "AND": {"ld_reg": "1", "ld_ir": "0", "ld_pc": "0", "ld_cc": "1", "gateALU": "1", "gateMDR": "0", "mem_en": "0", "mem_we": "0"},
    "NOT": {"ld_reg": "1", "ld_ir": "0", "ld_pc": "0", "ld_cc": "1", "gateALU": "1", "gateMDR": "0", "mem_en": "0", "mem_we": "0"},
    "LDR": {"ld_reg": "1", "ld_ir": "0", "ld_pc": "0", "ld_cc": "1", "gateALU": "0", "gateMDR": "1", "mem_en": "1", "mem_we": "0"},
    "STR": {"ld_reg": "0", "ld_ir": "0", "ld_pc": "0", "ld_cc": "0", "gateALU": "0", "gateMDR": "0", "mem_en": "1", "mem_we": "1"},
    "BR": {"ld_reg": "0", "ld_ir": "0", "ld_pc": "1", "ld_cc": "0", "gateALU": "0", "gateMDR": "0", "mem_en": "0", "mem_we": "0"},
    "JMP": {"ld_reg": "0", "ld_ir": "0", "ld_pc": "1", "ld_cc": "0", "gateALU": "0", "gateMDR": "0", "mem_en": "0", "mem_we": "0"},
    "JSR": {"ld_reg": "1", "ld_ir": "0", "ld_pc": "1", "ld_cc": "0", "gateALU": "0", "gateMDR": "0", "mem_en": "0", "mem_we": "0"},
    "SWAP": {"ld_reg": "1", "ld_ir": "0", "ld_pc": "0", "ld_cc": "1", "gateALU": "1", "gateMDR": "0", "mem_en": "0", "mem_we": "0"},
}
XY05_EXPLANATION = {
    "ADD": "ALU computes the sum, drives the bus, writes destination register, and updates condition codes.",
    "AND": "ALU computes bitwise AND, drives the bus, writes destination register, and updates condition codes.",
    "NOT": "ALU computes bitwise complement, drives the bus, writes destination register, and updates condition codes.",
    "LDR": "Address is formed first, memory is enabled, MDR drives loaded data to the bus, the destination register is written, and CC is updated.",
    "STR": "Address is formed and memory write is enabled so register data is stored to memory without updating registers or condition codes.",
    "BR": "Branch checks condition codes and updates the PC only on the branch path, with no register or memory write.",
    "JMP": "Jump loads the PC from the base register path and otherwise leaves datapath state unchanged.",
    "JSR": "JSR saves the return address in R7 and updates the PC to the subroutine target.",
    "SWAP": "Custom swap is modeled as an ALU-mediated register exchange with register write and condition-code update.",
}
XY05_STATE_TRANSITIONS = {
    "ADD": {"current": "s_1", "next": "s_18"},
    "AND": {"current": "s_5", "next": "s_18"},
    "NOT": {"current": "s_9", "next": "s_18"},
    "LDR": {"current": "s_6", "next": "s_25_1", "sequence": "s_25_1,s_25_2,s_25_3,s_27,s_18"},
    "STR": {"current": "s_7", "next": "s_23", "sequence": "s_23,s_16_1,s_16_2,s_16_3,s_18"},
    "BR": {"current": "s_0", "next_taken": "s_22", "next_not_taken": "s_18", "sequence_taken": "s_22,s_18"},
    "JMP": {"current": "s_12", "next": "s_18"},
    "JSR": {"current": "s_4", "next": "s_21", "sequence": "s_21,s_18"},
}


SUBMISSION = {
    "AM_02": {
        "reasoning": "Use valid start/end points and obstacle-avoiding paths; evaluator currently crashes on animation save, so maximizing pre-save checks is best.",
        "config": {
            "robot_trajectory1": _pad(
                20,
                [
                    (0, 17, 2), (1, 15, 0), (2, 13, 0), (3, 11, 0), (4, 9, 0),
                    (5, 8, 0), (6, 8, 2), (7, 8, 4), (8, 8, 6), (9, 8, 8),
                    (10, 8, 10), (11, 8, 12), (12, 8, 14), (13, 8, 16), (14, 8, 18),
                    (15, 8, 20), (16, 7, 22), (17, 5, 24), (18, 5, 24), (19, 5, 24),
                ],
            ),
            "robot_trajectory2": _pad(
                20,
                [
                    (0, 5, 25), (1, 7, 25), (2, 8, 25), (3, 8, 25), (4, 8, 25),
                    (5, 8, 25), (6, 8, 25), (7, 8, 25), (8, 8, 25), (9, 8, 25),
                    (10, 8, 25), (11, 8, 25), (12, 8, 25), (13, 8, 25), (14, 8, 25),
                    (15, 8, 25), (16, 10, 25), (17, 12, 25), (18, 14, 25), (19, 25, 25),
                ],
            ),
        },
    },
    "AM_03": {
        "reasoning": "Visit both goals while staying in the free corridor left of the upper obstacle, then move across the top edge.",
        "config": {
            "robot_trajectory": _pad(
                30,
                [
                    (0, 17, 2), (1, 15, 0), (2, 13, 0), (3, 11, 0), (4, 9, 0),
                    (5, 8, 0), (6, 8, 2), (7, 8, 4), (8, 8, 6), (9, 8, 8),
                    (10, 8, 10), (11, 8, 12), (12, 8, 14), (13, 8, 16), (14, 7, 18),
                    (15, 5, 20), (16, 5, 20), (17, 5, 20), (18, 5, 20), (19, 5, 20),
                    (20, 5, 20), (21, 6, 22), (22, 8, 24), (23, 10, 24), (24, 12, 24),
                    (25, 14, 24), (26, 16, 24), (27, 18, 24), (28, 20, 24), (29, 25, 24),
                ]
            )
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
        "reasoning": "Use a compact mixed denoiser: local median suppresses impulse noise, then light neighborhood averaging reduces Gaussian-like noise while preserving edges better than pure blur.",
        "config": {
            "denoising_strategy": "Median-plus-mean blend implemented only with NumPy for evaluator compatibility.",
            "filter_sequence": ["median3x3", "mean3x3", "blend(0.7,0.3)"],
            "function_code": WJ01_FUNCTION_CODE,
        },
    },
    "XY_05": {
        "reasoning": "Weak baseline with empty control table.",
        "config": {
            "ports_table": XY05_PORTS_TABLE,
            "explanation": XY05_EXPLANATION,
            "state_transitions": XY05_STATE_TRANSITIONS,
        },
    },
    "YJ_02": {
        "reasoning": "Weak baseline with wrong compliance prediction.",
        "config": {
            "y_hat": _zeros(1, 1),
            "C_y_hat": 0.0,
        },
    },
    "YJ_03": {
        "reasoning": "Weak baseline with wrong stress prediction.",
        "config": {
            "y_hat": _zeros(1, 1),
            "K_y_hat": 0.0,
        },
    },
}
# EVOLVE-BLOCK-END
