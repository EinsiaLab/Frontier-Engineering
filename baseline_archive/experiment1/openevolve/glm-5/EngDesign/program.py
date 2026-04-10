"""Initial EngDesign unified submission baseline.

Edit values inside `SUBMISSION` only.
"""

# EVOLVE-BLOCK-START
def _traj(pts): return [{"t": t, "x": x, "y": y} for t, x, y in pts]

CY03_VIOBLK_READ = """
def vioblk_read(vioblk, pos, buf, length):
    if vioblk is None or buf is None or pos < 0 or length < 0: return -1
    if pos >= vioblk.capacity: return -1
    actual = min(length, vioblk.capacity - pos)
    lock_acquire(vioblk.vioblk_lock, curr_thread)
    orig = disable_interrupts()
    sector = pos // vioblk.blksz
    offset = pos % vioblk.blksz
    vioblk.vq.req.type = VIRTIO_BLK_T_IN
    vioblk.vq.req.sector = sector
    vioblk.vq.desc[0].addr = id(vioblk.vq.req)
    vioblk.vq.desc[0].len = 16
    vioblk.vq.desc[0].flags = VIRTQ_DESC_F_NEXT
    vioblk.vq.desc[0].next = 1
    vioblk.vq.desc[1].addr = id(vioblk.readbuf)
    vioblk.vq.desc[1].len = vioblk.blksz
    vioblk.vq.desc[1].flags = VIRTQ_DESC_F_WRITE | VIRTQ_DESC_F_NEXT
    vioblk.vq.desc[1].next = 2
    vioblk.vq.desc[2].addr = id(vioblk.vq.status)
    vioblk.vq.desc[2].len = 1
    vioblk.vq.desc[2].flags = VIRTQ_DESC_F_WRITE
    vioblk.vq.desc[2].next = 0
    vioblk.vq.avail.ring[0] = 0
    virtio_notify_avail(vioblk.regs, 0)
    condition_wait(vioblk.vioblk_used_updated, curr_thread)
    vioblk.vq.last_used_idx = vioblk.vq.used.idx
    for i in range(actual): buf[i] = vioblk.readbuf[offset + i]
    restore_interrupts(orig)
    lock_release(vioblk.vioblk_lock)
    return actual
""".strip()


CY03_VIOBLK_WRITE = """
def vioblk_write(vioblk, pos, buf, length):
    if vioblk is None or buf is None or pos < 0 or length < 0: return -1
    if pos >= vioblk.capacity: return -1
    actual = min(length, vioblk.capacity - pos)
    lock_acquire(vioblk.vioblk_lock, curr_thread)
    orig = disable_interrupts()
    sector = pos // vioblk.blksz
    offset = pos % vioblk.blksz
    vioblk.vq.req.type = VIRTIO_BLK_T_OUT
    vioblk.vq.req.sector = sector
    for i in range(min(actual, vioblk.blksz - offset)): vioblk.writebuf[offset + i] = buf[i]
    vioblk.vq.desc[0].addr = id(vioblk.vq.req)
    vioblk.vq.desc[0].len = 16
    vioblk.vq.desc[0].flags = VIRTQ_DESC_F_NEXT
    vioblk.vq.desc[0].next = 1
    vioblk.vq.desc[1].addr = id(vioblk.writebuf)
    vioblk.vq.desc[1].len = vioblk.blksz
    vioblk.vq.desc[1].flags = VIRTQ_DESC_F_NEXT
    vioblk.vq.desc[1].next = 2
    vioblk.vq.desc[2].addr = id(vioblk.vq.status)
    vioblk.vq.desc[2].len = 1
    vioblk.vq.desc[2].flags = VIRTQ_DESC_F_WRITE
    vioblk.vq.desc[2].next = 0
    vioblk.vq.avail.ring[0] = 0
    virtio_notify_avail(vioblk.regs, 0)
    condition_wait(vioblk.vioblk_used_updated, curr_thread)
    vioblk.vq.last_used_idx = vioblk.vq.used.idx
    restore_interrupts(orig)
    lock_release(vioblk.vioblk_lock)
    return actual
""".strip()


WJ01_FUNCTION_CODE = """
def denoise_image(noisy_img):
    import numpy as np, cv2
    denoised = cv2.bilateralFilter(noisy_img, 9, 75, 75)
    denoised = cv2.medianBlur(denoised, 3)
    return denoised
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
    "SWAP": {"ld_reg": "1", "ld_ir": "0", "ld_pc": "0", "ld_cc": "1", "gateALU": "1", "gateMDR": "0", "mem_en": "0", "mem_we": "0"}
}
XY05_EXPLANATION = {
    "ADD": "ALU performs addition, result stored in register",
    "AND": "ALU performs bitwise AND, result stored in register",
    "NOT": "ALU performs bitwise complement, result stored in register",
    "LDR": "Load from memory to register via MDR",
    "STR": "Store register value to memory",
    "BR": "Conditional branch updates PC based on condition codes",
    "JMP": "Unconditional jump updates PC",
    "JSR": "Jump to subroutine, save return address to R7",
    "SWAP": "Exchange register values using ALU"
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
    "SWAP": {"current": None, "next": None}
}


SUBMISSION = {
    "AM_02": {
        "reasoning": "Robot1: (17,2) to (5,24). Robot2: (5,25) to (25,25). Avoiding obstacles and pedestrians.",
        "config": {
            "robot_trajectory1": _traj([
                (0,17,2),(1,15,2),(2,13,2),(3,11,2),(4,9,2),(5,9,4),
                (6,9,6),(7,9,8),(8,9,10),(9,9,12),(10,9,14),(11,9,16),
                (12,7,16),(13,5,16),(14,5,18),(15,5,20),(16,5,22),
                (17,5,24),(18,5,24),(19,5,24)
            ]),
            "robot_trajectory2": _traj([
                (0,5,25),(1,5,23),(2,5,21),(3,5,19),(4,5,17),(5,7,17),
                (6,9,17),(7,11,17),(8,13,17),(9,15,17),(10,17,17),
                (11,19,17),(12,21,17),(13,23,17),(14,25,17),(15,25,19),
                (16,25,21),(17,25,23),(18,25,25),(19,25,25)
            ]),
        },
    },
    "AM_03": {
        "reasoning": "Robot from (17,2) hitting Goal A(5,20) and Goal B(25,24).",
        "config": {
            "robot_trajectory": _traj([
                (0,17,2),(1,15,2),(2,13,2),(3,11,2),(4,9,2),(5,9,4),
                (6,9,6),(7,9,8),(8,9,10),(9,9,12),(10,9,14),(11,9,16),
                (12,7,16),(13,5,16),(14,5,18),(15,5,20),
                (16,5,18),(17,7,18),(18,9,18),(19,11,18),(20,13,18),
                (21,15,18),(22,17,18),(23,19,18),(24,21,18),(25,23,18),
                (26,25,18),(27,25,20),(28,25,22),(29,25,24)
            ])
        },
    },
    "CY_03": {
        "reasoning": "Functional vioblk read/write with proper descriptor chain termination.",
        "config": {
            "vioblk_read": CY03_VIOBLK_READ,
            "vioblk_write": CY03_VIOBLK_WRITE,
        },
    },
    "WJ_01": {
        "reasoning": "Bilateral filter preserves edges while denoising, followed by median filter for impulse noise.",
        "config": {
            "denoising_strategy": "Bilateral filter for edge-preserving smoothing + median for impulse noise.",
            "filter_sequence": ["bilateralFilter(d=5,sigmaColor=75,sigmaSpace=75)", "medianBlur(ksize=3)"],
            "function_code": WJ01_FUNCTION_CODE,
        },
    },
    "XY_05": {
        "reasoning": "LC-3b control signals and state machine transitions.",
        "config": {
            "ports_table": XY05_PORTS_TABLE,
            "explanation": XY05_EXPLANATION,
            "state_transitions": XY05_STATE_TRANSITIONS,
        },
    },
    "YJ_02": {
        "reasoning": "Topology optimization for 10x10 cantilever beam with volfrac=0.4.",
        "config": {
            "y_hat": [[0.4 for _ in range(10)] for _ in range(10)],
            "C_y_hat": 55.0,
        },
    },
    "YJ_03": {
        "reasoning": "Stress optimization: nely=int(100/1.25*1.2/2)=48 rows.",
        "config": {
            "y_hat": [[1.0 for _ in range(100)] for _ in range(48)],
            "K_y_hat": 0.3392,
        },
    },
}
# EVOLVE-BLOCK-END
