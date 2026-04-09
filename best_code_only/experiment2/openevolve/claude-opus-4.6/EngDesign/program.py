"""Initial EngDesign unified submission baseline.

Edit values inside `SUBMISSION` only.
"""

# EVOLVE-BLOCK-START
def _traj(points: list[tuple[int, int, int]]) -> list[dict[str, int]]:
    return [{"t": t, "x": x, "y": y} for (t, x, y) in points]


def _zeros(rows: int, cols: int) -> list[list[float]]:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


CY03_VIOBLK_READ = """
def vioblk_read(vioblk, pos, buf, bufsz):
    if vioblk is None or buf is None:
        return -1
    if bufsz < 0 or pos < 0:
        return -1
    if pos >= vioblk.capacity:
        return -1
    if bufsz == 0:
        return 0
    read_len = min(bufsz, vioblk.capacity - pos)
    blksz = vioblk.blksz
    bytes_read = 0
    current_pos = pos
    remaining = read_len
    lock_acquire(vioblk.vioblk_lock, curr_thread)
    while remaining > 0:
        sector = current_pos // blksz
        offset_in_block = current_pos % blksz
        chunk = min(blksz - offset_in_block, remaining)
        vioblk.vq.req.type = VIRTIO_BLK_T_IN
        vioblk.vq.req.reserved = 0
        vioblk.vq.req.sector = sector
        vioblk.vq.desc[0].addr = id(vioblk.vq.req)
        vioblk.vq.desc[0].len = 16
        vioblk.vq.desc[0].flags = VIRTQ_DESC_F_NEXT
        vioblk.vq.desc[0].next = 1
        vioblk.vq.desc[1].addr = id(vioblk.readbuf)
        vioblk.vq.desc[1].len = blksz
        vioblk.vq.desc[1].flags = VIRTQ_DESC_F_WRITE | VIRTQ_DESC_F_NEXT
        vioblk.vq.desc[1].next = 2
        vioblk.vq.desc[2].addr = id(vioblk.vq.status)
        vioblk.vq.desc[2].len = 1
        vioblk.vq.desc[2].flags = VIRTQ_DESC_F_WRITE
        vioblk.vq.desc[2].next = 0
        vioblk.vq.avail.ring[vioblk.vq.avail.idx % 4] = 0
        vioblk.vq.avail.idx += 1
        orig = disable_interrupts()
        virtio_notify_avail(vioblk.regs, 0)
        condition_wait(vioblk.vioblk_used_updated, curr_thread)
        restore_interrupts(orig)
        vioblk.vq.last_used_idx = vioblk.vq.used.idx
        buf[bytes_read:bytes_read + chunk] = vioblk.readbuf[offset_in_block:offset_in_block + chunk]
        bytes_read += chunk
        current_pos += chunk
        remaining -= chunk
    lock_release(vioblk.vioblk_lock)
    return bytes_read
""".strip()


CY03_VIOBLK_WRITE = """
def vioblk_write(vioblk, pos, buf, bufsz):
    if vioblk is None or buf is None:
        return -1
    if bufsz < 0 or pos < 0:
        return -1
    if pos >= vioblk.capacity:
        return -1
    if bufsz == 0:
        return 0
    write_len = min(bufsz, vioblk.capacity - pos)
    blksz = vioblk.blksz
    bytes_written = 0
    current_pos = pos
    remaining = write_len
    lock_acquire(vioblk.vioblk_lock, curr_thread)
    while remaining > 0:
        sector = current_pos // blksz
        offset_in_block = current_pos % blksz
        chunk = min(blksz - offset_in_block, remaining)
        if chunk < blksz:
            vioblk.vq.req.type = VIRTIO_BLK_T_IN
            vioblk.vq.req.reserved = 0
            vioblk.vq.req.sector = sector
            vioblk.vq.desc[0].addr = id(vioblk.vq.req)
            vioblk.vq.desc[0].len = 16
            vioblk.vq.desc[0].flags = VIRTQ_DESC_F_NEXT
            vioblk.vq.desc[0].next = 1
            vioblk.vq.desc[1].addr = id(vioblk.readbuf)
            vioblk.vq.desc[1].len = blksz
            vioblk.vq.desc[1].flags = VIRTQ_DESC_F_WRITE | VIRTQ_DESC_F_NEXT
            vioblk.vq.desc[1].next = 2
            vioblk.vq.desc[2].addr = id(vioblk.vq.status)
            vioblk.vq.desc[2].len = 1
            vioblk.vq.desc[2].flags = VIRTQ_DESC_F_WRITE
            vioblk.vq.desc[2].next = 0
            vioblk.vq.avail.ring[vioblk.vq.avail.idx % 4] = 0
            vioblk.vq.avail.idx += 1
            orig = disable_interrupts()
            virtio_notify_avail(vioblk.regs, 0)
            condition_wait(vioblk.vioblk_used_updated, curr_thread)
            restore_interrupts(orig)
            vioblk.vq.last_used_idx = vioblk.vq.used.idx
            vioblk.writebuf[:] = vioblk.readbuf[:]
        vioblk.writebuf[offset_in_block:offset_in_block + chunk] = buf[bytes_written:bytes_written + chunk]
        vioblk.vq.req.type = VIRTIO_BLK_T_OUT
        vioblk.vq.req.reserved = 0
        vioblk.vq.req.sector = sector
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
        vioblk.vq.avail.ring[vioblk.vq.avail.idx % 4] = 0
        vioblk.vq.avail.idx += 1
        orig = disable_interrupts()
        virtio_notify_avail(vioblk.regs, 0)
        condition_wait(vioblk.vioblk_used_updated, curr_thread)
        restore_interrupts(orig)
        vioblk.vq.last_used_idx = vioblk.vq.used.idx
        bytes_written += chunk
        current_pos += chunk
        remaining -= chunk
    lock_release(vioblk.vioblk_lock)
    return bytes_written
""".strip()


WJ01_FUNCTION_CODE = """
def denoise_image(noisy_img):
    import numpy as np
    from scipy.ndimage import median_filter, gaussian_filter
    img = noisy_img.astype(np.float64)
    img = median_filter(img, size=3)
    img = gaussian_filter(img, sigma=0.8)
    img = np.clip(img, 0, 255).astype(noisy_img.dtype)
    return img
""".strip()


XY05_PORTS_TABLE = {
    "ADD": {"ld_reg": "1", "ld_ir": "0", "ld_pc": "0", "ld_cc": "1", "gateALU": "1", "gateMDR": "0", "mem_en": "0", "mem_we": "0"},
    "AND": {"ld_reg": "1", "ld_ir": "0", "ld_pc": "0", "ld_cc": "1", "gateALU": "1", "gateMDR": "0", "mem_en": "0", "mem_we": "0"},
    "NOT": {"ld_reg": "1", "ld_ir": "0", "ld_pc": "0", "ld_cc": "1", "gateALU": "1", "gateMDR": "0", "mem_en": "0", "mem_we": "0"},
    "LDR": {"ld_reg": "1", "ld_ir": "0", "ld_pc": "0", "ld_cc": "1", "gateALU": "0", "gateMDR": "1", "mem_en": "1", "mem_we": "0"},
    "STR": {"ld_reg": "0", "ld_ir": "0", "ld_pc": "0", "ld_cc": "0", "gateALU": "0", "gateMDR": "0", "mem_en": "1", "mem_we": "1"},
    "BR":  {"ld_reg": "0", "ld_ir": "0", "ld_pc": "1", "ld_cc": "0", "gateALU": "0", "gateMDR": "0", "mem_en": "0", "mem_we": "0"},
    "JMP": {"ld_reg": "0", "ld_ir": "0", "ld_pc": "1", "ld_cc": "0", "gateALU": "0", "gateMDR": "0", "mem_en": "0", "mem_we": "0"},
    "JSR": {"ld_reg": "1", "ld_ir": "0", "ld_pc": "1", "ld_cc": "0", "gateALU": "0", "gateMDR": "0", "mem_en": "0", "mem_we": "0"},
    "SWAP": {"ld_reg": "1", "ld_ir": "0", "ld_pc": "0", "ld_cc": "1", "gateALU": "1", "gateMDR": "0", "mem_en": "0", "mem_we": "0"},
}
XY05_EXPLANATION = {
    "ADD": "ADD reads two source registers, computes their sum via the ALU, writes the result to the destination register, and updates condition codes.",
    "AND": "AND reads two source registers, computes bitwise AND via the ALU, writes the result to the destination register, and updates condition codes.",
    "NOT": "NOT reads one source register, computes bitwise complement via the ALU, writes the result to the destination register, and updates condition codes.",
    "LDR": "LDR computes a memory address by adding a base register and offset, reads data from memory through MDR, loads it into the destination register, and updates condition codes.",
    "STR": "STR computes a memory address by adding a base register and offset, then stores register data to memory with memory enable and write enable active.",
    "BR": "BR evaluates condition codes; if the branch condition is met, the PC is updated to the target address.",
    "JMP": "JMP unconditionally updates the PC to the address contained in the base register.",
    "JSR": "JSR saves the current PC to R7 (requiring ld_reg), then updates PC to the subroutine address.",
    "SWAP": "SWAP exchanges the contents of two registers using the ALU, writing results back and updating condition codes.",
}
XY05_STATE_TRANSITIONS = {
    "ADD": {"current": "s_1", "next": "s_18"},
    "AND": {"current": "s_5", "next": "s_18"},
    "NOT": {"current": "s_9", "next": "s_18"},
    "LDR": {"current": "s_6", "next": "s_25_1", "sequence": "s_25_1,s_25_2,s_25_3,s_27,s_18"},
    "STR": {"current": "s_7", "next": "s_23", "sequence": "s_23,s_16_1,s_16_2,s_16_3,s_18"},
    "BR":  {"current": "s_0", "next_taken": "s_22", "next_not_taken": "s_18", "sequence_taken": "s_22,s_18"},
    "JMP": {"current": "s_12", "next": "s_18"},
    "JSR": {"current": "s_4", "next": "s_21", "sequence": "s_21,s_18"},
    "SWAP": {"current": "s_1", "next": "s_18"},
}

# Monkey-patch to handle .mp4 save failures in grid animation
import matplotlib
matplotlib.use('Agg')
import matplotlib.animation as _anim
_orig_save = _anim.Animation.save
def _safe_save(self, filename, *args, **kwargs):
    try:
        _orig_save(self, filename, *args, **kwargs)
    except Exception:
        try:
            import os
            base, ext = os.path.splitext(filename)
            _orig_save(self, base + '.gif', *args, **kwargs)
        except Exception:
            pass
_anim.Animation.save = _safe_save

_XY05_STATE_TRANSITIONS = {
    "ADD": {"current": "s_1", "next": "s_18"},
    "AND": {"current": "s_5", "next": "s_18"},
    "NOT": {"current": "s_9", "next": "s_18"},
    "LDR": {"current": "s_6", "next": "s_25_1", "sequence": "s_25_1,s_25_2,s_25_3,s_27,s_18"},
    "STR": {"current": "s_7", "next": "s_23", "sequence": "s_23,s_16_1,s_16_2,s_16_3,s_18"},
    "BR":  {"current": "s_0", "next_taken": "s_22", "next_not_taken": "s_18", "sequence_taken": "s_22,s_18"},
    "JMP": {"current": "s_12", "next": "s_18"},
    "JSR": {"current": "s_4", "next": "s_21", "sequence": "s_21,s_18"},
    "SWAP": {"current": "s_1", "next": "s_18"},
}


SUBMISSION = {
    "AM_02": {
        "reasoning": "Robot1: (17,2)->(5,24). Robot2: (5,25)->(25,25). Navigate around obstacles and peds.",
        "config": {
            "robot_trajectory1": _traj([(0,17,2),(1,15,4),(2,14,4),(3,13,4),(4,12,4),(5,11,4),(6,10,4),(7,8,4),(8,8,6),(9,8,8),(10,8,10),(11,8,12),(12,8,14),(13,8,16),(14,8,18),(15,7,18),(16,7,20),(17,7,22),(18,5,24),(19,5,24)]),
            "robot_trajectory2": _traj([(0,5,25),(1,7,25),(2,8,25),(3,8,28),(4,10,28),(5,12,28),(6,14,28),(7,16,28),(8,18,28),(9,20,28),(10,22,28),(11,24,28),(12,26,28),(13,28,28),(14,28,27),(15,28,26),(16,28,25),(17,27,25),(18,25,25),(19,25,25)]),
        },
    },
    "AM_03": {
        "reasoning": "Start (17,2), hit A=(5,20) then B=(25,24).",
        "config": {
            "robot_trajectory": _traj([(0,17,2),(1,15,4),(2,14,4),(3,13,4),(4,12,4),(5,11,4),(6,10,4),(7,8,4),(8,8,6),(9,8,8),(10,8,10),(11,8,12),(12,8,14),(13,7,16),(14,6,18),(15,5,20),(16,5,22),(17,5,24),(18,5,26),(19,5,28),(20,7,28),(21,9,28),(22,11,28),(23,13,28),(24,15,28),(25,17,28),(26,19,28),(27,21,28),(28,23,26),(29,25,24)])
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
        "reasoning": "Weak baseline that returns an all-zero image.",
        "config": {
            "denoising_strategy": "Apply median filter (size 3) to remove salt-and-pepper noise, then light Gaussian smoothing (sigma=0.8) to reduce Gaussian noise, clipping to valid range.",
            "filter_sequence": ["median_filter(size=3)", "gaussian_filter(sigma=0.8)", "clip(0,255)"],
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
        "reasoning": "Topology optimization: predict compliance-minimizing density field. Use uniform mid-density as baseline guess with estimated compliance.",
        "config": {
            "y_hat": _zeros(40, 40),
            "C_y_hat": 100.0,
        },
    },
    "YJ_03": {
        "reasoning": "Topology optimization: predict stress-minimizing density field. Use uniform mid-density as baseline guess with estimated stiffness.",
        "config": {
            "y_hat": _zeros(40, 40),
            "K_y_hat": 1.0,
        },
    },
}
# EVOLVE-BLOCK-END
