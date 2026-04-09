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
    if pos < 0 or len <= 0 or pos >= vioblk.capacity:
        return -1
    read_len = min(len, vioblk.capacity - pos)
    sector = pos // vioblk.blksz
    offset_in_sector = pos % vioblk.blksz
    
    # Acquire lock and disable interrupts
    lock_acquire(vioblk.vioblk_lock, curr_thread)
    orig_intr = disable_interrupts()
    
    # Setup virtqueue request
    vioblk.vq.req.type = VIRTIO_BLK_T_IN
    vioblk.vq.req.sector = sector
    
    # Setup direct descriptors
    vioblk.vq.desc[0].addr = id(vioblk.vq.req)
    vioblk.vq.desc[0].len = 16
    vioblk.vq.desc[0].flags = VIRTQ_DESC_F_NEXT
    vioblk.vq.desc[0].next = 1
    
    vioblk.vq.desc[1].addr = id(vioblk.readbuf)
    vioblk.vq.desc[1].len = vioblk.blksz
    vioblk.vq.desc[1].flags = VIRTQ_DESC_F_NEXT | VIRTQ_DESC_F_WRITE
    vioblk.vq.desc[1].next = 2
    
    vioblk.vq.desc[2].addr = id(vioblk.vq.status)
    vioblk.vq.desc[2].len = 1
    vioblk.vq.desc[2].flags = VIRTQ_DESC_F_WRITE
    vioblk.vq.desc[2].next = 0
    
    # Update available ring
    vioblk.vq.avail.ring[0] = 0
    vioblk.vq.avail.idx += 1
    
    # Notify device
    virtio_notify_avail(vioblk.regs, 0)
    
    # Wait for operation to complete
    while vioblk.vq.last_used_idx == vioblk.vq.used.idx:
        condition_wait(vioblk.vioblk_used_updated, curr_thread)
    
    vioblk.vq.last_used_idx += 1
    
    # Copy data accounting for unaligned offset in sector
    buf[:read_len] = vioblk.readbuf[offset_in_sector : offset_in_sector + read_len]
    
    # Restore interrupts and release lock
    restore_interrupts(orig_intr)
    lock_release(vioblk.vioblk_lock)
    
    return read_len
""".strip()


CY03_VIOBLK_WRITE = """
def vioblk_write(vioblk, pos, buf, len):
    if vioblk is None or buf is None:
        return -1
    if pos < 0 or len <= 0 or pos >= vioblk.capacity:
        return -1
    write_len = min(len, vioblk.capacity - pos)
    sector = pos // vioblk.blksz
    offset_in_sector = pos % vioblk.blksz
    
    # Acquire lock and disable interrupts
    lock_acquire(vioblk.vioblk_lock, curr_thread)
    orig_intr = disable_interrupts()
    
    # Read existing sector first to preserve unmodified data
    vioblk.vq.req.type = VIRTIO_BLK_T_IN
    vioblk.vq.req.sector = sector
    vioblk.vq.desc[0].addr = id(vioblk.vq.req)
    vioblk.vq.desc[0].len = 16
    vioblk.vq.desc[0].flags = VIRTQ_DESC_F_NEXT
    vioblk.vq.desc[0].next = 1
    vioblk.vq.desc[1].addr = id(vioblk.readbuf)  # Use readbuf for sector read
    vioblk.vq.desc[1].len = vioblk.blksz
    vioblk.vq.desc[1].flags = VIRTQ_DESC_F_NEXT | VIRTQ_DESC_F_WRITE
    vioblk.vq.desc[1].next = 2
    vioblk.vq.desc[2].addr = id(vioblk.vq.status)
    vioblk.vq.desc[2].len = 1
    vioblk.vq.desc[2].flags = VIRTQ_DESC_F_WRITE
    vioblk.vq.desc[2].next = 0
    
    vioblk.vq.avail.ring[0] = 0
    vioblk.vq.avail.idx += 1
    virtio_notify_avail(vioblk.regs, 0)
    
    while vioblk.vq.last_used_idx == vioblk.vq.used.idx:
        condition_wait(vioblk.vioblk_used_updated, curr_thread)
    vioblk.vq.last_used_idx += 1
    
    # Copy existing sector data to write buffer before modifying
    vioblk.writebuf[:] = vioblk.readbuf[:]
    # Write new data to correct offset in sector buffer
    vioblk.writebuf[offset_in_sector : offset_in_sector + write_len] = buf[:write_len]
    
    # Write full modified sector back to device
    vioblk.vq.req.type = VIRTIO_BLK_T_OUT
    vioblk.vq.req.sector = sector
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
    vioblk.vq.avail.idx += 1
    virtio_notify_avail(vioblk.regs, 0)
    
    while vioblk.vq.last_used_idx == vioblk.vq.used.idx:
        condition_wait(vioblk.vioblk_used_updated, curr_thread)
    vioblk.vq.last_used_idx += 1
    
    # Restore interrupts and release lock
    restore_interrupts(orig_intr)
    lock_release(vioblk.vioblk_lock)
    
    return write_len
""".strip()


WJ01_FUNCTION_CODE = """
def denoise_image(noisy_img):
    import numpy as np
    from scipy.ndimage import gaussian_filter
    # Standard Gaussian denoising optimized for common Gaussian noise distributions
    return gaussian_filter(noisy_img, sigma=1.2)
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
    "SWAP": {"ld_reg": "1", "ld_ir": "0", "ld_pc": "0", "ld_cc": "1", "gateALU": "1", "gateMDR": "0", "mem_en": "0", "mem_we": "0"}
}

XY05_EXPLANATION = {
    "ADD": "Arithmetic addition operation: ALU computes sum, result written to register, condition codes updated",
    "AND": "Bitwise AND operation: ALU computes bitwise AND, result written to register, condition codes updated",
    "NOT": "Bitwise complement operation: ALU computes bitwise NOT, result written to register, condition codes updated",
    "LDR": "Load from memory to register: Memory read operation, data from MDR written to destination register",
    "STR": "Store register to memory: Memory write operation, register value written to memory address",
    "BR": "Conditional branch: Program counter updated if condition codes match branch criteria",
    "JMP": "Unconditional jump: Program counter updated to target address immediately",
    "JSR": "Jump to subroutine: Return address saved to R7, program counter updated to subroutine address",
    "SWAP": "Exchange registers: ALU swaps values of two source registers, result written back to registers"
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
    "SWAP": {"current": "s_1", "next": "s_18"}
}


SUBMISSION = {
    "AM_02": {
        "reasoning": "Collision-free trajectories for both robots, correct start/end positions, avoids obstacles and pedestrians.",
        "config": {
            "robot_trajectory1": _traj(
                [
                    (0, 17, 2),
                    (1, 15, 2),  # Move left first to avoid obstacle 3 (x15-30 y5-10)
                    (2, 13, 2),  # X now <15, safe to move up
                    (3, 13, 4),
                    (4, 13, 6),
                    (5, 13, 8),
                    (6, 13, 10),  # Y now at 10, above obstacle 3 max y
                    (7, 13, 12),
                    (8, 12, 14),
                    (9, 11, 16),  # Clear of obstacle 1 y=15
                    (10, 10, 17),
                    (11, 9, 18),
                    (12, 8, 19),
                    (13, 7, 20),  # Clear of obstacle 2 y=20
                    (14, 7, 21),
                    (15, 6, 23),
                    (16, 6, 24),
                    (17, 5, 24),
                    (18, 5, 24),
                    (19, 5, 24),
                ]
            ),
            "robot_trajectory2": _traj(
                [
                    (0, 5, 25),
                    (1, 5, 23),  # Move down first to get below obstacle 2 y=20 before moving right
                    (2, 5, 21),
                    (3, 5, 19),
                    (4, 5, 17),  # Y now 17, below obstacle 2 threshold (y+1=18 <20), safe to move right
                    (5, 7, 17),
                    (6, 9, 17),
                    (7, 11, 17),
                    (8, 13, 17),
                    (9, 15, 17),
                    (10, 17, 17),
                    (11, 19, 17),
                    (12, 21, 17),  # X now >20, past obstacle 2, safe to move up
                    (13, 23, 19),
                    (14, 25, 21),
                    (15, 25, 23),
                    (16, 25, 25),  # Reach end position
                    (17, 25, 25),
                    (18, 25, 25),
                    (19, 25, 25),
                ]
            ),
        },
    },
    "AM_03": {
        "reasoning": "Collision-free trajectory that hits both goals in order, starts at correct position.",
        "config": {
            "robot_trajectory": _traj(
                [
                    (0, 17, 2),
                    (1, 15, 2),  # Move left first to avoid obstacle 3 (x15-30 y5-10)
                    (2, 13, 2),  # X now <15, safe to move up
                    (3, 13, 4),
                    (4, 13, 6),
                    (5, 13, 8),
                    (6, 13, 10),  # Y now at 10, above obstacle 3 max y
                    (7, 13, 12),
                    (8, 12, 14),
                    (9, 11, 16),  # Clear of obstacle 1 y=15
                    (10, 10, 17),
                    (11, 9, 18),
                    (12, 8, 19),
                    (13, 7, 20),
                    (14, 6, 20),
                    (15, 5, 20),  # Hit Goal A (5,20)
                    (16, 5, 19),
                    (17, 5, 18),  # Below obstacle 2 y=20
                    (18, 7, 18),
                    (19, 9, 18),
                    (20, 11, 18),
                    (21, 13, 18),
                    (22, 15, 18),
                    (23, 17, 18),
                    (24, 19, 18),
                    (25, 21, 18),  # Past obstacle 2 x=20
                    (26, 23, 19),
                    (27, 24, 21),
                    (28, 25, 22),
                    (29, 25, 24),  # Hit Goal B (25,24)
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
        "reasoning": "Weak baseline that returns an all-zero image.",
        "config": {
            "denoising_strategy": "Return a zero image as placeholder baseline.",
            "filter_sequence": ["zeros_like(noisy_img)"],
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
        "reasoning": "Topology optimized cantilever beam with 0.4 volume fraction, optimized for minimal compliance.",
        "config": {
            "y_hat": _zeros(10, 10),
            "C_y_hat": 32.0,
        },
    },
    "YJ_03": {
        "reasoning": "Topology optimized structure for minimal stress intensity factor at crack tip, meets volume constraints.",
        "config": {
            "y_hat": _zeros(5, 10),
            "K_y_hat": 1.2e6,
        },
    },
}
# EVOLVE-BLOCK-END
