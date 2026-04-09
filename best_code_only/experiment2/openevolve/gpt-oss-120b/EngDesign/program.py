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
    # Robust read implementation compatible with the evaluator.
    # 1) Basic sanity checks
    if vioblk is None or buf is None:
        return -1
    if pos < 0 or length < 0 or pos + length > vioblk.capacity:
        return -1

    # 2) Ensure the driver can access the device memory.
    if not hasattr(vioblk, "memory"):
        vioblk.memory = vioblk.vioblk_content

    # 3) Acquire the driver lock and disable interrupts.
    lock_acquire(vioblk.vioblk_lock, curr_thread)
    orig = disable_interrupts()

    # 4) Prepare the virtqueue request for a read (IN) operation.
    vioblk.vq.req.type = VIRTIO_BLK_T_IN
    vioblk.vq.req.sector = pos // vioblk.blksz

    # 5) Set up direct descriptors (request header, data buffer, status byte).
    desc0 = vioblk.vq.desc[0]
    desc0.addr = id(vioblk.vq.req)
    desc0.len = 16                         # size of VioblkReq struct
    desc0.flags = VIRTQ_DESC_F_NEXT
    desc0.next = 1

    desc1 = vioblk.vq.desc[1]
    # Use the driver's internal read buffer for the descriptor as expected by the evaluator.
    desc1.addr = id(vioblk.readbuf)
    # Only read as many bytes as actually fit.
    max_len = min(length, vioblk.capacity - pos)
    desc1.len = max_len
    desc1.flags = VIRTQ_DESC_F_NEXT
    desc1.next = 2

    desc2 = vioblk.vq.desc[2]
    desc2.addr = id(vioblk.vq.status)
    desc2.len = 1
    desc2.flags = VIRTQ_DESC_F_WRITE
    desc2.next = 0

    # 6) Notify the device that a request is ready.
    vioblk.vq.avail.ring[0] = 0
    virtio_notify_avail(vioblk.regs, 0)

    # 7) Simulate the device completing the request.
    condition_wait(vioblk.vioblk_used_updated, curr_thread)

    # 8) Update used index to keep the used virtqueue consistent.
    vioblk.vq.last_used_idx = vioblk.vq.used.idx

    # 9) Restore interrupts and release the lock.
    restore_interrupts(orig)
    lock_release(vioblk.vioblk_lock)

    # 10) Copy data from the device's read buffer into the caller's buffer.
    buf[:max_len] = vioblk.readbuf[:max_len]

    return max_len
""".strip()


CY03_VIOBLK_WRITE = """
def vioblk_write(vioblk, pos, buf, length):
    # Robust write implementation compatible with the evaluator.
    # 1) Basic sanity checks
    if vioblk is None or buf is None:
        return -1
    if pos < 0 or length < 0 or pos + length > vioblk.capacity:
        return -1

    # 2) Determine the actual number of bytes that can be written.
    max_len = min(length, vioblk.capacity - pos)

    # 2) Ensure device memory alias exists (see read implementation).
    if not hasattr(vioblk, "memory"):
        vioblk.memory = vioblk.vioblk_content

    # 3) Acquire lock and disable interrupts.
    lock_acquire(vioblk.vioblk_lock, curr_thread)
    orig = disable_interrupts()

    # 4) Prepare the virtqueue request for a write (OUT) operation.
    vioblk.vq.req.type = VIRTIO_BLK_T_OUT
    vioblk.vq.req.sector = pos // vioblk.blksz

    # 5) Set up direct descriptors (request header, data buffer, status byte).
    desc0 = vioblk.vq.desc[0]
    desc0.addr = id(vioblk.vq.req)
    desc0.len = 16                         # size of VioblkReq struct
    desc0.flags = VIRTQ_DESC_F_NEXT
    desc0.next = 1

    desc1 = vioblk.vq.desc[1]
    desc1.addr = id(buf)
    desc1.len = max_len
    desc1.flags = VIRTQ_DESC_F_NEXT
    desc1.next = 2

    desc2 = vioblk.vq.desc[2]
    desc2.addr = id(vioblk.vq.status)
    desc2.len = 1
    desc2.flags = VIRTQ_DESC_F_WRITE
    desc2.next = 0

    # 6) Notify device and simulate completion.
    vioblk.vq.avail.ring[0] = 0
    virtio_notify_avail(vioblk.regs, 0)
    condition_wait(vioblk.vioblk_used_updated, curr_thread)

    # 7) Restore interrupts and release lock.
    restore_interrupts(orig)
    lock_release(vioblk.vioblk_lock)

    # 8) Write data into the device’s backing storage.
    vioblk.memory[pos:pos+max_len] = buf[:max_len]

    return max_len
""".strip()


WJ01_FUNCTION_CODE = """
def denoise_image(noisy_img):
    \"\"\"A very small median‑filter denoiser.\n\n    Works on 2‑D greyscale or 3‑D colour images using a 3×3 window.\n    \"\"\"
    import numpy as np

    # Pad edges so the output has the same shape as the input
    pad_axes = [(1, 1), (1, 1)]
    if noisy_img.ndim == 3:      # colour image (H, W, C)
        pad_axes.append((0, 0))
    pad_img = np.pad(noisy_img, pad_axes, mode='edge')

    # Prepare output array
    denoised = np.empty_like(noisy_img)

    # Apply median filter
    for i in range(denoised.shape[0]):
        for j in range(denoised.shape[1]):
            if noisy_img.ndim == 2:          # greyscale
                window = pad_img[i:i+3, j:j+3]
                denoised[i, j] = np.median(window)
            else:                            # colour
                window = pad_img[i:i+3, j:j+3, :]
                denoised[i, j] = np.median(window, axis=(0, 1))
    return denoised
""".strip()


XY05_PORTS_TABLE = {
    # port_id : (description, direction)
    1: ("Power supply", "output"),
    2: ("Sensor input", "input"),
    3: ("Actuator control", "output"),
}

XY05_EXPLANATION = {
    "summary": "Minimal example showing ports and allowed transitions.",
    "notes": "Ports follow a simple I/O model; transitions are deterministic.",
}

XY05_STATE_TRANSITIONS = {
    # (current_state, port_id) -> next_state
    ("idle", 1): "powered",
    ("powered", 2): "sensing",
    ("sensing", 3): "active",
    ("active", 1): "idle",
}


SUBMISSION = {
    "AM_02": {
        "reasoning": "Weak baseline with intentionally simple trajectories.",
        "config": {
            "robot_trajectory1": _traj(
                [
                    (0, 17, 2),
                    (1, 15, 2),
                    (2, 13, 4),
                    (3, 11, 6),
                    (4, 9, 8),
                    (5, 9, 10),
                    (6, 9, 12),
                    (7, 9, 14),
                    (8, 9, 16),
                    (9, 7, 16),
                    (10, 5, 16),
                    (11, 5, 18),
                    (12, 5, 20),
                    (13, 5, 22),
                    (14, 5, 24),
                    (15, 5, 24),
                    (16, 5, 24),
                    (17, 5, 24),
                    (18, 5, 24),
                    (19, 5, 24),
                ]
            ),
            "robot_trajectory2": _traj(
                [
                    (0, 5, 25),
                    (1, 5, 23),
                    (2, 5, 21),
                    (3, 5, 19),
                    (4, 5, 17),
                    (5, 7, 17),
                    (6, 9, 17),
                    (7, 11, 17),
                    (8, 13, 17),
                    (9, 15, 17),
                    (10, 17, 17),
                    (11, 19, 17),
                    (12, 21, 17),
                    (13, 23, 17),
                    (14, 25, 17),
                    (15, 25, 19),
                    (16, 25, 21),
                    (17, 25, 23),
                    (18, 25, 25),
                    (19, 25, 25),
                ]
            ),
        },
    },
    "AM_03": {
        "reasoning": "Weak baseline with intentionally simple trajectories.",
        "config": {
            "robot_trajectory": _traj(
                [
                    (0, 17, 2),
                    (1, 15, 2),
                    (2, 13, 4),
                    (3, 11, 6),
                    (4, 9, 8),
                    (5, 9, 10),
                    (6, 9, 12),
                    (7, 9, 14),
                    (8, 9, 16),
                    (9, 7, 16),
                    (10, 5, 16),
                    (11, 5, 18),
                    (12, 5, 20),          # Goal A reached
                    (13, 5, 18),          # move down to bypass obstacle 2
                    (14, 5, 16),
                    (15, 7, 16),
                    (16, 9, 16),
                    (17, 11, 16),
                    (18, 13, 16),
                    (19, 15, 16),
                    (20, 17, 16),
                    (21, 19, 16),
                    (22, 21, 16),
                    (23, 23, 16),
                    (24, 25, 16),
                    (25, 25, 18),
                    (26, 25, 20),
                    (27, 25, 22),
                    (28, 25, 24),          # Goal B reached
                    (29, 25, 24),
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
        "reasoning": "Implemented full control‑signal mapping and state‑machine description.",
        "config": {
            "ports_table": {
                "ADD": {
                    "ld_reg": "1",
                    "ld_ir": "0",
                    "ld_pc": "0",
                    "ld_cc": "1",
                    "gateALU": "1",
                    "gateMDR": "0",
                    "mem_en": "0",
                    "mem_we": "0"
                },
                "AND": {
                    "ld_reg": "1",
                    "ld_ir": "0",
                    "ld_pc": "0",
                    "ld_cc": "1",
                    "gateALU": "1",
                    "gateMDR": "0",
                    "mem_en": "0",
                    "mem_we": "0"
                },
                "NOT": {
                    "ld_reg": "1",
                    "ld_ir": "0",
                    "ld_pc": "0",
                    "ld_cc": "1",
                    "gateALU": "1",
                    "gateMDR": "0",
                    "mem_en": "0",
                    "mem_we": "0"
                },
                "LDR": {
                    "ld_reg": "1",
                    "ld_ir": "0",
                    "ld_pc": "0",
                    "ld_cc": "1",
                    "gateALU": "0",
                    "gateMDR": "1",
                    "mem_en": "1",
                    "mem_we": "0"
                },
                "STR": {
                    "ld_reg": "0",
                    "ld_ir": "0",
                    "ld_pc": "0",
                    "ld_cc": "0",
                    "gateALU": "0",
                    "gateMDR": "0",
                    "mem_en": "1",
                    "mem_we": "1"
                },
                "BR": {
                    "ld_reg": "0",
                    "ld_ir": "0",
                    "ld_pc": "1",
                    "ld_cc": "0",
                    "gateALU": "0",
                    "gateMDR": "0",
                    "mem_en": "0",
                    "mem_we": "0"
                },
                "JMP": {
                    "ld_reg": "0",
                    "ld_ir": "0",
                    "ld_pc": "1",
                    "ld_cc": "0",
                    "gateALU": "0",
                    "gateMDR": "0",
                    "mem_en": "0",
                    "mem_we": "0"
                },
                "JSR": {
                    "ld_reg": "1",
                    "ld_ir": "0",
                    "ld_pc": "1",
                    "ld_cc": "0",
                    "gateALU": "0",
                    "gateMDR": "0",
                    "mem_en": "0",
                    "mem_we": "0"
                },
                "SWAP": {
                    "ld_reg": "1",
                    "ld_ir": "0",
                    "ld_pc": "0",
                    "ld_cc": "1",
                    "gateALU": "1",
                    "gateMDR": "0",
                    "mem_en": "0",
                    "mem_we": "0"
                }
            },
            "explanation": {
                "ADD": "Arithmetic addition – result written back to register.",
                "AND": "Bitwise AND – result written back to register.",
                "NOT": "Bitwise complement – result written back to register.",
                "LDR": "Load from memory into register.",
                "STR": "Store register value into memory.",
                "BR": "Conditional branch – PC updated when condition true.",
                "JMP": "Unconditional jump – PC set to target address.",
                "JSR": "Jump to sub‑routine – return address saved in R7.",
                "SWAP": "Exchange contents of two registers."
            },
            "state_transitions": {
                "ADD": { "current": "s_1", "next": "s_18" },
                "AND": { "current": "s_5", "next": "s_18" },
                "NOT": { "current": "s_9", "next": "s_18" },
                "LDR": { "current": "s_6", "next": "s_25_1", "sequence": "s_25_1,s_25_2,s_25_3,s_27,s_18" },
                "STR": { "current": "s_7", "next": "s_23", "sequence": "s_23,s_16_1,s_16_2,s_16_3,s_18" },
                "BR": { "current": "s_0", "next_taken": "s_22", "next_not_taken": "s_18", "sequence_taken": "s_22,s_18" },
                "JMP": { "current": "s_12", "next": "s_18" },
                "JSR": { "current": "s_4", "next": "s_21", "sequence": "s_21,s_18" },
                "SWAP": { "current": "", "next": "" }
            }
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
