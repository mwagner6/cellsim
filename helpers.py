import taichi as ti
import numpy as np
@ti.kernel
def init_trajectories(n_tracked: int, max_steps: int, trajectory_positions: ti.types.ndarray(), trajectory_valid: ti.types.ndarray()):
    for i, j in ti.ndrange(n_tracked, max_steps):
        trajectory_valid[i, j] = 0
        for k in ti.static(range(3)):
            trajectory_positions[i, j, k] = 0.0