import taichi as ti
import numpy as np
from typing import Tuple

from beamgen import EllipticalBeamGenerator
from defocus_microscope import DefocusMicroscope
from helpers import init_trajectories
from fieldgen import VolumeGenerator

@ti.data_oriented
class Simulation:

    def __init__(self, N: int = 128):
        self.N: int = N
        self.volume_shape: Tuple[int, int, int] = (N, N, N)
        self.tracking = False
        self.microscopes = []
        self.typemap = {"empty": 0}

        ti.init(ti.gpu)

        # Keep volume-related as fields since they're accessed via self in kernels
        self.volume = ti.field(dtype=ti.f32, shape=self.volume_shape)
        self.interactions = ti.field(dtype=ti.f32, shape=self.volume_shape)
        self.out = ti.field(dtype=ti.f32, shape=self.volume_shape)

    def init_types(self, types, scatter_prec):
        self.spacegen = VolumeGenerator()
        for i, type in enumerate(types):
            self.spacegen.initType(type["p_absorb"], type["p_scatter"], type["scatter_map"], type["r_index"])
            self.typemap[type["name"]] = i + 1

        self.type_n = len(types) + 1
        self.spacegen.resolveTypes(scatter_prec)

        self.types_ti = ti.field(dtype=ti.f32, shape=(self.type_n, 3))
        self.scatters_ti = ti.field(dtype=ti.u16, shape=(self.type_n, scatter_prec))

    def initSpheres(self, spheres):
        spheres["types"] = [self.typemap[t] for t in spheres["types"]]
        self.spacegen.initSpheres(spheres, self.volume, self.types_ti, self.scatters_ti, self.N)

    def project_volume(self, axes):
        outputs = {}
        volume_np = self.volume.to_numpy()
        for axis in axes:
            if axis == "x":
                projection = volume_np.sum(axis=2)
                outputs[axis] = projection[:, :, np.newaxis]
            elif axis == "y":
                projection = volume_np.sum(axis=1)
                outputs[axis] = projection[:, np.newaxis, :]
            elif axis == "z":
                projection = volume_np.sum(axis=0)
                outputs[axis] = projection[np.newaxis, :, :]

        return outputs
    
    def volume_to_np(self):
        self.volume_np = self.volume.to_numpy()

    def vectorize_vol(self):
        self.vectorized_vol = np.vectorize(self.volume_np)

    def absorption_np(self):
        def absorption(m):
            return self.types_ti[m, 0]
        
        return absorption(self.vectorized_vol)
    
    def scatter_np(self):
        def scatter(m):
            return self.types_ti[m, 1]
        
        return scatter(self.vectorized_vol)
    
    def refraction_np(self):
        def refraction(m):
            return self.types_ti[m, 2]
        
        return refraction(self.vectorized_vol)

    def init_beam_generator(self,
                            central_point,
                            distance,
                            theta,
                            phi,
                            mask_diameter,
                            sigma_a,
                            sigma_b,
                            divergence_a,
                            divergence_b,
                            divergence_sigma_a,
                            divergence_sigma_b
                            ):
        self.beamgen = EllipticalBeamGenerator(
            central_point=central_point,
            distance=distance,
            theta=theta,
            phi=phi,
            mask_diameter=mask_diameter,
            sigma_a=sigma_a,
            sigma_b=sigma_b,
            divergence_a=divergence_a,
            divergence_b=divergence_b,
            divergence_sigma_a=divergence_sigma_a,
            divergence_sigma_b=divergence_sigma_b
        )

    def init_photons(self, n):
        self.n_photons = n

        # Use ndarrays instead of fields for photon data
        self.positions = ti.ndarray(dtype=ti.f32, shape=(n, 3))
        self.directions = ti.ndarray(dtype=ti.f32, shape=(n, 3))
        self.intensities = ti.ndarray(dtype=ti.f32, shape=(n,))
        self.entered = ti.ndarray(dtype=ti.i8, shape=(n,))
        self.exited = ti.ndarray(dtype=ti.i8, shape=(n,))
        self.bounces = ti.ndarray(dtype=ti.i32, shape=(n,))
        self.last_scatter_pos = ti.ndarray(dtype=ti.f32, shape=(n, 3))

        # Generate photons directly into ndarrays
        self.beamgen._generate_and_filter_samples(self.positions, self.directions, 0.0)

        # Initialize other arrays
        self._init_photon_arrays(self.intensities, self.entered, self.exited,
                                    self.bounces, self.last_scatter_pos, n)

    @ti.kernel
    def _init_photon_arrays(self, intensities: ti.types.ndarray(),
                            entered: ti.types.ndarray(),
                            exited: ti.types.ndarray(),
                            bounces: ti.types.ndarray(),
                            last_scatter_pos: ti.types.ndarray(),
                            n: ti.i32):
        for i in range(n):
            intensities[i] = 0.01
            entered[i] = 0
            exited[i] = 0
            bounces[i] = 0
            for j in ti.static(range(3)):
                last_scatter_pos[i, j] = 0.0


    def init_tracking(self, n_tracked, steps):
        self.tracking = True
        self.track_steps = steps
        self.n_tracked = n_tracked
        self.trajectory_positions = ti.ndarray(dtype=ti.f32, shape=(n_tracked, steps, 3))
        self.trajectory_valid = ti.ndarray(dtype=ti.i8, shape=(n_tracked, steps))
        self.step_idx = 0
        init_trajectories(n_tracked, steps, self.trajectory_positions, self.trajectory_valid)


    @ti.kernel
    def _propagate_photons(self, positions: ti.types.ndarray(),
                            directions: ti.types.ndarray(),
                            exited: ti.types.ndarray(),
                            step: ti.f32):
        for i in range(positions.shape[0]):
            if exited[i] == 0:
                for j in ti.static(range(3)):
                    positions[i, j] += directions[i, j] * step

    @ti.kernel
    def _propagate_photons_with_tracking(self, positions: ti.types.ndarray(),
                                        directions: ti.types.ndarray(),
                                        exited: ti.types.ndarray(),
                                        trajectory_positions: ti.types.ndarray(),
                                        trajectory_valid: ti.types.ndarray(),
                                        step_idx: ti.i32,
                                        n_tracked: ti.i32,
                                        step: ti.f32):
        for i in range(positions.shape[0]):
            if exited[i] == 0:
                if i < n_tracked:
                    for k in ti.static(range(3)):
                        trajectory_positions[i, step_idx, k] = positions[i, k]
                    trajectory_valid[i, step_idx] = 1

                for j in ti.static(range(3)):
                    positions[i, j] += directions[i, j] * step

    def propagate_photons(self, step: float):
        if self.tracking:
            self._propagate_photons_with_tracking(
                self.positions, self.directions, self.exited,
                self.trajectory_positions, self.trajectory_valid,
                self.step_idx, self.n_tracked, step
            )
            self.step_idx += 1
        else:
            self._propagate_photons(self.positions, self.directions, self.exited, step)


    @ti.kernel
    def _interact_photons(self,
                            positions: ti.types.ndarray(),
                            directions: ti.types.ndarray(),
                            intensities: ti.types.ndarray(),
                            entered: ti.types.ndarray(),
                            exited: ti.types.ndarray(),
                            bounces: ti.types.ndarray(),
                            last_scatter_pos: ti.types.ndarray(),
                            volume: ti.template(),
                            typemap: ti.template(),
                            scattermap: ti.template(),
                            interactions: ti.template(),
                            out: ti.template(),
                            N: ti.i32):
        for i in range(positions.shape[0]):
            if intensities[i] > 0:
                x = ti.cast(positions[i, 0], ti.i32)
                y = ti.cast(positions[i, 1], ti.i32)
                z = ti.cast(positions[i, 2], ti.i32)

                if 0 <= x < N and 0 <= y < N and 0 <= z < N:
                    if entered[i] == 0:
                        entered[i] = 1

                    rand = ti.random()

                    if rand < typemap[volume[x, y, z], 0]:
                        interactions[x, y, z] += intensities[i]
                        intensities[i] = 0

                    elif rand < typemap[volume[x, y, z], 0] + typemap[volume[x, y, z], 1]:
                        phi = 2*3.14159265359*ti.random()
                        u = 2*ti.random()-1
                        directions[i, 0] = ti.sqrt(1-u*u) * ti.cos(phi)
                        directions[i, 1] = ti.sqrt(1-u*u) * ti.sin(phi)
                        directions[i, 2] = u
                        bounces[i] += 1

                        last_scatter_pos[i, 0] = positions[i, 0]
                        last_scatter_pos[i, 1] = positions[i, 1]
                        last_scatter_pos[i, 2] = positions[i, 2]

                elif entered[i] == 1 and intensities[i] > 0:
                    if x < 0 or x >= N or y < 0 or y >= N or z < 0 or z >= N:
                        exited[i] = 1
                        entered[i] = 0
                        if z < N:
                            x_clamped = ti.max(0, ti.min(N-1, x))
                            y_clamped = ti.max(0, ti.min(N-1, y))
                            z_clamped = ti.max(0, ti.min(N-1, z))
                            out[x_clamped, y_clamped, z_clamped] += intensities[i]

    def interact_photons(self):
        self._interact_photons(
            self.positions, self.directions, self.intensities,
            self.entered, self.exited, self.bounces, self.last_scatter_pos,
            self.volume, self.types_ti, self.scatters_ti, self.interactions, self.out, self.N
        )


    def run_simulation(self, steps, step_length):
        for _ in range(steps):
            self.propagate_photons(step_length)
            self.interact_photons()


    def get_interactions(self):
        return self.interactions.to_numpy()

    def get_exits(self):
        return self.out.to_numpy()

    def init_microscope(self, observation_face, volume_size, voxel_size, NA, magnification, n_medium, sensor_size, pixel_size, wavelength, focal_depth):
        self.microscopes.append(DefocusMicroscope(
            observation_face=observation_face,
            volume_size=volume_size,
            voxel_size=voxel_size,
            NA = NA,
            magnification=magnification,
            n_medium=n_medium,
            sensor_size=sensor_size,
            pixel_size=pixel_size,
            wavelength=wavelength,
            focal_depth=focal_depth
        ))

    def to_numpy(self):
        self.positions_np = self.positions.to_numpy()
        self.directions_np = self.directions.to_numpy()
        self.intensities_np = self.intensities.to_numpy()
        self.entered_np = self.entered.to_numpy()
        self.exited_np = self.exited.to_numpy()
        self.last_scatter_pos_np = self.last_scatter_pos.to_numpy()
        self.bounces_np = self.bounces.to_numpy()
        if self.tracking:
            self.trajectory_positions_np = self.trajectory_positions.to_numpy()
            self.trajectory_valid_np = self.trajectory_valid.to_numpy()

    def tracking_lines(self):
        lines = []
        for i in range(self.n_tracked):
            valid_steps = np.where(self.trajectory_valid_np[i] == 1)[0]

            if len(valid_steps) > 1:
                path = self.trajectory_positions_np[i, valid_steps, :]

                for i in range(len(path) - 1):
                    p1 = path[i]
                    p2 = path[i+1]

                    p1_in = np.all((p1 >= 0) & (p1 <= self.N))
                    p2_in = np.all((p2 >= 0) & (p2 <= self.N))

                    if p1_in and p2_in:
                        line = np.array([p1, p2])
                        lines.append(line)

        return lines

    def defocus_image(self, i):
        return self.microscopes[i].form_image_with_defocus(
            self.positions_np, self.directions_np, self.intensities_np,
            self.entered_np, self.exited_np, self.last_scatter_pos_np, self.bounces_np
        )

