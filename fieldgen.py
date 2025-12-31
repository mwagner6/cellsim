import taichi as ti 
import numpy as np

@ti.data_oriented
class VolumeGenerator:
    def __init__(self, wavelength):
        self.typemap = [
            {
                "rel_density": 0,
                "rel_scale": 1,
                "rel_pigment": 0,
                "r_index": 1.3333,
            }
        ]
        self.wavelength = wavelength
        self.A_MELANIN = 6.49E7
        self.B_MELANIN = 3.48

        self.A_SCAT_1UM = 0.005
        self.B_SCAT = 1.5

        self.A_ANISO_1UM = 0.90
        self.B_ANISO = 0.05

    def initType(self, rel_scale, rel_density, rel_pigment, r_index):
        self.typemap.append({"rel_density": rel_density, "rel_scale": rel_scale, "rel_pigment": rel_pigment, "r_index": r_index})

    def resolveTypes(self, scatter_prec):
        scatter_angles = 2 * np.pi * np.linspace(0, 180, scatter_prec) / 360
        self.types_np = np.ndarray((len(self.typemap), 3), dtype=np.float32)

        self.scatters_np = np.ndarray((len(self.typemap), scatter_prec), dtype=np.uint16)
        for i, type in enumerate(self.typemap):
            self.types_np[i, 0] = type["rel_pigment"] * self.A_MELANIN * self.wavelength **(-self.B_MELANIN)
            self.types_np[i, 1] = type["rel_density"] * self.A_SCAT_1UM * self.wavelength **(-self.B_SCAT)
            self.types_np[i, 2] = type["r_index"]

            aniso_g = self.A_ANISO_1UM * self.wavelength **(self.B_ANISO/type["rel_scale"])
            cos_scat = np.cos(scatter_angles)

            num = 1 - aniso_g**2
            denom = 4 * np.pi * (1 + aniso_g**2 - 2 * aniso_g * cos_scat)**1.5

            scat_probs = np.log(num / denom)
            scat_probs += np.amin(scat_probs)

            scatter_dist = np.random.choice(scatter_angles, size=(scatter_prec), p=(scat_probs / np.sum(scat_probs)))
            self.scatters_np[i, :] = scatter_dist

        self.types_ti = ti.ndarray(dtype=ti.f32, shape=(len(self.typemap), 3))
        self.types_ti.from_numpy(self.types_np)


    def initSpheres(self, spheres, map, typemap, scattermap, N):
        for sphere in spheres:
            self._initSphere(map, int(N), sphere["center"][0], sphere["center"][1], sphere["center"][2], np.array(sphere["radii"], dtype=np.float32), np.array(sphere["types"], dtype=np.uint16))
        typemap.from_numpy(self.types_np)
        scattermap.from_numpy(self.scatters_np)

    def initEllipsoids(self, ellipsoids, map, typemap, scattermap, N):
        """
        Initialize ellipsoids in the volume with arbitrary orientations.

        Args:
            ellipsoids: List of ellipsoid dicts with keys:
                - "center": [x, y, z] center position
                - "axes": [[ax1, ay1, az1], [ax2, ay2, az2], ...] semi-axis lengths for each layer
                - "types": [type1, type2, ...] material type for each layer (innermost to outermost)
                - "axis1_dir": [x, y, z] direction vector for first principal axis (optional, default=[1,0,0])
                - "axis2_dir": [x, y, z] direction vector for second principal axis (optional, default=[0,1,0])
                  Note: axis2_dir will be orthogonalized to axis1_dir, and axis3 will be computed as cross product
            map: Volume field to write to
            typemap: Type properties field
            scattermap: Scattering map field
            N: Volume size
        """
        for ellipsoid in ellipsoids:
            axes_array = np.array(ellipsoid["axes"], dtype=np.float32)  # Shape: (n_layers, 3)
            types_array = np.array(ellipsoid["types"], dtype=np.uint16)

            # Get orientation vectors (default to axis-aligned)
            axis1_dir = np.array(ellipsoid.get("axis1_dir", [1.0, 0.0, 0.0]), dtype=np.float32)
            axis2_dir = np.array(ellipsoid.get("axis2_dir", [0.0, 1.0, 0.0]), dtype=np.float32)

            # Normalize axis1
            axis1_dir = axis1_dir / np.linalg.norm(axis1_dir)

            # Orthogonalize axis2 to axis1 (Gram-Schmidt)
            axis2_dir = axis2_dir - np.dot(axis2_dir, axis1_dir) * axis1_dir
            axis2_dir = axis2_dir / np.linalg.norm(axis2_dir)

            # Compute axis3 as cross product (right-handed coordinate system)
            axis3_dir = np.cross(axis1_dir, axis2_dir)
            axis3_dir = axis3_dir / np.linalg.norm(axis3_dir)

            # Build rotation matrix (columns are the axis directions)
            # This transforms from ellipsoid space to world space
            rotation_matrix = np.column_stack([axis1_dir, axis2_dir, axis3_dir]).astype(np.float32)

            self._initEllipsoid(
                map, int(N),
                ellipsoid["center"][0], ellipsoid["center"][1], ellipsoid["center"][2],
                axes_array, types_array, rotation_matrix
            )
        typemap.from_numpy(self.types_np)
        scattermap.from_numpy(self.scatters_np)

    @ti.kernel
    def _initSphere(self, map: ti.template(), N: ti.i32, cx: ti.f32, cy: ti.f32, cz: ti.f32, radii: ti.types.ndarray(), types: ti.types.ndarray()):
        for i, j, k in ti.ndrange(N, N, N):
            dist = (ti.Vector([float(i), float(j), float(k)]) - ti.Vector([cx, cy, cz])).norm()
            for c in range(radii.shape[0]):
                if dist < radii[c]:
                    map[i, j, k] = ti.cast(types[c], ti.u16)
                else:
                    break

    @ti.kernel
    def _initEllipsoid(self, map: ti.template(), N: ti.i32, cx: ti.f32, cy: ti.f32, cz: ti.f32,
                       axes: ti.types.ndarray(), types: ti.types.ndarray(),
                       rotation_matrix: ti.types.ndarray()):
        """
        Initialize an oriented ellipsoid with potentially multiple layers.

        Args:
            axes: ndarray of shape (n_layers, 3) containing semi-axis lengths [a, b, c] for each layer
            types: ndarray of shape (n_layers,) containing material type indices
            rotation_matrix: ndarray of shape (3, 3) rotation matrix transforming world coords to ellipsoid coords
        """
        for i, j, k in ti.ndrange(N, N, N):
            # Position relative to center (world coordinates)
            dx = float(i) - cx
            dy = float(j) - cy
            dz = float(k) - cz

            # Transform to ellipsoid coordinate system using inverse rotation
            # Since rotation matrices are orthogonal, inverse = transpose
            dx_local = rotation_matrix[0, 0] * dx + rotation_matrix[1, 0] * dy + rotation_matrix[2, 0] * dz
            dy_local = rotation_matrix[0, 1] * dx + rotation_matrix[1, 1] * dy + rotation_matrix[2, 1] * dz
            dz_local = rotation_matrix[0, 2] * dx + rotation_matrix[1, 2] * dy + rotation_matrix[2, 2] * dz

            # Check each layer from innermost to outermost
            for layer in range(axes.shape[0]):
                # Get semi-axes for this layer
                a = axes[layer, 0]  # semi-axis along local x (axis1_dir)
                b = axes[layer, 1]  # semi-axis along local y (axis2_dir)
                c = axes[layer, 2]  # semi-axis along local z (axis3_dir = axis1 × axis2)

                # Ellipsoid equation in local coordinates: (x/a)^2 + (y/b)^2 + (z/c)^2 < 1
                ellipsoid_val = (dx_local / a) ** 2 + (dy_local / b) ** 2 + (dz_local / c) ** 2

                if ellipsoid_val < 1.0:
                    map[i, j, k] = ti.cast(types[layer], ti.u16)
                else:
                    # If outside this layer, no need to check outer layers
                    break
