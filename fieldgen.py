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

    @ti.kernel
    def _initSphere(self, map: ti.template(), N: ti.i32, cx: ti.f32, cy: ti.f32, cz: ti.f32, radii: ti.types.ndarray(), types: ti.types.ndarray()):
        for i, j, k in ti.ndrange(N, N, N):
            dist = (ti.Vector([float(i), float(j), float(k)]) - ti.Vector([cx, cy, cz])).norm()
            for c in range(radii.shape[0]):
                if dist < radii[c]:
                    map[i, j, k] = ti.cast(types[c], ti.u16)
                else:
                    break
