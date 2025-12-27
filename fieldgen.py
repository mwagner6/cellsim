import taichi as ti 
import numpy as np

@ti.data_oriented
class VolumeGenerator:
    def __init__(self):
        self.typemap = [
            {
                "p_absorb": 0,
                "p_scatter": 0,
                "scatter_map": [1]*360,
                "r_index": 1
            }
        ]

    def initType(self, p_absorb, p_scatter, scatter_map, r_index):
        self.typemap.append({"p_absorb": p_absorb, "p_scatter": p_scatter, "scatter_map": scatter_map, "r_index": r_index})

    def resolveTypes(self, scatter_prec):
        self.types_np = np.ndarray((len(self.typemap), 3), dtype=np.float32)

        self.scatters_np = np.ndarray((len(self.typemap), scatter_prec), dtype=np.uint16)
        for i, type in enumerate(self.typemap):
            self.types_np[i, 0] = type["p_absorb"]
            self.types_np[i, 1] = type["p_scatter"]
            self.types_np[i, 2] = type["r_index"]

            scatter_dist = np.random.choice(range(360), size=(scatter_prec), p=np.array(type["scatter_map"])/sum(type["scatter_map"]))
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
