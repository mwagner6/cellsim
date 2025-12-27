import numpy as np


class DefocusMicroscope:
    """
    Virtual microscope with depth-dependent defocus blur.
    Each photon is rendered as a Gaussian whose width depends on the distance
    from its last scatter position to the focal plane.
    """
    
    def __init__(self, 
                 observation_face: str = '+z',
                 volume_size: int = 128,
                 voxel_size: float = 1.0,  # microns
                 NA: float = 0.65,
                 magnification: float = 20.0,
                 n_medium: float = 1.33,
                 sensor_size: tuple = (512, 512),
                 pixel_size: float = 6.5,  # microns
                 wavelength: float = 550.0,  # nm
                 focal_depth: float = 64.0):  # voxel coordinates
        """
        Initialize microscope with defocus capability.
        
        Args:
            focal_depth: Depth of focal plane in voxel coordinates (e.g., 64 for center)
        """
        self.observation_face = observation_face
        self.volume_size = volume_size
        self.voxel_size = voxel_size
        self.NA = NA
        self.magnification = magnification
        self.n_medium = n_medium
        self.sensor_size = sensor_size
        self.pixel_size = pixel_size
        self.wavelength = wavelength
        self.focal_depth = focal_depth
        
        # Calculate acceptance angle
        self.acceptance_angle = np.arcsin(NA / n_medium)
        
        # Map face to normal direction and coordinate system
        self.face_config = self._setup_face_geometry()
        
        # Field of view in object space (microns)
        #self.fov_x = sensor_size[0] * pixel_size / magnification
        #self.fov_y = sensor_size[1] * pixel_size / magnification
        self.fov_x = sensor_size[0]
        self.fov_y = sensor_size[1]

        # Physical volume dimensions
        self.volume_extent = volume_size * voxel_size  # microns
        
        # PSF parameters - minimum blur (in-focus)
        self.airy_radius = 0.61 * (wavelength / 1000) / NA  # microns
        self.min_sigma_pixels = (self.airy_radius * magnification / pixel_size) / 2.355
        
        # Depth of field (microns)
        self.dof = (wavelength / 1000) / (2 * NA**2)  # microns
        self.dof_voxels = self.dof / voxel_size  # convert to voxel units
        
        print(f"Taichi Microscope with Defocus initialized:")
        print(f"  Observation face: {observation_face}")
        print(f"  Focal depth: {focal_depth} voxels ({focal_depth * voxel_size:.1f} μm)")
        print(f"  Depth of field: {self.dof:.2f} μm ({self.dof_voxels:.2f} voxels)")
        print(f"  NA: {NA}, Acceptance angle: {np.degrees(self.acceptance_angle):.1f}°")
        print(f"  Min sigma (in-focus): {self.min_sigma_pixels:.2f} pixels")
    
    def _setup_face_geometry(self):
        """Set up coordinate system for each face."""
        configs = {
            '+z': {
                'normal': np.array([0, 0, 1]),
                'face_coord': 2,  # z-axis
                'face_value': self.volume_size,
                'tangent_coords': [0, 1],  # x, y
                'depth_coord': 2,  # z is depth
            },
            '-z': {
                'normal': np.array([0, 0, -1]),
                'face_coord': 2,
                'face_value': 0,
                'tangent_coords': [0, 1],
                'depth_coord': 2,
            },
            '+x': {
                'normal': np.array([1, 0, 0]),
                'face_coord': 0,
                'face_value': self.volume_size,
                'tangent_coords': [1, 2],  # y, z
                'depth_coord': 0,  # x is depth
            },
            '-x': {
                'normal': np.array([-1, 0, 0]),
                'face_coord': 0,
                'face_value': 0,
                'tangent_coords': [1, 2],
                'depth_coord': 0,
            },
            '+y': {
                'normal': np.array([0, 1, 0]),
                'face_coord': 1,
                'face_value': self.volume_size,
                'tangent_coords': [0, 2],  # x, z
                'depth_coord': 1,  # y is depth
            },
            '-y': {
                'normal': np.array([0, -1, 0]),
                'face_coord': 1,
                'face_value': 0,
                'tangent_coords': [0, 2],
                'depth_coord': 1,
            },
        }
        return configs[self.observation_face]
    
    def calculate_defocus_sigma(self, scatter_depths, exit_depths, bounces):
        """
        Calculate defocus blur (sigma) based on distance from focal plane.

        Args:
            scatter_depths: Depth coordinate of last scatter position (voxels)
            exit_depths: Depth coordinate of exit position (voxels)
            bounces: Number of scattering events for each photon

        Returns:
            sigma in pixels for each photon
        """
        # For photons that never scattered (bounces == 0), use minimum sigma (in-focus)
        # For scattered photons, calculate defocus based on scatter position

        # Distance from focal plane (in voxels)
        defocus_distance = np.abs(scatter_depths - self.focal_depth)

        # Geometric blur increases linearly with defocus distance
        # Circle of confusion radius = defocus * tan(acceptance_angle)
        coc_radius_voxels = defocus_distance * np.tan(self.acceptance_angle)
        coc_radius_microns = coc_radius_voxels * self.voxel_size
        coc_radius_pixels = coc_radius_microns * self.magnification / self.pixel_size

        # Total blur is combination of diffraction (min_sigma) and defocus
        # Use quadrature sum (sqrt of sum of squares)
        total_sigma = np.sqrt(self.min_sigma_pixels**2 + (coc_radius_pixels / 2.355)**2)

        # For unscattered photons (bounces == 0), force minimum sigma
        total_sigma = np.where(bounces == 0, self.min_sigma_pixels, total_sigma)

        return total_sigma
    
    def filter_photons_by_face(self, positions_np, intensities_np, entered_np, exited_np):
        """Filter photons that exited through the observation face."""
        config = self.face_config
        face_coord = config['face_coord']
        face_value = config['face_value']
        
        has_exited = exited_np == 1
        tolerance = 1.5
        
        if face_value == 0:
            at_face = positions_np[:, face_coord] < tolerance
        else:
            at_face = positions_np[:, face_coord] > (face_value - tolerance)
        
        return has_exited & at_face
    
    def filter_by_na(self, directions_np, mask):
        """Filter photons by numerical aperture."""
        valid_dirs = directions_np[mask]
        if len(valid_dirs) == 0:
            return mask
        
        normal = self.face_config['normal']
        cos_angles = np.sum(valid_dirs * normal, axis=1)
        angles = np.arccos(np.clip(cos_angles, -1, 1))
        within_na = angles <= self.acceptance_angle
        
        result_mask = mask.copy()
        result_mask[mask] = within_na
        return result_mask
    
    def form_image_with_defocus(self, positions_np, directions_np, intensities_np,
                                 entered_np, exited_np, last_scatter_pos_np, bounces_np):
        """
        Form microscope image with depth-dependent defocus blur.
        Each photon is rendered as a Gaussian splat.

        Args:
            last_scatter_pos_np: Position where each photon last scattered (n_photons, 3)
            bounces_np: Number of scattering events for each photon (n_photons,)
        """
        # Step 1: Filter by face
        face_mask = self.filter_photons_by_face(
            positions_np, intensities_np, entered_np, exited_np
        )

        # Step 2: Filter by NA
        na_mask = self.filter_by_na(directions_np, face_mask)

        n_accepted = na_mask.sum()
        if n_accepted == 0:
            print(f"Warning: No photons accepted for face {self.observation_face}")
            return np.zeros(self.sensor_size)

        # Step 3: Get data for accepted photons
        config = self.face_config
        tangent_coords = config['tangent_coords']
        depth_coord = config['depth_coord']

        # Exit positions in microns
        exit_pos = positions_np[na_mask] * self.voxel_size
        scatter_pos = last_scatter_pos_np[na_mask]
        weights = intensities_np[na_mask]
        bounces = bounces_np[na_mask]
        print(min(bounces))

        # Project to sensor coordinates
        x_obj = exit_pos[:, tangent_coords[0]]
        y_obj = exit_pos[:, tangent_coords[1]]

        #sensor_x = x_obj * self.magnification / self.pixel_size + self.sensor_size[0] / 2
        #sensor_y = y_obj * self.magnification / self.pixel_size + self.sensor_size[1] / 2

        sensor_x = x_obj
        sensor_y = y_obj

        # Get scatter depths and exit depths
        scatter_depths = scatter_pos[:, depth_coord]
        exit_depths = positions_np[na_mask, depth_coord]

        # Calculate per-photon sigma (accounting for unscattered photons via bounces)
        sigmas = self.calculate_defocus_sigma(scatter_depths, exit_depths, bounces)

        # Render each photon as a Gaussian
        image = self._render_gaussian_photons(sensor_x, sensor_y, weights, sigmas)

        # Count unscattered photons
        n_unscattered = (bounces == 0).sum()

        print(f"Face {self.observation_face}: {face_mask.sum()} photons at face, "
              f"{n_accepted} within NA")
        print(f"  Unscattered photons: {n_unscattered} ({100*n_unscattered/n_accepted:.1f}%)")
        print(f"  Sigma range: [{sigmas.min():.2f}, {sigmas.max():.2f}] pixels")
        print(f"  Peak intensity: {image.max():.2e}")

        return image.T
    
    def _render_gaussian_photons(self, sensor_x, sensor_y, weights, sigmas):
        """
        Render photons as Gaussian splats.
        
        This is the key function that implements depth-dependent blur!
        """
        image = np.zeros(self.sensor_size, dtype=np.float32)
        
        # For efficiency, group photons by similar sigma
        sigma_bins = np.linspace(sigmas.min(), sigmas.max(), 50)
        sigma_indices = np.digitize(sigmas, sigma_bins)
        
        for bin_idx in range(len(sigma_bins) + 1):
            mask = sigma_indices == bin_idx
            if not mask.any():
                continue
            
            bin_sigma = sigmas[mask].mean()
            bin_x = sensor_x[mask]
            bin_y = sensor_y[mask]
            bin_w = weights[mask]
            
            # Create Gaussian kernel for this bin
            kernel_radius = int(np.ceil(3 * bin_sigma))
            if kernel_radius < 1:
                kernel_radius = 1
            
            kernel_size = 2 * kernel_radius + 1
            y_k, x_k = np.ogrid[-kernel_radius:kernel_radius+1, -kernel_radius:kernel_radius+1]
            kernel = np.exp(-(x_k**2 + y_k**2) / (2 * bin_sigma**2))
            kernel /= kernel.sum()
            
            # Splat photons onto image
            for i in range(len(bin_x)):
                ix = int(np.round(bin_x[i]))
                iy = int(np.round(bin_y[i]))
                
                # Skip if outside sensor
                if ix < kernel_radius or ix >= self.sensor_size[0] - kernel_radius:
                    continue
                if iy < kernel_radius or iy >= self.sensor_size[1] - kernel_radius:
                    continue
                
                # Add Gaussian splat
                x_start = ix - kernel_radius
                x_end = ix + kernel_radius + 1
                y_start = iy - kernel_radius
                y_end = iy + kernel_radius + 1
                
                image[x_start:x_end, y_start:y_end] += bin_w[i] * kernel
        
        return image.T