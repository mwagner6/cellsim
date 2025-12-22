"""
Virtual Microscope Module
Post-processes photon exit data to simulate microscope imaging
"""

import numpy as np
from scipy import ndimage
from typing import Tuple, Optional, Dict
import matplotlib.pyplot as plt


class VirtualMicroscope:
    """
    Virtual microscope for post-processing photon exit data
    Simulates objective lens collection, PSF, and image formation
    """
    
    def __init__(self, 
                 NA: float = 0.65,
                 magnification: float = 20.0,
                 n_medium: float = 1.33,
                 sensor_size: Tuple[int, int] = (512, 512),
                 pixel_size: float = 6.5e-3,  # mm (6.5 μm typical)
                 tube_lens_focal_length: float = 200.0):  # mm
        """
        Initialize virtual microscope
        
        Args:
            NA: Numerical aperture of objective
            magnification: Objective magnification
            n_medium: Refractive index of immersion medium
            sensor_size: Detector array size in pixels
            pixel_size: Physical pixel size in mm
            tube_lens_focal_length: Tube lens focal length in mm
        """
        self.NA = NA
        self.magnification = magnification
        self.n_medium = n_medium
        self.sensor_size = sensor_size
        self.pixel_size = pixel_size
        self.tube_lens_focal_length = tube_lens_focal_length
        
        # Calculate derived parameters
        self.acceptance_angle = np.arcsin(NA / n_medium)
        self.objective_focal_length = tube_lens_focal_length / magnification
        
        # Field of view in object space (mm)
        self.fov_x = sensor_size[0] * pixel_size / magnification
        self.fov_y = sensor_size[1] * pixel_size / magnification
        
        # Depth of field (mm)
        wavelength_nm = 550  # Reference wavelength
        self.dof = wavelength_nm * 1e-6 / (2 * NA**2)
        
        print(f"Virtual Microscope initialized:")
        print(f"  NA: {NA}")
        print(f"  Magnification: {magnification}x")
        print(f"  Acceptance angle: {np.degrees(self.acceptance_angle):.1f}°")
        print(f"  Field of view: {self.fov_x:.3f} x {self.fov_y:.3f} mm")
        print(f"  Depth of field: {self.dof*1000:.1f} μm")
        print(f"  Sensor: {sensor_size[0]}x{sensor_size[1]} pixels")
    
    def filter_by_na(self, photons: np.ndarray, 
                     observation_direction: str = '+z') -> np.ndarray:
        """
        Filter photons by numerical aperture acceptance cone
        
        Args:
            photons: Structured array of exit photons
            observation_direction: Direction of observation (+z, -z, +x, -x, +y, -y)
        
        Returns:
            Boolean mask of accepted photons
        """
        # Map observation direction to vector
        dir_map = {
            '+z': np.array([0, 0, 1]),
            '-z': np.array([0, 0, -1]),
            '+x': np.array([1, 0, 0]),
            '-x': np.array([-1, 0, 0]),
            '+y': np.array([0, 1, 0]),
            '-y': np.array([0, -1, 0])
        }
        
        obs_dir = dir_map.get(observation_direction, np.array([0, 0, 1]))
        
        # Calculate angle between photon direction and optical axis
        photon_dirs = np.column_stack([
            photons['dir_x'],
            photons['dir_y'],
            photons['dir_z']
        ])
        
        # Dot product with observation direction
        cos_angles = np.sum(photon_dirs * obs_dir, axis=1)
        angles = np.arccos(np.clip(cos_angles, -1, 1))
        
        # Accept photons within NA cone
        accepted = angles <= self.acceptance_angle
        
        return accepted
    
    def project_to_sensor(self, photons: np.ndarray,
                         z_focus: float = 0.5,
                         observation_direction: str = '+z') -> Tuple[np.ndarray, np.ndarray]:
        """
        Project photons to sensor plane through objective lens
        
        Args:
            photons: Exit photons (after NA filtering)
            z_focus: Focus position in object space (mm)
            observation_direction: Direction of observation
        
        Returns:
            (sensor_x, sensor_y) coordinates in pixels
        """
        # Transform coordinates based on observation direction
        if observation_direction == '+z':
            obj_x = photons['x']
            obj_y = photons['y']
            obj_z = photons['z']
            dir_x = photons['dir_x']
            dir_y = photons['dir_y']
            dir_z = photons['dir_z']
        elif observation_direction == '-z':
            obj_x = photons['x']
            obj_y = photons['y']
            obj_z = -photons['z']
            dir_x = photons['dir_x']
            dir_y = photons['dir_y']
            dir_z = -photons['dir_z']
        # Add other directions as needed
        
        # Ray trace to focus plane
        if np.abs(dir_z).min() < 1e-6:
            # Handle near-parallel rays
            t = np.zeros_like(dir_z)
        else:
            t = (z_focus - obj_z) / dir_z
        
        # Position at focus plane
        x_focus = obj_x + t * dir_x
        y_focus = obj_y + t * dir_y
        
        # Apply magnification and convert to sensor coordinates
        sensor_x = x_focus * self.magnification / self.pixel_size
        sensor_y = y_focus * self.magnification / self.pixel_size
        
        # Center on sensor
        sensor_x += self.sensor_size[0] / 2
        sensor_y += self.sensor_size[1] / 2
        
        return sensor_x, sensor_y
    
    def calculate_psf(self, wavelength: float = 550.0) -> np.ndarray:
        """
        Calculate point spread function (Airy disk approximation)
        
        Args:
            wavelength: Wavelength in nm
        
        Returns:
            2D PSF kernel
        """
        # Airy disk radius in object space
        airy_radius = 0.61 * wavelength * 1e-6 / self.NA  # mm
        
        # Convert to pixels on sensor
        airy_radius_pixels = airy_radius * self.magnification / self.pixel_size
        
        # Create PSF kernel (simplified Gaussian approximation)
        kernel_size = int(6 * airy_radius_pixels) | 1  # Ensure odd
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        
        y, x = np.ogrid[-center:kernel_size-center, -center:kernel_size-center]
        r = np.sqrt(x*x + y*y)
        
        # Gaussian approximation to Airy pattern
        sigma = airy_radius_pixels / 2.355  # FWHM to sigma
        kernel = np.exp(-(r**2) / (2 * sigma**2))
        
        # Normalize
        kernel /= kernel.sum()
        
        return kernel
    
    def form_image(self, photons: np.ndarray,
                  z_focus: float = 0.5,
                  observation_direction: str = '+z',
                  apply_psf: bool = True,
                  noise_model: Optional[str] = None) -> np.ndarray:
        """
        Form microscope image from exit photons
        
        Args:
            photons: Structured array of exit photons
            z_focus: Focus position in mm
            observation_direction: Observation direction
            apply_psf: Whether to convolve with PSF
            noise_model: 'poisson', 'gaussian', or None
        
        Returns:
            2D image array
        """
        # Filter by NA
        accepted = self.filter_by_na(photons, observation_direction)
        accepted_photons = photons[accepted]
        
        if len(accepted_photons) == 0:
            return np.zeros(self.sensor_size)
        
        # Project to sensor
        sensor_x, sensor_y = self.project_to_sensor(
            accepted_photons, z_focus, observation_direction
        )
        
        # Bin photons to image
        image, _, _ = np.histogram2d(
            sensor_x, sensor_y,
            bins=[np.arange(self.sensor_size[0] + 1),
                  np.arange(self.sensor_size[1] + 1)],
            weights=accepted_photons['weight']
        )
        
        # Apply PSF if requested
        if apply_psf:
            # Get average wavelength
            avg_wavelength = np.average(
                accepted_photons['wavelength'],
                weights=accepted_photons['weight']
            )
            psf = self.calculate_psf(avg_wavelength)
            image = ndimage.convolve(image, psf, mode='constant')
        
        # Apply noise model if requested
        if noise_model == 'poisson':
            # Scale to photon counts and add Poisson noise
            photons_per_count = 100  # Arbitrary scaling
            counts = image * photons_per_count
            image = np.random.poisson(counts) / photons_per_count
        elif noise_model == 'gaussian':
            # Add Gaussian noise
            noise_level = 0.01 * image.max()
            image += np.random.normal(0, noise_level, image.shape)
        
        # Ensure non-negative
        image = np.maximum(image, 0)
        
        return image.T  # Transpose for correct orientation
    
    def generate_focus_stack(self, photons: np.ndarray,
                           z_positions: np.ndarray,
                           observation_direction: str = '+z') -> np.ndarray:
        """
        Generate a focus stack (Z-stack) of images
        
        Args:
            photons: Exit photon data
            z_positions: Array of focus positions in mm
            observation_direction: Observation direction
        
        Returns:
            3D array [z, y, x] of images
        """
        stack = np.zeros((len(z_positions), *self.sensor_size))
        
        for i, z_focus in enumerate(z_positions):
            stack[i] = self.form_image(
                photons, z_focus, observation_direction
            )
        
        return stack
    
    def calculate_depth_response(self, photons: np.ndarray,
                                z_range: Tuple[float, float] = (0.3, 0.7),
                                n_points: int = 50) -> Dict:
        """
        Calculate depth response curve (intensity vs focus position)
        
        Args:
            photons: Exit photon data
            z_range: Range of focus positions to test
            n_points: Number of focus positions
        
        Returns:
            Dictionary with z_positions and intensities
        """
        z_positions = np.linspace(z_range[0], z_range[1], n_points)
        intensities = []
        
        for z in z_positions:
            image = self.form_image(photons, z_focus=z, apply_psf=False)
            intensities.append(image.sum())
        
        return {
            'z_positions': z_positions,
            'intensities': np.array(intensities),
            'peak_z': z_positions[np.argmax(intensities)]
        }
    
    def visualize_image(self, image: np.ndarray, 
                       title: str = "Microscope Image",
                       colormap: str = 'gray',
                       save_path: Optional[str] = None):
        """
        Visualize microscope image
        
        Args:
            image: 2D image array
            title: Plot title
            colormap: Matplotlib colormap
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Calculate physical dimensions
        extent = [0, self.fov_x * 1000,  # Convert to μm
                  0, self.fov_y * 1000]
        
        im = ax.imshow(image, cmap=colormap, extent=extent, origin='lower')
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_title(title)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Intensity (a.u.)')
        
        # Add scale bar
        scalebar_length = 100  # μm
        scalebar_x = extent[1] * 0.8
        scalebar_y = extent[3] * 0.1
        ax.plot([scalebar_x - scalebar_length, scalebar_x],
                [scalebar_y, scalebar_y], 'w-', linewidth=3)
        ax.text(scalebar_x - scalebar_length/2, scalebar_y + extent[3]*0.02,
                f'{scalebar_length} μm', color='white', ha='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def analyze_exit_distribution(self, photons: np.ndarray) -> Dict:
        """
        Analyze statistical properties of exit photons
        
        Args:
            photons: Exit photon data
        
        Returns:
            Dictionary of statistics
        """
        # Angular distribution
        angles = np.arccos(photons['dir_z'])
        
        # Spatial distribution
        x_spread = photons['x'].std()
        y_spread = photons['y'].std()
        
        # Path length statistics
        path_mean = photons['path_length'].mean()
        path_std = photons['path_length'].std()
        
        # Scattering statistics
        scatter_mean = photons['num_scatters'].mean()
        scatter_max = photons['num_scatters'].max()
        
        return {
            'n_photons': len(photons),
            'mean_exit_angle': np.degrees(angles.mean()),
            'std_exit_angle': np.degrees(angles.std()),
            'spatial_spread_x': x_spread * 1000,  # μm
            'spatial_spread_y': y_spread * 1000,
            'mean_path_length': path_mean,
            'std_path_length': path_std,
            'mean_scatters': scatter_mean,
            'max_scatters': scatter_max,
            'total_weight': photons['weight'].sum(),
            'mean_weight': photons['weight'].mean()
        }


class MultiViewMicroscope:
    """
    Extension for multi-angle viewing (6+ directions)
    """
    
    def __init__(self, base_microscope: VirtualMicroscope):
        """
        Initialize multi-view microscope
        
        Args:
            base_microscope: Base VirtualMicroscope instance
        """
        self.microscope = base_microscope
        self.views = ['+z', '-z', '+x', '-x', '+y', '-y']
    
    def generate_all_views(self, photons: np.ndarray,
                          z_focus: float = 0.5) -> Dict[str, np.ndarray]:
        """
        Generate images from all 6 cardinal directions
        
        Args:
            photons: Exit photon data
            z_focus: Focus position
        
        Returns:
            Dictionary mapping view names to images
        """
        images = {}
        
        for view in self.views:
            images[view] = self.microscope.form_image(
                photons, z_focus, observation_direction=view
            )
        
        return images
    
    def create_overview_figure(self, images: Dict[str, np.ndarray],
                              title: str = "Multi-View Microscopy"):
        """
        Create figure showing all views
        
        Args:
            images: Dictionary of view images
            title: Figure title
        
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Layout: 2 rows, 3 columns
        positions = {
            '+z': (1, 2),  # Top view
            '-z': (2, 2),  # Bottom view
            '+x': (1, 3),  # Right view
            '-x': (1, 1),  # Left view
            '+y': (2, 1),  # Front view
            '-y': (2, 3)   # Back view
        }
        
        for view, (row, col) in positions.items():
            if view in images:
                ax = plt.subplot(2, 3, (row-1)*3 + col)
                ax.imshow(images[view], cmap='gray')
                ax.set_title(f'{view} view')
                ax.axis('off')
        
        plt.tight_layout()
        return fig


def test_virtual_microscope():
    """Test virtual microscope with simulated data"""
    print("\n" + "="*60)
    print("VIRTUAL MICROSCOPE TEST")
    print("="*60)
    
    # Create synthetic exit photon data
    n_photons = 10000
    
    # Create structured array
    dtype = np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('dir_x', np.float32), ('dir_y', np.float32), ('dir_z', np.float32),
        ('wavelength', np.float32), ('weight', np.float32),
        ('stokes_I', np.float32), ('stokes_Q', np.float32),
        ('stokes_U', np.float32), ('stokes_V', np.float32),
        ('path_length', np.float32), ('num_scatters', np.int32)
    ])
    
    photons = np.zeros(n_photons, dtype=dtype)
    
    # Simulate photons exiting from a point source with angular spread
    center_x, center_y = 0.5, 0.5  # mm
    
    for i in range(n_photons):
        # Random exit angle (within NA cone)
        theta = np.random.uniform(0, np.pi/6)  # Up to 30 degrees
        phi = np.random.uniform(0, 2*np.pi)
        
        # Position (slight spread around center)
        photons[i]['x'] = center_x + np.random.normal(0, 0.05)
        photons[i]['y'] = center_y + np.random.normal(0, 0.05)
        photons[i]['z'] = 1.0  # Exit at top surface
        
        # Direction
        photons[i]['dir_x'] = np.sin(theta) * np.cos(phi)
        photons[i]['dir_y'] = np.sin(theta) * np.sin(phi)
        photons[i]['dir_z'] = np.cos(theta)
        
        # Properties
        photons[i]['wavelength'] = 550.0
        photons[i]['weight'] = np.exp(-theta*2)  # Weight decreases with angle
        photons[i]['stokes_I'] = 1.0
        photons[i]['path_length'] = np.random.gamma(2, 0.5)
        photons[i]['num_scatters'] = np.random.poisson(5)
    
    # Create microscope
    microscope = VirtualMicroscope(NA=0.65, magnification=20)
    
    # Form image
    print("\nForming image...")
    image = microscope.form_image(photons, z_focus=0.5)
    
    # Analyze exit distribution
    stats = microscope.analyze_exit_distribution(photons)
    print("\nExit photon statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Test focus stack
    print("\nGenerating focus stack...")
    z_positions = np.linspace(0.3, 0.7, 5)
    stack = microscope.generate_focus_stack(photons, z_positions)
    print(f"  Stack shape: {stack.shape}")
    
    return microscope, image


if __name__ == "__main__":
    # Run test
    microscope, image = test_virtual_microscope()
    
    # Try to visualize (will work if matplotlib display is available)
    try:
        fig = microscope.visualize_image(image)
        plt.show()
    except:
        print("\nVisualization skipped (no display available)")
    
    print("\n" + "="*60)
    print("VIRTUAL MICROSCOPE TEST COMPLETE")
    print("="*60)
