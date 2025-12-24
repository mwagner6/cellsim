import taichi as ti
import numpy as np
from typing import Tuple

@ti.data_oriented
class EllipticalBeamGenerator:
    """
    Generate photons from an elliptical source with Gaussian distribution.
    The photons are all parallel to the beam axis (from source center to target point).
    Uses Taichi for efficient parallel computation.
    """
    
    def __init__(self, 
                 central_point: Tuple[float, float, float],
                 distance: float,
                 theta: float,  # polar angle in radians
                 phi: float,    # azimuthal angle in radians
                 mask_diameter: float,  # diameter to which the beam is masked
                 sigma_a: float,    # Gaussian beam width parameter
                 sigma_b: float,    # Gaussian beam width parameter
                 divergence_a: float,
                 divergence_b: float,
                 divergence_sigma_a: float,
                 divergence_sigma_b: float):
        """
        Initialize the elliptical beam generator.
        
        Parameters:
        -----------
        central_point : tuple
            Target point (x0, y0, z0)
        distance : float
            Distance from central point to source center
        theta : float
            Polar angle (from z-axis) in radians
        phi : float
            Azimuthal angle (from x-axis in xy-plane) in radians
        mask_diameter: float
            Limit photon generation to within this diameter (typically set by illuminated volume)
        sigma_a: float
            Semi-major axis beam size
        sigma_b: float
            Semi-minor axis beam size
        divergence_a: float
            Semi-major axis beam divergence (radians) at 1 sigma distance from axis
        divergence_b: float
            Semi-minor axis beam divergence (radians) at 1 sigma distance from axis
        divergence_sigma_a: float
            Semi-major axis beam divergence spread (radians)
        divergence_sigma_b: float
            Semi-minor axis beam divergence spread (radians)
        """
        self.central_point = np.array(central_point)
        self.distance = distance
        self.theta = theta
        self.phi = phi
        self.mask_diameter = mask_diameter
        self.sigma_a = sigma_a
        self.sigma_b = sigma_b
        self.divergence_a = divergence_a
        self.divergence_b = -divergence_b  # it is unclear why this needs to be reversed
        self.divergence_sigma_a = divergence_sigma_a
        self.divergence_sigma_b = divergence_sigma_b
        
        # Calculate source center position using spherical coordinates
        self.source_center = self._calculate_source_center()
        
        # Calculate beam direction (from source to central point)
        self.beam_direction = self._calculate_beam_direction()

        # Calculate focal distances if convergent
        self._calculate_focal_parameters()
        
        # Create local coordinate system for the ellipse
        self.local_x, self.local_y, self.local_z = self._create_local_coords()
        
        # Create Taichi fields to store parameters (scalar fields)
        self.ti_source_center = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.ti_local_x = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.ti_local_y = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.ti_local_z = ti.Vector.field(3, dtype=ti.f32, shape=())
        
        # Copy to Taichi fields
        self.ti_source_center[None] = ti.Vector(self.source_center.tolist())
        self.ti_local_x[None] = ti.Vector(self.local_x.tolist())
        self.ti_local_y[None] = ti.Vector(self.local_y.tolist())
        self.ti_local_z[None] = ti.Vector(self.local_z.tolist())
    
    def _calculate_source_center(self) -> np.ndarray:
        """Calculate the position of the source center in global coordinates."""
        # Convert spherical to Cartesian (source is at distance d from central point)
        dx = self.distance * np.sin(self.theta) * np.cos(self.phi)
        dy = self.distance * np.sin(self.theta) * np.sin(self.phi)
        dz = self.distance * np.cos(self.theta)
        
        # Source is positioned away from central point
        source_center = self.central_point + np.array([dx, dy, dz])
        return source_center
    
    def _calculate_beam_direction(self) -> np.ndarray:
        """Calculate normalized beam direction (from source to central point)."""
        direction = self.central_point - self.source_center
        return direction / np.linalg.norm(direction)
    
    def _calculate_focal_parameters(self):
        """
        Calculate focal points and distances for convergent beams.
        
        For convergent beams, the focal distance is where rays from 1 sigma 
        would converge if extrapolated.
        """
        # For major axis
        if abs(self.divergence_a) > 0 and self.sigma_a > 0:
            # At 1 sigma, the ray has angle divergence_a
            # tan(angle) = sigma / focal_distance
            self.focal_distance_a = - self.sigma_a / np.tan(self.divergence_a)
            self.focal_point_a = self.source_center + self.focal_distance_a * self.beam_direction
        else:
            self.focal_distance_a = np.inf
            self.focal_point_a = None
        
        # For minor axis
        if abs(self.divergence_b) > 0 and self.sigma_b > 0:  # Convergent
            self.focal_distance_b = - self.sigma_b / np.tan(self.divergence_b)
            self.focal_point_b = self.source_center + self.focal_distance_b * self.beam_direction
        else:
            self.focal_distance_b = np.inf
            self.focal_point_b = None

    def _create_local_coords(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a local coordinate system for the ellipse plane.
        local_z points along the beam direction (inward)
        local_x and local_y span the ellipse plane
        """
        local_z = self.beam_direction
        
        # Choose an arbitrary vector not parallel to local_z
        if abs(local_z[2]) < 0.9:
            arbitrary = np.array([0, 0, 1])
        else:
            arbitrary = np.array([1, 0, 0])
        
        # Create orthonormal basis using Gram-Schmidt
        local_x = arbitrary - np.dot(arbitrary, local_z) * local_z
        local_x = local_x / np.linalg.norm(local_x)
        
        local_y = np.cross(local_z, local_x)
        
        return local_x, local_y, local_z

    @ti.func
    def _ti_calculate_photon_direction(self, u_coord: ti.f32, v_coord: ti.f32) -> ti.math.vec3:
        """
        Calculate the direction vector for a photon based on its position.
        
        The divergence angle is proportional to the distance from the center,
        scaled so that photons at 1 sigma have the specified divergence angle.
        
        Parameters:
        -----------
        u_coord : float
            Local coordinate along major axis
        v_coord : float
            Local coordinate along minor axis
        
        Returns:
        --------
        direction : vec3
            Unit direction vector for the photon
        """
        # Calculate position-dependent divergence angles
        position_divergence_a = 0.0
        position_divergence_b = 0.0
        
        if self.sigma_a > 0:
            position_divergence_a = (u_coord / self.sigma_a) * self.divergence_a
        
        if self.sigma_b > 0:
            position_divergence_b = (v_coord / self.sigma_b) * self.divergence_b
        
        # Add random divergence component
        random_divergence_a = ti.randn() * self.divergence_sigma_a
        random_divergence_b = ti.randn() * self.divergence_sigma_b
        
        total_angle_a = position_divergence_a + random_divergence_a
        total_angle_b = position_divergence_b + random_divergence_b
        
        # Start with beam direction
        local_z = self.ti_local_z[None]
        local_x = self.ti_local_x[None]
        local_y = self.ti_local_y[None]
        
        direction = local_z
        
        # Rotation around local_y axis (affects x-z plane)
        if total_angle_a != 0.0:
            cos_a = ti.cos(total_angle_a)
            sin_a = ti.sin(total_angle_a)
            # Rodrigues' rotation formula
            direction = cos_a * direction + sin_a * ti.math.cross(local_y, direction)
        
        # Rotation around local_x axis (affects y-z plane)
        if total_angle_b != 0.0:
            cos_b = ti.cos(total_angle_b)
            sin_b = ti.sin(total_angle_b)
            direction = cos_b * direction + sin_b * ti.math.cross(local_x, direction)
        
        direction = ti.math.normalize(direction)
        
        return direction

    @ti.kernel
    def _generate_and_filter_samples(self, 
                                     positions: ti.types.ndarray(),
                                     directions: ti.types.ndarray(),
                                     rotation_angle: ti.f32) -> ti.i32:
        """
        Generate photon positions and directions using Taichi kernel.
        Each thread generates one valid photon by retrying until successful.
        Returns the number of samples generated (should equal n_photons).
        """
        n_photons = positions.shape[0]
        source_center = self.ti_source_center[None]
        local_x = self.ti_local_x[None]
        local_y = self.ti_local_y[None]
        
        cos_r = ti.cos(rotation_angle)
        sin_r = ti.sin(rotation_angle)
        mask_radius_sq = (self.mask_diameter / 2.0) ** 2
        
        # Each thread generates one valid sample
        for i in range(n_photons):
            # Keep trying until we get a valid sample
            valid = False
            u = 0.0
            v = 0.0
            
            # Try up to 1000 times to get a valid sample
            for attempt in range(1000):
                if not valid:
                    # Generate 2D Gaussian samples
                    u = ti.randn() * self.sigma_a
                    v = ti.randn() * self.sigma_b
                    
                    # Apply rotation in the ellipse plane
                    if rotation_angle != 0.0:
                        u_rot = cos_r * u - sin_r * v
                        v_rot = sin_r * u + cos_r * v
                        u = u_rot
                        v = v_rot
                    
                    # Check if within mask
                    if u * u + v * v < mask_radius_sq:
                        valid = True
            
            # Position in global coordinates
            pos = source_center + u * local_x + v * local_y
            
            # Direction with position-dependent divergence
            direction = self._ti_calculate_photon_direction(u, v)
            
            # Store result
            positions[i, 0] = pos[0]
            positions[i, 1] = pos[1]
            positions[i, 2] = pos[2]
            
            directions[i, 0] = direction[0]
            directions[i, 1] = direction[1]
            directions[i, 2] = direction[2]
        
        return n_photons

    def generate_photons(self, n_photons: int,
                        rotation_angle: float = 0.0,
                        truncate_at: float = 3.0) -> Tuple[ti.types.ndarray(), ti.types.ndarray()]:
        """
        Generate n photons with Gaussian distribution in the ellipse.
        Returns Taichi arrays for efficient GPU/parallel computation.

        Parameters:
        -----------
        n_photons : int
            Number of photons to generate
        rotation_angle : float
            Rotation angle of the ellipse in its plane (radians)
        truncate_at : float
            Truncate Gaussian at this many sigmas (to keep points within ellipse)

        Returns:
        --------
        positions : Taichi ndarray of shape (n_photons, 3)
            3D positions of photons
        directions : Taichi ndarray of shape (n_photons, 3)
            Direction vectors (all parallel to beam axis)
        """
        # Allocate Taichi arrays
        positions = ti.ndarray(dtype=ti.f32, shape=(n_photons, 3))
        directions = ti.ndarray(dtype=ti.f32, shape=(n_photons, 3))

        # Generate samples
        count = self._generate_and_filter_samples(
            positions,
            directions,
            float(rotation_angle)
        )

        return positions.to_numpy(), directions.to_numpy()

    @ti.kernel
    def _generate_to_fields(self,
                           positions_field: ti.template(),
                           directions_field: ti.template(),
                           rotation_angle: ti.f32):
        """
        Generate photons directly into Taichi fields (no CPU transfer).
        This is more efficient than generate_photons() when working with fields.
        """
        n_photons = positions_field.shape[0]
        source_center = self.ti_source_center[None]
        local_x = self.ti_local_x[None]
        local_y = self.ti_local_y[None]

        cos_r = ti.cos(rotation_angle)
        sin_r = ti.sin(rotation_angle)
        mask_radius_sq = (self.mask_diameter / 2.0) ** 2

        # Each thread generates one valid sample
        for i in range(n_photons):
            # Keep trying until we get a valid sample
            valid = False
            u = 0.0
            v = 0.0

            # Try up to 1000 times to get a valid sample
            for attempt in range(1000):
                if not valid:
                    # Generate 2D Gaussian samples
                    u = ti.randn() * self.sigma_a
                    v = ti.randn() * self.sigma_b

                    # Apply rotation in the ellipse plane
                    if rotation_angle != 0.0:
                        u_rot = cos_r * u - sin_r * v
                        v_rot = sin_r * u + cos_r * v
                        u = u_rot
                        v = v_rot

                    # Check if within mask
                    if u * u + v * v < mask_radius_sq:
                        valid = True

            # Position in global coordinates
            pos = source_center + u * local_x + v * local_y

            # Direction with position-dependent divergence
            direction = self._ti_calculate_photon_direction(u, v)

            # Store directly into fields
            positions_field[i, 0] = pos[0]
            positions_field[i, 1] = pos[1]
            positions_field[i, 2] = pos[2]

            directions_field[i, 0] = direction[0]
            directions_field[i, 1] = direction[1]
            directions_field[i, 2] = direction[2]

    def generate_photons_to_fields(self,
                                   positions_field: ti.template(),
                                   directions_field: ti.template(),
                                   rotation_angle: float = 0.0):
        """
        Generate photons directly into existing Taichi fields.
        This avoids CPU-GPU transfers and is more efficient.

        Parameters:
        -----------
        positions_field : Taichi field
            Field of shape (n_photons, 3) to write positions to
        directions_field : Taichi field
            Field of shape (n_photons, 3) to write directions to
        rotation_angle : float
            Rotation angle of the ellipse in its plane (radians)
        """
        self._generate_to_fields(positions_field, directions_field, float(rotation_angle))