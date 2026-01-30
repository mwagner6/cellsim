# Simulation.py Summary

## Overview

[simulation.py](simulation.py) implements a GPU-accelerated Monte Carlo photon transport simulation using the Taichi framework. It models light propagation through 3D volumes containing biological structures (organoids, cells) with different material properties, simulating absorption, scattering, and imaging through virtual microscopes.

## Architecture

### Component Overview

```mermaid
graph TB
    subgraph Simulation Class
        SIM[Simulation]
        VOL[(volume<br/>3D voxel grid)]
        INT[(interactions<br/>absorption map)]
        OUT[(out<br/>exit intensity)]
        TYPES[(types_ti<br/>material properties)]
        SCATTER[(scatters_ti<br/>angle distributions)]
    end

    subgraph External Components
        BG[EllipticalBeamGenerator]
        VG[VolumeGenerator]
        DM[DefocusMicroscope]
    end

    SIM --> VOL
    SIM --> INT
    SIM --> OUT
    SIM --> TYPES
    SIM --> SCATTER

    VG -->|generates| VOL
    VG -->|populates| TYPES
    VG -->|populates| SCATTER
    BG -->|creates photons| SIM
    SIM -->|sends photons| DM
```

### Main Class: `Simulation`

The `@ti.data_oriented` decorated class manages the entire simulation pipeline, from volume generation to photon tracking to microscope image formation.

**Key Initialization Parameters** ([simulation.py:13](simulation.py#L13)):
- `N` - Grid resolution (default 128³ voxels)
- `wavelength` - Light wavelength in μm (default 0.65)
- `n_photons` - Total photons to simulate
- `batch_size` - Photons per batch to manage VRAM

**Core Data Structures**:
- `volume` - 3D grid storing material type IDs (uint16)
- `interactions` - Accumulated absorption energy per voxel
- `out` - Exit intensity map
- `types_ti` - Material properties table (absorption, scattering, refraction)
- `scatters_ti` - Scattering angle distributions per material

## Workflow

```mermaid
flowchart LR
    subgraph Setup
        A[init_types] --> B[init geometry]
        B --> C[init_beam_generator]
        C --> D[init_microscope]
    end

    subgraph Run
        D --> E[simulation_loop]
    end

    subgraph Output
        E --> F[defocus_image]
        E --> G[get_interactions]
        E --> H[tracking_lines]
    end
```

### 1. Material Type Initialization

**`init_types(types, scatter_prec)`** ([simulation.py:35](simulation.py#L35)):
- Initializes the `VolumeGenerator` with material definitions
- Each material has: relative scale, density, pigment, and refractive index
- Creates scattering lookup tables with specified precision

### 2. Geometry Definition

Three methods for creating 3D structures:

```mermaid
graph TD
    subgraph Geometry Methods
        S[initSpheres]
        E[initEllipsoids]
        O[organoidGen]
    end

    subgraph Organoid Layers
        direction TB
        L1[fringe<br/>outermost]
        L2[main<br/>cytoplasm]
        L3[necrotic<br/>dead cells]
        L4[lumen<br/>empty center]
    end

    O --> L1
    L1 --> L2
    L2 --> L3
    L3 --> L4

    S --> VOL[(3D Volume)]
    E --> VOL
    O --> VOL
```

**`initSpheres(spheres)`** ([simulation.py:47](simulation.py#L47)):
- Creates multi-layered spherical structures

**`initEllipsoids(ellipsoids)`** ([simulation.py:83](simulation.py#L83)):
- Creates arbitrarily oriented ellipsoids with full 3D rotation control
- Supports custom axis orientations via `axis1_dir` and `axis2_dir` vectors
- Enables multi-layer structures (e.g., nucleus inside cytoplasm)

**`organoidGen(...)`** ([simulation.py:52](simulation.py#L52)):
- Convenience method for biological organoid structures
- Generates concentric ellipsoid layers: fringe → main → necrotic → lumen
- Pre-configured with typical biological material properties

### 3. Beam Configuration

**`init_beam_generator(...)`** ([simulation.py:169](simulation.py#L169)):
- Sets up an `EllipticalBeamGenerator` for illumination
- Controls beam geometry (position, angle, divergence)
- Supports elliptical beam profiles with Gaussian distributions

### 4. Photon Simulation Loop

**`simulation_loop(max_steps, step_length, track, n_tracked)`** ([simulation.py:196](simulation.py#L196)):

```mermaid
flowchart TB
    START([Start]) --> RESET[Reset microscope images]
    RESET --> CHECK{photons_left > 0?}

    CHECK -->|Yes| BATCH[Calculate batch size<br/>min&#40;photons_left, batchsize&#41;]
    BATCH --> INIT[init_photons<br/>Generate positions & directions]
    INIT --> TRACK{First batch &<br/>tracking enabled?}

    TRACK -->|Yes| INITTRACK[init_tracking]
    TRACK -->|No| RUN
    INITTRACK --> RUN

    RUN[run_simulation<br/>propagate + interact loop]
    RUN --> NUMPY[to_numpy<br/>Transfer to CPU]
    NUMPY --> MICRO[Add photons to<br/>all microscopes]
    MICRO --> CHECK

    CHECK -->|No| END([End])
```

The main simulation pipeline runs in batches:

1. **Batch Management**: Processes photons in chunks to avoid VRAM overflow
2. **Photon Generation** ([simulation.py:231](simulation.py#L231)): Creates photon arrays with positions/directions from beam generator
3. **Propagation & Interaction Loop** ([simulation.py:410](simulation.py#L410)):
   - **Propagate**: Move photons along their direction vectors ([simulation.py:276-314](simulation.py#L276-L314))
   - **Interact**: Handle absorption/scattering in the volume ([simulation.py:317-407](simulation.py#L317-L407))
4. **Microscope Accumulation**: Add photons to virtual detector images

### 5. Photon Interactions

**`_interact_photons(...)`** ([simulation.py:318](simulation.py#L318)) - Taichi kernel:

```mermaid
stateDiagram-v2
    [*] --> InVolume: Photon enters volume

    InVolume --> CheckRandom: At voxel (x,y,z)

    CheckRandom --> Absorbed: rand < absorption_prob
    CheckRandom --> Scattered: rand < absorption + scatter_prob
    CheckRandom --> Continue: Otherwise

    Absorbed --> [*]: intensity = 0<br/>Add to interactions grid

    Scattered --> SampleAngle: Get deflection from scattermap
    SampleAngle --> RotateDir: Apply random azimuthal rotation
    RotateDir --> UpdateBounce: bounces++, save position
    UpdateBounce --> Propagate

    Continue --> Propagate: Move by step_length

    Propagate --> OutOfBounds: Position outside volume
    Propagate --> InVolume: Still in volume

    OutOfBounds --> Exited: Mark exited=1<br/>Record in out grid
    Exited --> [*]
```

For each photon at position (x,y,z):
1. **Absorption Check**: Random value < absorption probability → photon absorbed, intensity added to `interactions[x,y,z]`
2. **Scattering Event**: Sample deflection angle from material's scattering distribution
   - Construct orthonormal basis around current direction
   - Apply deflection angle with random azimuthal rotation
   - Update direction vector and bounce counter
3. **Exit Detection**: Track when photons leave the volume and record exit positions in `out` array

### 6. Microscope Integration

**`init_microscope(...)`** ([simulation.py:422](simulation.py#L422)):
- Creates `DefocusMicroscope` instances observing different faces
- Configures optical parameters (NA, magnification, sensor properties)

**`defocus_image(i)`** ([simulation.py:469](simulation.py#L469)):
- Retrieves accumulated image from microscope `i` after simulation completes

### 7. Optional Trajectory Tracking

**`init_tracking(n_tracked, steps)`** ([simulation.py:266](simulation.py#L266)):
- Records positions of first `n_tracked` photons at each step
- Enables visualization with `tracking_lines()` ([simulation.py:448](simulation.py#L448))

## Key Technical Details

- **GPU Acceleration**: All compute-intensive operations use Taichi kernels for parallel execution
- **Memory Management**: Uses `ti.ndarray` instead of fields for photon data to enable batch processing
- **Scattering Physics**: Implements realistic angle-dependent scattering via pre-computed lookup tables
- **Multi-microscope**: Supports simultaneous observation from multiple viewpoints

## Data Flow

```mermaid
flowchart TB
    subgraph Input
        MAT[Material Definitions<br/>rel_scale, rel_density,<br/>rel_pigment, r_index]
        BEAM[Beam Parameters<br/>position, angle,<br/>divergence, profile]
    end

    subgraph Volume Setup
        VG[VolumeGenerator]
        VOL[(3D Volume<br/>N × N × N voxels)]
        TYPES[(Type Properties<br/>absorption, scatter,<br/>refraction)]
    end

    subgraph Photon Simulation
        GEN[Generate Photons<br/>position + direction]
        PROP[Propagate<br/>pos += dir × step]
        INTER[Interact<br/>absorb / scatter]
    end

    subgraph Output
        MICRO[Microscope Images]
        ABSORB[Absorption Map]
        TRACKS[Trajectory Data]
    end

    MAT --> VG
    VG --> VOL
    VG --> TYPES

    BEAM --> GEN
    GEN --> PROP
    PROP <--> INTER
    VOL -.->|material lookup| INTER
    TYPES -.->|properties| INTER

    INTER -->|exited photons| MICRO
    INTER -->|absorbed energy| ABSORB
    PROP -->|positions| TRACKS
```
