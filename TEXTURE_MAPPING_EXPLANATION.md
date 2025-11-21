# Texture Mapping Investigation Report

## Scene Created
**File:** `data/texture_demo_sphere.json`

This scene demonstrates texture mapping on spheres with:
- Left sphere: Image texture mapping using an EXR environment map
- Right sphere: Procedural checkerboard texture with 8x8 tiling
- Both spheres use diffuse materials with different textures

## Question 1: In which function does texture mapping occur when called from your path integrator?

### Answer:
Texture mapping occurs in the **`IdealDiffusion::evaluate()`** function in `src/bsdf.cpp`.

### Call Chain from Path Integrator:
1. **Path Integrator** (`src/path.cpp`) → Calls `bsdf->evaluate(interaction)`
2. **IdealDiffusion::evaluate()** (`src/bsdf.cpp` line 37-42) → Calls `texture->evaluate(interaction)`
3. **ImageTexture::evaluate()** (`src/texture.cpp` line 48-52) → Performs the actual texture lookup

### Key Code:
```cpp
// In IdealDiffusion::evaluate() - src/bsdf.cpp:37
Vec3f IdealDiffusion::evaluate(SurfaceInteraction &interaction) const {
  const Vec3f normal = obtainOrientedNormal(interaction, twosided);
  if (Dot(interaction.wi, normal) < 0 || Dot(interaction.wo, normal) < 0)
    return {0, 0, 0};
  return texture->evaluate(interaction) * INV_PI;  // <- Texture mapping here!
}

// In ImageTexture::evaluate() - src/texture.cpp:48
Vec3f ImageTexture::evaluate(const SurfaceInteraction &interaction) const {
  Vec2f dstdx, dstdy;
  const auto &st = texmap->Map(interaction, dstdx, dstdy);
  return mipmap->LookUp(st, dstdx, dstdy);  // <- MIPMap lookup
}
```

## Question 2: How does the texture mapping support mipmap?

### Answer:
The texture mapping supports mipmap through a **multi-level image pyramid** with **trilinear interpolation** or **EWA (Elliptical Weighted Average) filtering**.

### Implementation Details:

#### 1. MIPMap Construction (`include/rdr/mipmap.h` and initialization in `src/texture.cpp:30-34`):
```cpp
// When ImageTexture is loaded:
mipmap = make_ref<MIPMap>(Vec2u(width, height), data);
```

The MIPMap class pre-computes a pyramid of progressively lower-resolution versions of the texture.

#### 2. Differential Computation:
The key to mipmap is computing how much the texture coordinates change across screen pixels:

```cpp
// In UVMapping2D::Map() - src/texture.cpp:12-14
Vec2f UVMapping2D::Map(const SurfaceInteraction &interaction, 
                       Vec2f &dstdx, Vec2f &dstdy) const {
  dstdx = scale * Vec2f(interaction.dudx, interaction.dvdx);  // ∂s/∂x, ∂t/∂x
  dstdy = scale * Vec2f(interaction.dudy, interaction.dvdy);  // ∂s/∂y, ∂t/∂y
  return scale * interaction.uv + delta;
}
```

#### 3. Level Selection and Lookup (`mipmap->LookUp()` in `include/rdr/mipmap.h:32-33`):
```cpp
Vec3f LookUp(const Vec2f &st, Vec2f dstdx, Vec2f dstdy) const noexcept;
```

The MIPMap uses `dstdx` and `dstdy` to:
- **Calculate the mipmap level** based on the rate of change of texture coordinates
- **Prevent aliasing** by selecting an appropriate pre-filtered level
- **Interpolate** between adjacent mipmap levels for smooth transitions

#### 4. Two Lookup Methods:
- **TriLinearInterpolation**: Bilinear interpolation at two adjacent mipmap levels, then linear blend
- **EWA (Elliptical Weighted Average)**: More sophisticated anisotropic filtering

## Question 3: UV Coordinates for Sphere and Why We Need Differentials

### Part A: How UV Coordinates are Computed for a Sphere

In `src/shape.cpp` (Sphere::intersect, lines 70-80), the sphere uses **spherical coordinates**:

```cpp
// After computing intersection point delta_p relative to sphere center:
constexpr InternalScalarType phiMax   = 2 * PI;
constexpr InternalScalarType thetaMax = PI;

InternalScalarType phi = std::atan2(delta_p.y, delta_p.x);
if (phi < 0) phi += 2 * PI;

InternalScalarType theta = std::acos(delta_p.z / radius);

InternalScalarType u = phi / phiMax;    // u = φ/(2π) ∈ [0,1]
InternalScalarType v = theta / thetaMax; // v = θ/π ∈ [0,1]
```

### Computed UV Coordinates:
- **u = φ/(2π)**: Azimuthal angle normalized to [0,1], wraps around equator
- **v = θ/π**: Polar angle normalized to [0,1], from north pole (v=0) to south pole (v=1)

### Part B: Why Do We Need to Compute Differentials?

The differentials (dudx, dvdx, dudy, dvdy) are **essential for anti-aliasing** and proper texture filtering.

#### The Problem Without Differentials:
When a textured surface is viewed:
- **Far away**: Many texture pixels (texels) map to a single screen pixel → **aliasing/moiré patterns**
- **Close up**: Single texel maps to many screen pixels → **magnification**

#### How Differentials Solve This:

1. **Measuring Screen-Space Rate of Change** (`src/interaction.cpp:29-76`):
```cpp
void SurfaceInteraction::CalculateRayDifferentials(const DifferentialRay &ray) {
  // Compute where adjacent screen-space rays intersect the surface
  auto px = ray.dx_origin + tx * ray.dx_direction;  // x+1 pixel ray intersection
  auto py = ray.dy_origin + ty * ray.dy_direction;  // y+1 pixel ray intersection
  
  internal.dpdx = px - p;  // ∂p/∂x (change in surface position per screen x)
  internal.dpdy = py - p;  // ∂p/∂y (change in surface position per screen y)
  
  // Solve linear system to get texture coordinate differentials:
  // dpdx = dudx * dpdu + dvdx * dpdv
  // dpdy = dudy * dpdu + dvdy * dpdv
  SolveLinearSystem2x2(A, Bx, &internal.dudx, &internal.dvdx);
  SolveLinearSystem2x2(A, By, &internal.dudy, &internal.dvdy);
}
```

2. **For Sphere** (`src/shape.cpp:93-119`), we compute:
```cpp
// Parametric derivatives of sphere surface
InternalVecType dpdu = phiMax * InternalVecType(-delta_p.y, delta_p.x, 0);
InternalVecType dpdv = thetaMax * InternalVecType(delta_p.z * cos_phi, 
                                                   delta_p.z * sin_phi,
                                                   -radius * std::sin(theta));
```

These represent:
- **dpdu** (∂p/∂u): How the surface position changes as u increases
- **dpdv** (∂p/∂v): How the surface position changes as v increases

3. **MIPMap Level Selection**:
```cpp
// In mipmap->LookUp(st, dstdx, dstdy)
// dstdx = (∂s/∂x, ∂t/∂x) tells us texture footprint in x direction
// dstdy = (∂s/∂y, ∂t/∂y) tells us texture footprint in y direction

// Calculate appropriate mipmap level:
Float delta_max² = max(|dstdx|², |dstdy|²)
Float level = 0.5 * log2(delta_max²)
```

#### Why This Matters:
- **Without differentials**: All texture lookups use level 0 (full resolution) → severe aliasing
- **With differentials**: 
  - **Grazing angles/far distances**: High differentials → higher mipmap levels (pre-filtered)
  - **Direct views/close up**: Low differentials → lower mipmap levels (detailed)
  - **Result**: Smooth, alias-free texture mapping

### Summary:
Differentials allow the renderer to:
1. **Measure the texture footprint** of a screen pixel
2. **Select the appropriate mipmap level** automatically
3. **Prevent aliasing** through proper pre-filtering
4. **Maintain performance** by avoiding supersampling

## Rendering Command

To render the demo scene:
```bash
cd /home/azuma/Downloads/cs171-hw3-2025fall-tzhazuma
./build/bin/rdr data/texture_demo_sphere.json
```

The output will be saved as `output.exr` which you can view with an EXR viewer.

## Code File References

1. **Texture evaluation entry point**: `src/bsdf.cpp` (IdealDiffusion::evaluate, line 37)
2. **Texture coordinate mapping**: `src/texture.cpp` (UVMapping2D::Map, line 11)
3. **Image texture lookup**: `src/texture.cpp` (ImageTexture::evaluate, line 48)
4. **MIPMap implementation**: `include/rdr/mipmap.h` and `src/mipmap.cpp`
5. **Sphere UV computation**: `src/shape.cpp` (Sphere::intersect, lines 70-119)
6. **Differential computation**: `src/interaction.cpp` (CalculateRayDifferentials, lines 10-76)
