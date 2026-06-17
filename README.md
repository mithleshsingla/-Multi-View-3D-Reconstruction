# Multi-View 3D Reconstruction — Take-Home Assignment
**Nutpaa Technologies | Role: Computer Vision Engineer**

---

## Setup & Installation

```bash
# Create virtual environment
python3 -m venv multiview3d
source multiview3d/bin/activate        # Linux/macOS
# multiview3d\Scripts\activate         # Windows

# Install dependencies (no OpenCV required)
pip install numpy scipy matplotlib
```

### File Structure
```
Assignment_1/
├── data.json                   # Input data (3D corners, observations, K)
├── task1_pose_estimation.py    # Camera pose estimation  (DLT + LM)
├── task2_triangulation.py      # 3D triangulation        (DLT + optimal)
├── task3_robustness.py         # Robustness analysis     (standalone)
├── task4_profiling.py          # Actual runtime profiling
└── README.md
```

### Running Each Task
```bash
python task1_pose_estimation.py   # Estimates R, t for Cam2 and Cam3
python task2_triangulation.py     # Triangulates all 8 cube corners
python task3_robustness.py        # Runs 3A, 3B, 3C; saves 3 PNG plots
python task4_profiling.py         # Measures real runtimes; saves PNG
```

---

## Component Explanations & Mathematics

### Task 1 — Camera Pose Estimation

**Goal:** Given known 3D world points X and their noisy 2D observations x,
estimate the 6-DoF camera pose (R, t) for Camera 2 and Camera 3.

---

#### Step 1 — Hartley Normalisation

Before building the DLT system, pixel coordinates are normalised:
- Centroid shifted to origin
- RMS distance from origin scaled to √2

World points are similarly normalised (RMS → √3).

**Why it matters:** The DLT design matrix A mixes pixel values O(100–1000)
with the homogeneous coordinate 1.0. Without normalisation the condition
number of A is up to 10⁸× worse, making SVD numerically unstable and the
null-space solution inaccurate.

Normalisation transforms:
```
T_2d = | s   0  -s·cx |      T_3d = | s  0  0  -s·Cx |
       | 0   s  -s·cy |             | 0  s  0  -s·Cy |
       | 0   0    1   |             | 0  0  s  -s·Cz |
                                    | 0  0  0    1   |

where s = √2 / mean_distance   (2D)
      s = √3 / mean_distance   (3D)
```

---

#### Step 2 — Direct Linear Transform (DLT)

The projection equation in homogeneous form:
```
λ [u, v, 1]ᵀ = P [X, Y, Z, 1]ᵀ        P = K [R | t]   (3×4 matrix)
```

Cross-multiplying to eliminate the unknown scale λ gives two linear
equations per point (the cross-product constraint):
```
(u p³ᵀ − p¹ᵀ) X̃ = 0
(v p³ᵀ − p²ᵀ) X̃ = 0
```
where p¹, p², p³ are rows 0, 1, 2 of P and X̃ = [X,Y,Z,1]ᵀ.

Stacking N points gives the 2N×12 homogeneous system:
```
A p = 0     where  p = vec(P)  (12-vector)
```

**Confidence weighting:** Each row pair is multiplied by √wᵢ so that
higher-confidence observations contribute proportionally more to the
null-space solution.

**Solution via SVD null-space:**
```
A = U Σ Vᵀ   →   p = last column of V
```
The last column of V is the unit-norm vector minimising ‖Ap‖², i.e.
the null-space of A. After reshaping to 3×4 the result is de-normalised:
```
P_final = T_2d⁻¹ · P_normalised · T_3d
```

---

#### Step 3 — Decompose P → R, t

Given known K:
```
K⁻¹ P = [R | t]
```
The left 3×3 block M is not exactly orthogonal due to noise and DLT
algebra.  Polar decomposition forces a valid rotation:
```
M = U Σ Vᵀ   →   R = U Vᵀ
```
If det(R) = −1 (reflection), flip the sign of the last column of U.
The translation is recovered as t = t_raw / mean(Σ).

---

#### Step 4 — Levenberg–Marquardt Refinement

DLT minimises an algebraic error ‖Ap‖² that does not correspond to
pixel distances. LM minimises the **weighted geometric reprojection error**:
```
min_{R,t}  Σᵢ  wᵢ ‖proj(Xᵢ; K,R,t) − xᵢ‖²
```

R is parameterised as a **Rodrigues vector** rvec = θ·k̂ (3 parameters):
```
R = I·cosθ + (1−cosθ)·k̂k̂ᵀ + [k̂]×·sinθ
```
This keeps optimisation on the rotation manifold without re-orthogonalising
R after each step.

**Multi-start strategy:** A 1m cube has near-symmetric projections — DLT
can land in a mirrored local minimum.  We run LM from 9 initializations
(DLT result + 8 look-at-origin grid poses at varying azimuths and
distances) and keep the result with the lowest reprojection error.
This costs ~90ms but guarantees escape from cube-symmetry traps.

---

#### Task 1 Results

| Camera | Reproj error | Noise level | Status |
|--------|-------------|-------------|--------|
| Cam 1 (known pose, sanity check) | 2.30 px | σ = 2 px | ✓ matches noise |
| Cam 2 (estimated) | 1.39 px | σ = 2 px | ✓ below noise |
| Cam 3 (estimated) | 2.74 px | σ = 3 px | ✓ matches noise |

Camera centres (world coordinates):
- Cam 1: [0.00,  0.00, −5.00]  (known reference)
- Cam 2: [−2.10, 0.00, −4.57]  (2.1m to the left)
- Cam 3: [2.33, −0.05, −4.74]  (2.3m to the right)

---

### Task 2 — 3D Triangulation

**Goal:** With all 3 poses known, reconstruct each of the 8 cube corners
using every camera where the point is visible.

---

#### DLT Triangulation

For each camera i with P_i = K_i [R_i | t_i], the cross-product constraint
gives two rows per camera. Stacking M cameras:
```
A X̃ = 0     (2M × 4)
```
Solution: last column of V from SVD of A, then de-homogenise X = X̃[:3]/X̃[3].
Confidence weights applied as √w per row pair.

#### Optimal Triangulation

DLT minimises algebraic error; optimal minimises **geometric** error:
```
min_X  Σᵢ  wᵢ ‖proj(X; Kᵢ,Rᵢ,tᵢ) − xᵢ‖²
```
Solved with LM initialised from the DLT result.

#### Confidence Weighting Impact

Weighting reduced mean error by +0.04 mm (DLT) and +0.02 mm (optimal).
Modest gain because all confidences are 0.72–0.99.  With a genuinely
low-confidence outlier observation the gain would be much larger.

#### Error Analysis

| Point | Views | Error opt-w (mm) | Root cause |
|-------|-------|-----------------|------------|
| front-bottom-left  | 3 | 4.9  | Good parallax, 3 views |
| front-bottom-right | 3 | 12.3 | Moderate noise |
| back-bottom-right  | 2 | **35.7** | **22° ray angle, only 2 views** |
| back-top-right     | 3 | **25.4** | **22–46° angles** |
| back-top-left      | 2 | **24.1** | **23° ray angle, only 2 views** |

Large errors (24–36 mm) are **geometrically expected**:
- Back-face points are occluded in Camera 3 (the noisiest camera, σ=3px)
- Remaining camera pairs have low parallax angles (20–23°)
- Error ≈ noise / sin(θ) ≈ (2px × 6mm/px) / sin(22°) ≈ 16mm per camera

---

### Task 3 — Robustness Analysis

#### 3A — Noise Sensitivity

| σ (px) | Mean 3D error | ΔR Cam2 | ΔR Cam3 |
|--------|--------------|---------|---------|
| 1 | 21.8 ± 2.5 mm | 0.36° | 0.35° |
| 2 | 29.0 ± 3.5 mm | 0.71° | 0.73° |
| 5 | 60.8 ± 19 mm  | 2.38° | 1.75° |
| 10 | 118 ± 37 mm  | 4.57° | 4.08° |

Errors grow monotonically. The large std at σ=5,10 indicates occasional
convergence to wrong local minima — realistic behaviour at high noise.

#### 3B — Degenerate Geometry

Cam3 interpolated from original position toward Cam1 (coplanar, z=−5):

| Alpha | Tri angle | Cond(A) | Mean 3D error |
|-------|-----------|---------|---------------|
| 1.00 | 26.0° | 109,544 | 23.7 mm |
| 0.40 | 12.7° | 110,104 | 32.7 mm |
| 0.10 | 5.7°  | 110,363 | 62.5 mm |
| 0.05 | 4.6°  | 110,387 | **81.9 mm** |

Triangulation angle falling below 10° is the practical danger threshold.
3D error grows 3.5× as geometry degenerates.

#### 3C — Outlier Rejection

2 Camera 2 observations shifted +150px (simulating matching failure):

| Method | Reproj error | ΔR | Status |
|--------|-------------|-----|--------|
| Clean baseline | 1.39 px | 0.00° | reference |
| Vanilla LM | 56.00 px | 6.65° | ⚠ FAILED |
| RANSAC (4-pt, 500 iter) | 0.46 px | 0.50° | ✓ RECOVERED |
| Huber LM (warm-start) | 50.06 px | 0.56° | ✓ RECOVERED |
| Cauchy LM (warm-start) | 50.03 px | 0.50° | ✓ RECOVERED |

RANSAC must sample **4 points, not 6**: with 6 observations and 2 outliers,
any 6-point sample always includes both outliers (P=0).
Sampling 4 gives P(clean sample) = 1/15 ≈ 7%; over 500 iterations
P(at least one clean) ≈ 1.0.

---

## Profiling Output (Actual Measured on This Machine)

All times measured with `time.perf_counter`. Each micro-benchmark
discards one warm-up call then averages N independent runs.

### Per-Stage Timings

| Stage | Mean time | N runs |
|-------|-----------|--------|
| Data loading (json.load) | 0.024 ms | 100 |
| extract_obs | 0.003 ms | 1000 |
| normalise_3d / normalise_2d | 0.010 ms | 5000 |
| build_dlt_matrix | 0.026 ms | 2000 |
| np.linalg.svd (DLT core) | 0.016 ms | 2000 |
| dlt_solve (full, with normalisation) | 0.078 ms | 500 |
| decompose_P (polar decomp → R, t) | 0.013 ms | 2000 |
| **refine_lm (single LM run)** | **0.608 ms** | 50 |
| **estimate_pose (9 starts, Cam2)** | **36.9 ms** | 10 |
| **estimate_pose (9 starts, Cam3)** | **52.9 ms** | 10 |
| project (6 pts, 1 camera) | 0.003 ms | 10000 |
| reproj_err (6 pts, 1 camera) | 0.009 ms | 5000 |
| triangulate_dlt (3 views) | 0.021 ms | 500 |
| triangulate_optimal (3 views) | 0.339 ms | 100 |
| reconstruct_all (8 corners) | 2.645 ms | 20 |
| **run_pipeline (full)** | **95.2 ms** | 5 |

### Task-Level Timings

| Task | Actual wall-clock time |
|------|----------------------|
| Task 3A (4σ × 10 trials) | **4.316 s** |
| Task 3B (7 alpha steps) | **0.573 s** |
| Task 3C (all outlier methods) | **13.932 s** |

### Bottleneck Identified

**estimate_pose** (multi-start LM) accounts for:
- Cam2: 36.9 ms = **38.7%** of full pipeline
- Cam3: 52.9 ms = **55.5%** of full pipeline
- Combined: **94.2%** of total pipeline runtime

cProfile confirms: **79% of all CPU time** is inside
`scipy.optimize._numdiff.approx_derivative` — SciPy's LM is computing
the Jacobian **numerically** via finite differences, calling the
residual function ~23,000 times per profiled run.

### Proposed Optimisations

**Optimisation 1 — Analytic Jacobian (immediate, ~5–10× LM speedup)**

The Jacobian ∂residuals/∂[rvec, t] has a closed-form expression:
```
∂proj/∂rvec = (∂proj/∂Xc) · (∂Xc/∂rvec)
∂proj/∂t   = (∂proj/∂Xc) · I
```
Providing `jac=analytic_jacobian` to `least_squares()` eliminates all
23,000 finite-difference calls, reducing LM cost by 5–10×.

**Optimisation 2 — Replace multi-start DLT+LM with EPnP (161× speedup)**

EPnP (Lepetit et al., IJCV 2009) expresses world points as weighted
sums of 4 virtual control points, reduces pose estimation to a single
12×12 eigenvalue problem, and runs in O(n) time:

| Method | Time per camera | Task 3A total |
|--------|----------------|---------------|
| Current (9-start LM) | ~45 ms avg | 4.3 s |
| EPnP (estimated) | ~0.1 ms | ~0.03 s |
| **Speedup** | **~450×** | **~150×** |

EPnP is non-iterative and achieves accuracy comparable to iterative LM
for n ≥ 6 points, making it suitable for real-time applications.

---

## Known Limitations & Assumptions

1. **No lens distortion** — all cameras follow the ideal pinhole model.
   Real lenses have radial and tangential distortion; ignoring it
   introduces systematic bias at image corners.

2. **Known, perfect intrinsics K** — in practice K is estimated via
   checkerboard calibration, introducing its own uncertainty.

3. **Known 3D structure (PnP setting)** — the 8 cube corners are
   given. For unknown structure, full Structure-from-Motion (feature
   matching → essential matrix → bundle adjustment) would be needed.

4. **Static scene** — no motion blur, no temporal changes assumed.

5. **RANSAC sample size** — sampling 4 points for a 12-DoF problem is
   under-constrained. We compensate with multi-start LM per sample.
   A proper minimal-case solver (P3P, 3 points) would be cleaner.

6. **Cube symmetry** — the 1m cube has near-identical projections for
   mirrored camera poses. Multi-start LM resolves this but adds runtime.
   A non-symmetric calibration target would eliminate the problem.

7. **Task 3C Huber/Cauchy reprojection error is high (50px)** — the
   robust LM is warm-started from RANSAC's R, t which are correct, so
   rotation error is small (0.5°). The high reprojection error is computed
   on ALL 6 observations including the 2 outliers — expected behaviour,
   since the robust loss intentionally down-weights but does not exclude them.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥ 1.24 | Linear algebra (SVD, matrix ops) |
| scipy | ≥ 1.10 | Levenberg–Marquardt (least_squares) |
| matplotlib | ≥ 3.7 | Visualisation and plots |

No OpenCV is used. All algorithms are implemented from scratch.