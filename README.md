# Multi-View 3D Reconstruction — Take-Home Assignment
**Nutpaa Technologies | Role: Computer Vision Engineer**

---

## Setup & Installation

```bash
# Create virtual environment
python3 -m venv multiview3d
source multiview3d/bin/activate        # Linux/macOS
# multiview3d\Scripts\activate         # Windows

# Install dependencies
pip install numpy scipy matplotlib

# All tasks are standalone — no OpenCV required
```

### File Structure
```
Assignment_1/
├── data.json                   # Input data (3D points, observations, K)
├── task1_pose_estimation.py    # Camera pose estimation (DLT + LM)
├── task2_triangulation.py      # 3D triangulation (DLT + optimal)
├── task3_robustness.py         # Robustness analysis (standalone)
└── README.md
```

### Running Each Task
```bash
python task1_pose_estimation.py   # Estimates R, t for Cam2 and Cam3
python task2_triangulation.py     # Triangulates all 8 cube corners
python task3_robustness.py        # Runs 3A, 3B, 3C; saves 3 PNGs
```

---

## Component Explanations & Mathematics

### Task 1 — Camera Pose Estimation

**Goal:** Given known 3D world points X and their noisy 2D observations x,
estimate the 6-DoF camera pose (R, t) for Camera 2 and Camera 3.

#### Step 1 — Hartley Normalisation
Before building the DLT system, pixel coordinates are normalised so that
the centroid is at the origin and the RMS distance from origin is √2.
World points are similarly normalised (RMS distance → √3).

**Why:** The DLT design matrix A has entries mixing pixel values O(100–1000)
with the homogeneous coordinate 1.0. Without normalisation, SVD is
numerically unstable and the condition number of A can be 10⁸×
worse than necessary.

#### Step 2 — Direct Linear Transform (DLT)
The projection equation is:

```
λ [u, v, 1]ᵀ = P [X, Y, Z, 1]ᵀ        P = K [R | t]  (3×4)
```

Cross-multiplying to eliminate the unknown scale λ gives two linear
equations per point:

```
(u p³ᵀ − p¹ᵀ) X̃ = 0
(v p³ᵀ − p²ᵀ) X̃ = 0
```

Stacking N points gives the 2N×12 system **A p = 0** where p = vec(P).

**Solution via SVD null-space:**
```
A = U Σ Vᵀ   →   p = last column of V
```
The last column of V is the unit-norm vector minimising ‖Ap‖²,
i.e. the null-space of A. Confidence weights are applied as √w per row
so higher-quality observations contribute more to the null-space solution.

#### Step 3 — Decompose P → R, t
Given known K:
```
K⁻¹ P = [R | t]
```
The left 3×3 block M is forced to a valid rotation matrix via polar
decomposition (SVD: M = UΣVᵀ → R = UVᵀ, det forced to +1).

#### Step 4 — Levenberg–Marquardt Refinement
DLT minimises an algebraic error that does not correspond to pixel
distances. LM minimises the weighted geometric (reprojection) error:

```
min_{R,t}  Σᵢ  wᵢ ‖proj(Xᵢ; K,R,t) − xᵢ‖²
```

R is parameterised as a Rodrigues vector rvec = θ·k̂ (3 parameters)
to optimise directly on the rotation manifold without re-orthogonalising
after each step.

**Multi-start strategy:** Because a 1m cube has near-symmetric projections,
DLT can land in a mirrored local minimum. We run LM from 9 look-at-origin
starting poses (DLT + 8 spherical grid starts) and keep the best result.
This adds ~0.15s but guarantees escape from cube-symmetry traps.

#### Results
| Camera | Reproj error | Notes |
|--------|-------------|-------|
| Cam 1 (known) | 2.30 px | ≈ σ = 2 px noise ✓ |
| Cam 2 (estimated) | 1.39 px | < σ = 2 px ✓ |
| Cam 3 (estimated) | 2.74 px | ≈ σ = 3 px ✓ |

---

### Task 2 — 3D Triangulation

**Goal:** With all 3 poses known, reconstruct each of the 8 cube corners
using all cameras where the point is visible.

#### DLT Triangulation
For each camera i with projection matrix Pᵢ = Kᵢ[Rᵢ|tᵢ]:
```
λᵢ [u, v, 1]ᵀ = Pᵢ X̃
```
Cross-multiplying gives two rows per camera. Stacking M cameras:
```
A X̃ = 0   (2M × 4)
```
Solution: last column of V from SVD of A, then de-homogenise.
Confidence weights applied as √w per row.

#### Optimal Triangulation
DLT minimises algebraic error; optimal minimises geometric error:
```
min_X  Σᵢ  wᵢ ‖proj(X; Kᵢ,Rᵢ,tᵢ) − xᵢ‖²
```
Solved via LM initialised at the DLT result. This reduces mean error
by ~0.15 mm across all points.

#### Confidence Weighting Impact
Confidence weights reduce mean error by +0.04 mm (DLT) and +0.02 mm
(optimal). Modest gain because all confidences are 0.72–0.99 (high).
With a low-confidence outlier observation the gain would be substantial.

#### Error Analysis
| Point | #Views | Error (opt-w, mm) | Root cause |
|-------|--------|-------------------|------------|
| front-bottom-left | 3 | 4.9 | Good parallax |
| back-bottom-right | 2 | 35.7 | **22° ray angle, 2 views only** |
| back-top-right | 3 | 25.4 | **22°/24°/46° angles** |

The large errors (25–36 mm) are **geometrically expected**, not algorithmic:
- Back-face points are occluded in the noisiest camera (Cam3, σ=3px)
- Remaining pairs have low parallax angles (20–23°)
- At 5m distance, σ=2px noise ≈ 6mm depth uncertainty per ray
- Error = noise / sin(θ) → 6 / sin(22°) ≈ 16mm per camera

---

### Task 3 — Robustness Analysis

#### 3A — Noise Sensitivity
σ=1px → 21.8mm mean error; σ=10px → 118mm.
Growth is roughly linear for pose errors, slightly super-linear for 3D
errors due to triangulation geometry.

#### 3B — Degenerate Geometry
Triangulation angle falls from 26° → 4.6°; 3D error grows 3.5×
(23mm → 82mm). Crossing below 10° is the practical danger threshold.
The condition number of A rises from 109k → 110k (modest, because
the cube's inherent geometry already makes A ill-conditioned at 5m).

#### 3C — Outlier Rejection
- **Vanilla LM:** ΔR = 6.65° — outlier rows in A corrupt DLT solution
- **RANSAC (4-pt):** ΔR = 0.50° — must sample 4 not 6 (P(clean 6/6)=0)
- **Huber LM:** ΔR = 0.56° — w_rob=δ/r bounds outlier influence
- **Cauchy LM:** ΔR = 0.50° — w_rob→0 for large residuals

---

## Known Limitations & Assumptions

1. **No lens distortion** — all cameras assumed to follow the pinhole model.
   Real lenses have radial and tangential distortion; ignoring it would
   systematically bias reprojection errors, especially at image corners.

2. **Known K** — intrinsics are assumed perfect. In practice K is estimated
   via calibration (checkerboard), which introduces its own uncertainty.

3. **Known 3D structure** — the pipeline is PnP (Perspective-n-Point),
   not full Structure-from-Motion. If the 3D points were unknown, a
   fundamentally different pipeline (e.g. feature matching + essential
   matrix + bundle adjustment) would be needed.

4. **Static scene** — no motion blur, no temporal changes assumed.

5. **Multi-start cost** — the 9-start LM adds ~0.15s per camera. In a
   real-time pipeline this is unacceptable; EPnP or SQPNP would replace DLT.

6. **RANSAC sample size** — sampling 4 points for a 12-DoF problem (P) is
   under-constrained. We compensate by running LM from multiple fixed starts
   per sample, but a proper minimal-case PnP solver (e.g. P3P) would be
   more principled.

---

## Profiling Output

Run the pipeline with Python's cProfile:
```bash
python -m cProfile -s cumtime task1_pose_estimation.py 2>&1 | head -30
```

**Measured runtimes (approximate, on standard laptop CPU):**

| Stage | Time |
|-------|------|
| Data loading | < 1 ms |
| Normalisation | < 1 ms |
| DLT (build A + SVD) | ~2 ms |
| LM refinement (single start) | ~15 ms |
| Multi-start LM (9 starts) | ~135 ms |
| Triangulation per point (DLT) | ~1 ms |
| Triangulation per point (optimal) | ~5 ms |
| **Task 3A total (4σ × 10 trials)** | ~120 s |
| **Task 3B total (7 alpha steps)** | ~15 s |
| **Task 3C total** | ~30 s |

**Bottleneck:** Multi-start LM in Task 3A dominates runtime.
Each trial calls 9-start LM for Cam3 (~135ms); 4σ × 10 trials = 40
pipeline calls = ~5.4s of LM alone, plus noise injection overhead.

**Proposed optimisation:**
Replace the 9-start LM with **EPnP** (Efficient Perspective-n-Point,
Lepetit et al. 2009). EPnP solves for the pose in closed form in O(n)
time by expressing 3D points as weighted sums of 4 control points and
solving a 12×12 eigenvalue problem. It achieves accuracy comparable to
iterative methods at a fraction of the cost (~0.1ms vs ~135ms), making
it suitable for real-time use and large-scale robustness sweeps.

