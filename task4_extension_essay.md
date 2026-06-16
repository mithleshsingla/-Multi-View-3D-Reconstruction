# Extension Essay
## "If the object were articulated (e.g. a 5-joint robot arm) instead of a rigid cube, how would you modify this pipeline?"

---

### Overview

A rigid cube has a single pose (6 DoF: 3 rotation + 3 translation). A
5-joint robot arm has **one global base pose** plus **5 joint angles** —
11 DoF total — and each link can move independently. This fundamentally
breaks every assumption in the current pipeline. Below I describe the
required modifications to pose estimation, triangulation, and optimisation.

---

### 1. Pose Estimation

**Rigid case:** DLT + LM finds one (R, t) per camera using all 8 cube corners
as a single rigid body.

**Articulated case:**

The arm has 6 rigid links (base + 5 segments). Each link has its own
coordinate frame defined by the **Denavit-Hartenberg (DH) convention**:

```
T_i = Rot_z(θᵢ) · Trans_z(dᵢ) · Trans_x(aᵢ) · Rot_x(αᵢ)
```

The 3D position of any point on link k is:

```
X_world = T_base · T_1(θ₁) · T_2(θ₂) · ... · T_k(θₖ) · X_local
```

So the projection equation becomes:

```
x = K [R_cam | t_cam] · T_base · ∏ᵢ₌₁ᵏ T_i(θᵢ) · X_local
```

The unknowns are now **{R_cam, t_cam, θ₁, …, θ₅}** (11 DoF).

**DLT cannot be used** — the projection equation is no longer linear in
the unknowns (joint angles appear inside trigonometric functions in the
rotation matrices). Instead, initialise with:

1. **Joint angle priors** from encoders or a forward-kinematics model.
2. **Segmentation** — detect which pixels belong to which link
   (instance segmentation or skeleton detection) and treat each link
   as a mini rigid body for a coarse initialisation.

For a 5-joint arm, a minimum of **11 point correspondences across all
links** (at least 2 per link to constrain its orientation) are needed
for a well-determined system. Occluded links require handling via the
kinematic chain — if link 3 is occluded, its joint angle can be inferred
from links 2 and 4 via the chain constraint (soft constraint in the loss).

---

### 2. Triangulation

**Rigid case:** All 8 corners are stationary — observations from different
cameras can be combined freely since the object has not moved between frames.

**Articulated case:**

Each link moves independently. Two new problems arise:

**2a. Temporal consistency:**
If cameras are not perfectly synchronised, the arm may have moved between
exposures. Observations from camera i at time tᵢ and camera j at time tⱼ
cannot be combined directly. Solution: use **rolling-shutter correction**
or enforce **time-stamped synchronisation** (hardware trigger, or
software interpolation of joint angles between exposures).

**2b. Per-link triangulation:**
Points on the same link share a rigid transformation, so triangulation
must be done **per link** using only the cameras and frames where that
link is visible. The kinematic chain constraint couples the links:

```
X_link_k = T_base · T_1 · T_2 · ... · T_k · X_local_k
```

This means triangulation and pose estimation cannot be separated — they
must be solved jointly via **body-aware bundle adjustment** (see §3).

**2c. Self-occlusion:**
A 5-joint arm will frequently occlude its own links. The visibility mask
changes per camera and per time step, making the observation table
dynamic. Robust triangulation must detect and discard self-occluded
observations automatically (e.g. via depth ordering in the kinematic chain).

---

### 3. Optimisation

**Rigid case:** Bundle adjustment minimises:

```
min_{R,t,X}  Σᵢ Σⱼ  wᵢⱼ ‖proj(Xⱼ; Kᵢ,Rᵢ,tᵢ) − xᵢⱼ‖²
```

**Articulated case:** The objective becomes:

```
min_{R_cam, t_cam, θ₁,...,θ₅, X_local}
    Σᵢ Σⱼ Σₖ  wᵢⱼₖ ‖proj(FK(θ,Xⱼₖ); Kᵢ,Rᵢ,tᵢ) − xᵢⱼₖ‖²
  + λ_smooth Σₜ ‖θ(t) − θ(t−1)‖²          (temporal smoothness)
  + λ_limit  Σₗ max(0, |θₗ| − θₗ_max)²    (joint limit penalty)
```

where FK(θ, X_local) is the **forward kinematics** function mapping
joint angles and local coordinates to world positions.

Key changes to the optimisation:

**3a. Jacobian structure:**
The Jacobian ∂proj/∂θₗ must account for the chain rule through all
downstream links. A change in θ₁ (the base joint) affects the position
of all 5 downstream links — the Jacobian is **dense** in θ but sparse
in the link-local coordinates. Automatic differentiation (PyTorch autograd
or JAX) is the most practical implementation strategy.

**3b. Joint limits as constraints:**
Physical joints have hard angle limits (e.g. elbow cannot hyperextend).
These are box constraints: θₗ ∈ [θₗ_min, θₗ_max]. Standard LM assumes
unconstrained optimisation; use **projected gradient descent** or
**interior point methods** instead, or add a log-barrier penalty.

**3c. Temporal bundle adjustment:**
If video is available, joint angles should be estimated jointly across
frames to enforce physical continuity. This yields a large sparse system
(variables: θ(1), …, θ(T) plus camera poses) that can be solved
efficiently with the **Schur complement trick** (marginalise structure,
solve for motion).

**3d. Initialisation sensitivity:**
The articulated objective has many more local minima than the rigid case
— the product of rotation matrices creates a highly non-convex landscape.
Good initialisation is critical: use **encoder readings** as priors
if available, or a **learned pose estimator** (e.g. HRNet for skeleton
keypoints) to warm-start the optimisation.

---

### Summary Table

| Aspect | Rigid Cube | 5-Joint Arm |
|--------|-----------|-------------|
| DoF | 6 | 11+ |
| Projection equation | Linear in P | Nonlinear (FK) |
| DLT applicable | ✓ | ✗ |
| Triangulation | Per-point, any camera | Per-link, synchronised |
| Optimisation | Bundle adjustment | Articulated BA + joint limits |
| Jacobian | Sparse, analytic | Dense through chain, use autodiff |
| Occlusion | Simple visibility mask | Self-occlusion from kinematic chain |
| Initialisation | Multi-start LM | Encoder priors + learned keypoints |

The core mathematical machinery — SVD, Levenberg–Marquardt, reprojection
error minimisation — remains the same. What changes is the **forward
model** (from a rigid projection to a kinematic chain), the **parameter
space** (from 6 to 11+ DoF), and the **constraints** (joint limits,
temporal continuity, self-occlusion). These changes make articulated
reconstruction a significantly harder problem, but one that is
tractable with modern automatic differentiation frameworks and
learned initialisation strategies.

