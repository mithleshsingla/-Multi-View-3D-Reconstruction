"""
Task 1: Camera Pose Estimation
==============================
Nutpaa Technologies — Multi-View 3D Reconstruction Take-Home

Goal:
    Estimate the 6-DoF pose (R, t) for Camera 2 and Camera 3 from
    known 3D world points and their noisy 2D projections.

Method:
    1. Direct Linear Transform (DLT) — closed-form initial estimate of
       the 3×4 projection matrix P, factored into R and t using known K.
    2. Levenberg–Marquardt (LM) nonlinear refinement — minimises the
       weighted reprojection error.
    3. Multi-start LM — sweeps candidate initializations to escape local
       minima caused by the cube's near-symmetric geometry.

Assumptions:
    - All cameras share one known intrinsic matrix K (no lens distortion).
    - Camera 1 is the reference frame: R1 = I, t1 = [0, 0, 5].
    - Occluded observations (visible=False) are SKIPPED — no interpolation.
    - Confidence scores weight the reprojection residuals.
"""

import json
import numpy as np
from scipy.optimize import least_squares


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

# Ground-truth 3-D cube corners in world coordinates (fixed)
CUBE_PTS = np.array([
    [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5],
    [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
    [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5],
    [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
], dtype=np.float64)


def load_data(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_camera_observations(cam_data: dict):
    """
    Return only VISIBLE observations — skip nulls entirely.

    Returns
    -------
    pts3d : (N,3)  world XYZ
    pts2d : (N,2)  observed pixel (u,v)
    confs : (N,)   confidence weights
    """
    pts3d, pts2d, confs = [], [], []
    for obs in cam_data["observations"]:
        if not obs["visible"]:
            continue
        pts3d.append(CUBE_PTS[obs["point_index"]])
        pts2d.append([obs["u"], obs["v"]])
        confs.append(obs["confidence"])
    return (np.array(pts3d, dtype=np.float64),
            np.array(pts2d, dtype=np.float64),
            np.array(confs, dtype=np.float64))


# ─────────────────────────────────────────────────────────────────────────────
# 2.  HARTLEY NORMALISATION (numerical conditioning for DLT)
# ─────────────────────────────────────────────────────────────────────────────

def normalise_2d(pts: np.ndarray):
    """
    Shift centroid to origin; scale so RMS distance = √2.
    Returns normalised points and the 3×3 transform T.
    """
    c  = pts.mean(axis=0)
    s  = np.sqrt(2) / max(np.sqrt(((pts - c)**2).sum(axis=1)).mean(), 1e-8)
    T  = np.array([[s, 0, -s*c[0]],
                   [0, s, -s*c[1]],
                   [0, 0,       1]], dtype=np.float64)
    ph = np.column_stack([pts, np.ones(len(pts))])
    return (T @ ph.T).T[:, :2], T


def normalise_3d(pts: np.ndarray):
    """
    Same idea in 3-D; RMS distance → √3.
    Returns normalised points and the 4×4 homogeneous transform T.
    """
    c  = pts.mean(axis=0)
    s  = np.sqrt(3) / max(np.sqrt(((pts - c)**2).sum(axis=1)).mean(), 1e-8)
    T  = np.eye(4, dtype=np.float64)
    T[:3, :3] *= s
    T[:3,  3]  = -s * c
    ph = np.column_stack([pts, np.ones(len(pts))])
    return (T @ ph.T).T[:, :3], T


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DLT — Direct Linear Transform
# ─────────────────────────────────────────────────────────────────────────────

def build_dlt_matrix(pts3d_n: np.ndarray,
                     pts2d_n: np.ndarray,
                     weights: np.ndarray) -> np.ndarray:
    """
    Build 2N×12 design matrix A.

    Projection equation: λ[u,v,1]ᵀ = P [X,Y,Z,1]ᵀ
    Cross-multiplying to eliminate λ gives two rows per point:
        row 2i  : [  0ᵀ | -X̃ᵀ |  v·X̃ᵀ ]
        row 2i+1: [  X̃ᵀ |  0ᵀ | -u·X̃ᵀ ]
    Each row pair is scaled by √confidence.
    """
    N = len(pts3d_n)
    A = np.zeros((2 * N, 12))
    for i, (X, x, w) in enumerate(zip(pts3d_n, pts2d_n, weights)):
        Xh = np.append(X, 1.0)
        u, v = x
        sw = np.sqrt(max(w, 0.0))
        A[2*i,   4:8 ] = -Xh;    A[2*i,   8:12] =  v * Xh
        A[2*i+1, 0:4 ] =  Xh;    A[2*i+1, 8:12] = -u * Xh
        A[2*i]   *= sw
        A[2*i+1] *= sw
    return A


def dlt_solve(pts3d: np.ndarray,
              pts2d: np.ndarray,
              weights: np.ndarray) -> np.ndarray:
    """
    Solve A p = 0  via SVD null-space → reshape to 3×4 matrix P.

    The null-space vector (last column of V from SVD of A) is the
    unit-norm solution minimising ‖Ap‖².  De-normalise at the end.
    """
    pts3d_n, T3 = normalise_3d(pts3d)
    pts2d_n, T2 = normalise_2d(pts2d)
    A  = build_dlt_matrix(pts3d_n, pts2d_n, weights)
    _, _, Vt = np.linalg.svd(A)
    P_n = Vt[-1].reshape(3, 4)
    return np.linalg.inv(T2) @ P_n @ T3   # de-normalise


def decompose_P(P: np.ndarray, K: np.ndarray):
    """
    Extract R and t from  P = K [R | t]  given known K.

    K⁻¹ P = [R | t]
    Left 3×3 block → nearest rotation via SVD polar decomposition.
    Scale t by the mean singular value of the left block.
    """
    Kinv = np.linalg.inv(K)
    RT   = Kinv @ P
    M, tv = RT[:, :3], RT[:, 3]
    U, sv, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:          # guard against reflections
        U[:, -1] *= -1
        R = U @ Vt
    return R, tv / sv.mean()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PROJECTION & REPROJECTION ERROR
# ─────────────────────────────────────────────────────────────────────────────

def project(pts3d: np.ndarray, K: np.ndarray,
            R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    X_cam = R @ X_world + t
    x_hom = K @ X_cam
    (u,v) = x_hom[:2] / x_hom[2]
    """
    Xc  = (R @ pts3d.T).T + t
    xh  = (K @ Xc.T).T
    return xh[:, :2] / xh[:, 2:3]


def reproj_err(pts3d, pts2d, K, R, t, w) -> float:
    """Weighted mean reprojection error in pixels."""
    e = np.linalg.norm(project(pts3d, K, R, t) - pts2d, axis=1)
    return float(np.average(e, weights=w))


# ─────────────────────────────────────────────────────────────────────────────
# 5.  RODRIGUES ↔ MATRIX CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

def rvec2R(rv: np.ndarray) -> np.ndarray:
    """
    Rodrigues vector → rotation matrix.
    rvec = θ·k̂  (angle × unit axis)
    R = I cosθ + (1−cosθ) k̂k̂ᵀ + [k̂]× sinθ
    Minimal 3-parameter representation for LM optimisation.
    """
    th = np.linalg.norm(rv)
    if th < 1e-9:
        return np.eye(3)
    k  = rv / th
    K_ = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3)*np.cos(th) + (1-np.cos(th))*np.outer(k,k) + np.sin(th)*K_


def R2rvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → Rodrigues vector."""
    tr = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(tr)
    if abs(th) < 1e-9:
        return np.zeros(3)
    ax = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) / (2*np.sin(th))
    return th * ax


# ─────────────────────────────────────────────────────────────────────────────
# 6.  LEVENBERG–MARQUARDT REFINEMENT
# ─────────────────────────────────────────────────────────────────────────────

def lm_residuals(params, pts3d, pts2d, K, w):
    """
    Weighted pixel residuals for LM.
    params = [rvec(3), t(3)]
    Returns 2N vector: √wᵢ · (proj_i − obs_i)
    Squaring and summing gives  Σ wᵢ ‖eᵢ‖²  — the weighted LS objective.
    """
    R   = rvec2R(params[:3])
    t   = params[3:6]
    res = (project(pts3d, K, R, t) - pts2d) * np.sqrt(w[:, None])
    return res.ravel()


def refine_lm(R0, t0, pts3d, pts2d, K, w):
    """Single LM run from one starting pose."""
    p0  = np.concatenate([R2rvec(R0), t0])
    res = least_squares(lm_residuals, p0, args=(pts3d, pts2d, K, w),
                        method='lm', ftol=1e-12, xtol=1e-12, gtol=1e-12,
                        max_nfev=10000)
    return rvec2R(res.x[:3]), res.x[3:6]


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MULTI-START: generate look-at-origin candidate poses
# ─────────────────────────────────────────────────────────────────────────────

def make_look_at_starts(n_az=16, n_el=5, distances=(3., 5., 7.)):
    """
    Sample camera positions on a sphere at multiple radii; build rotation
    so the camera Z-axis points at the world origin.

    Why multi-start?
        A 1 m cube has near-symmetric projections for mirrored camera poses.
        DLT can converge to a reflected solution that forms a different
        LM basin.  Sampling the sphere exhaustively lets us find the basin
        with the lowest reprojection error.
    """
    starts = []
    for d in distances:
        for az in np.linspace(-np.pi, np.pi, n_az, endpoint=False):
            for el in np.linspace(-0.5, 0.5, n_el):
                tx = d * np.sin(az) * np.cos(el)
                ty = d * np.sin(el)
                tz = d * np.cos(az) * np.cos(el)
                z  = np.array([-tx, -ty, -tz]);  z /= np.linalg.norm(z)
                up = np.array([0., 1., 0.])
                x  = np.cross(up, z)
                if np.linalg.norm(x) < 1e-6:
                    up = np.array([1., 0., 0.])
                    x  = np.cross(up, z)
                x /= np.linalg.norm(x)
                y  = np.cross(z, x)
                starts.append((np.stack([x, y, z]), np.array([tx, ty, tz])))
    return starts


# ─────────────────────────────────────────────────────────────────────────────
# 8.  FULL POSE ESTIMATION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def estimate_pose(cam_data: dict, K: np.ndarray,
                  cam_name: str, multistart: bool = False) -> dict:
    """
    End-to-end pose estimation for one camera.

    Pipeline
    --------
    1. Extract visible observations (skip occluded).
    2. DLT via SVD null-space  →  initial P  →  decompose to R, t.
    3. LM refinement (single-start or multi-start).
    4. Report DLT and LM reprojection errors.

    Parameters
    ----------
    multistart : if True, run LM from 241 grid starts + DLT start
                 and keep the best result.  Use for cameras with bad DLT init.
    """
    print(f"\n{'='*60}")
    print(f"  Estimating pose for {cam_name}")
    print(f"{'='*60}")

    pts3d, pts2d, confs = extract_camera_observations(cam_data)
    N = len(pts3d)
    print(f"  Visible points : {N}  (occluded: {8 - N})")
    print(f"  Confidences    : {np.round(confs, 3)}")

    if N < 6:
        raise ValueError(f"Need ≥ 6 points for DLT (got {N}).")

    # ── DLT ──────────────────────────────────────────────────────────────
    print("\n  [DLT] Building normalised system and solving SVD null-space ...")
    P      = dlt_solve(pts3d, pts2d, confs)
    R_dlt, t_dlt = decompose_P(P, K)
    err_dlt = reproj_err(pts3d, pts2d, K, R_dlt, t_dlt, confs)
    print(f"  [DLT] P:\n{np.round(P, 4)}")
    print(f"  [DLT] R:\n{np.round(R_dlt, 4)}")
    print(f"  [DLT] t: {np.round(t_dlt, 4)}")
    print(f"  [DLT] Weighted reprojection error: {err_dlt:.4f} px")

    # ── LM refinement ────────────────────────────────────────────────────
    if multistart:
        starts = [(R_dlt, t_dlt)] + make_look_at_starts()
        print(f"\n  [Multi-start LM] Trying {len(starts)} initializations ...")
        best_err = 1e9
        best_R = best_t = None
        for R0, t0 in starts:
            R_, t_ = refine_lm(R0, t0, pts3d, pts2d, K, confs)
            err = reproj_err(pts3d, pts2d, K, R_, t_, confs)
            if err < best_err:
                best_err, best_R, best_t = err, R_, t_
        R_lm, t_lm, err_lm = best_R, best_t, best_err
    else:
        print("\n  [LM] Refining from DLT initialisation ...")
        R_lm, t_lm = refine_lm(R_dlt, t_dlt, pts3d, pts2d, K, confs)
        err_lm = reproj_err(pts3d, pts2d, K, R_lm, t_lm, confs)

    print(f"  [LM] R:\n{np.round(R_lm, 5)}")
    print(f"  [LM] t: {np.round(t_lm, 5)}")
    print(f"  [LM] Weighted reprojection error : {err_lm:.4f} px")
    print(f"  Improvement DLT → LM            : {err_dlt - err_lm:.4f} px")

    C = -R_lm.T @ t_lm
    print(f"  Camera centre (world)            : {np.round(C, 5)}")

    return dict(camera=cam_name, R=R_lm, t=t_lm,
                R_dlt=R_dlt, t_dlt=t_dlt,
                err_dlt_px=err_dlt, err_lm_px=err_lm,
                camera_centre=C, num_visible=N)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  CAMERA 1 SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def verify_camera1(data: dict, K: np.ndarray):
    """
    Project Camera 1's known pose onto its observations.
    Error should be ≈ σ = 2 px (the data noise level).
    Confirms our projection and error functions are correct.
    """
    R1 = np.array(data["camera1"]["R"], dtype=np.float64)
    t1 = np.array(data["camera1"]["t"], dtype=np.float64)
    pts3d, pts2d, confs = extract_camera_observations(data["camera1"])
    proj = project(pts3d, K, R1, t1)

    print("\n" + "="*60)
    print("  SANITY CHECK — Camera 1 (known pose)")
    print("="*60)
    print(f"  Known R1 = I, t1 = [0,0,5]")
    print(f"  Weighted reprojection error (should be ≈ σ=2px): "
          f"{reproj_err(pts3d, pts2d, K, R1, t1, confs):.4f} px")
    print(f"\n  {'Label':>20s}  {'u_obs':>7s}  {'v_obs':>7s}  "
          f"{'u_proj':>7s}  {'v_proj':>7s}  {'err_px':>7s}")
    for obs, ob, pr in zip(data["camera1"]["observations"], pts2d, proj):
        if obs["visible"]:
            print(f"  {obs['label']:>20s}  {ob[0]:7.2f}  {ob[1]:7.2f}  "
                  f"  {pr[0]:7.2f}  {pr[1]:7.2f}  "
                  f"{np.linalg.norm(ob-pr):7.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 10.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    data = load_data("data.json")
    K    = np.array(data["camera_intrinsics"]["K"], dtype=np.float64)
    print(f"Intrinsic matrix K:\n{K}")

    verify_camera1(data, K)

    # Camera 2: DLT gives a good init → single-start LM is sufficient
    res2 = estimate_pose(data["camera2"], K, "Camera 2", multistart=False)

    # Camera 3: DLT lands in a bad local minimum (cube symmetry + high noise)
    #           → use multi-start LM to find the correct basin
    res3 = estimate_pose(data["camera3"], K, "Camera 3", multistart=True)

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  FINAL SUMMARY")
    print("="*60)
    for res in [res2, res3]:
        print(f"\n  {res['camera']}")
        print(f"    R:\n{np.round(res['R'], 5)}")
        print(f"    t:              {np.round(res['t'], 5)}")
        print(f"    Camera centre:  {np.round(res['camera_centre'], 5)}")
        print(f"    Error (DLT):    {res['err_dlt_px']:.4f} px")
        print(f"    Error (LM):     {res['err_lm_px']:.4f} px")

    return res2, res3


if __name__ == "__main__":
    main()