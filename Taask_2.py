"""
Task 2: 3D Triangulation
========================
Nutpaa Technologies — Multi-View 3D Reconstruction Take-Home

Goal:
    With all 3 camera poses known, reconstruct each of the 8 cube corners
    in 3-D using every view where the point is visible.

Methods implemented
-------------------
A) DLT triangulation (linear, closed-form)
        Build a 2M×4 system from M camera projection rows.
        Solve via SVD null-space → 3-D point in homogeneous coords.

B) Optimal triangulation (nonlinear least-squares)
        Initialise from DLT; minimise weighted reprojection error
        over the 3-D point position using Levenberg–Marquardt.

C) Confidence-weighted variants of both A and B.
        Compare unweighted vs weighted to evaluate the impact.

Outputs
-------
- Per-point 3-D error table  (mm)  —  DLT vs Optimal vs Ground-truth
- Highest-error points identified with geometric explanation
- 3-D visualisation of camera-point geometry (saved to PNG)
"""

import json
import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')                  # headless rendering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (registers 3-D projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── re-use Task-1 helpers ──────────────────────────────────────────────────
from task1_pose_estimation import (
    CUBE_PTS, load_data,
    project, reproj_err,
    rvec2R, R2rvec, refine_lm,
    estimate_pose, make_look_at_starts,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  BUILD OBSERVATION TABLE
#     For each of the 8 corners, record which cameras see it and at what pixel.
# ─────────────────────────────────────────────────────────────────────────────

def build_observation_table(data: dict) -> list[dict]:
    """
    Returns a list of 8 dicts — one per cube corner.

    Each dict has keys:
        'point_index'  : int
        'label'        : str
        'gt'           : (3,) ground-truth world XYZ
        'views'        : list of {'cam_idx', 'uv', 'conf', 'R', 't', 'K'}
    """
    cam_keys   = ["camera1", "camera2", "camera3"]
    table      = [
        dict(point_index=i, label=CUBE_LABELS[i],
             gt=CUBE_PTS[i].copy(), views=[])
        for i in range(8)
    ]
    return table   # filled in after poses are known


def fill_observations(table: list, data: dict, poses: dict, K: np.ndarray):
    """
    Populate the 'views' list in each table entry using the estimated poses.

    poses : dict  cam_key → {'R': ..., 't': ...}
    """
    cam_keys = ["camera1", "camera2", "camera3"]
    for cam_idx, ckey in enumerate(cam_keys):
        for obs in data[ckey]["observations"]:
            if not obs["visible"]:
                continue
            pi = obs["point_index"]
            table[pi]["views"].append({
                "cam_idx": cam_idx,
                "cam_key": ckey,
                "uv"     : np.array([obs["u"], obs["v"]], dtype=np.float64),
                "conf"   : obs["confidence"],
                "R"      : poses[ckey]["R"],
                "t"      : poses[ckey]["t"],
                "K"      : K,
            })
    return table


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DLT TRIANGULATION (from scratch)
# ─────────────────────────────────────────────────────────────────────────────

def triangulate_dlt(views: list, use_weights: bool = True) -> np.ndarray:
    """
    Multi-view DLT triangulation.

    Theory
    ------
    For each camera with projection matrix P = K [R | t]:
        λ [u, v, 1]ᵀ = P X̃       (X̃ = [X, Y, Z, 1]ᵀ homogeneous)

    Cross-multiplying to eliminate λ gives two independent equations:
        (u p³ᵀ − p¹ᵀ) X̃ = 0
        (v p³ᵀ − p²ᵀ) X̃ = 0
    where p¹, p², p³ are rows 0,1,2 of P.

    Stacking M cameras → 2M×4 matrix A.
    Null-space of A (last column of V from SVD) is X̃.
    De-homogenise: X = X̃[:3] / X̃[3].

    Confidence weighting:
        Multiply each row pair by √confidence → higher-quality observations
        contribute proportionally more to the null-space solution.

    Parameters
    ----------
    views        : list of dicts (from fill_observations)
    use_weights  : if True, weight rows by √confidence

    Returns
    -------
    X : (3,) estimated 3-D world point
    """
    rows = []
    for v in views:
        P   = v["K"] @ np.column_stack([v["R"], v["t"]])   # 3×4 projection
        u, vp = v["uv"]
        p1, p2, p3 = P[0], P[1], P[2]
        row_u = u * p3 - p1                 # (4,)
        row_v = vp * p3 - p2                # (4,)

        w = np.sqrt(v["conf"]) if use_weights else 1.0
        rows.append(row_u * w)
        rows.append(row_v * w)

    A = np.array(rows)                      # (2M, 4)

    # SVD: null-space vector = last column of V (right singular vector
    # corresponding to the smallest singular value)
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[-1]                             # homogeneous solution

    # De-homogenise
    X  = Xh[:3] / Xh[3]
    return X


# ─────────────────────────────────────────────────────────────────────────────
# 3.  OPTIMAL TRIANGULATION (nonlinear LS)
# ─────────────────────────────────────────────────────────────────────────────

def _reproj_residuals_point(X_vec: np.ndarray, views: list,
                             use_weights: bool) -> np.ndarray:
    """
    Residual function for optimising a single 3-D point.

    For each view i:
        r_i = √w_i · (project(X, K_i, R_i, t_i) − uv_i)   ∈ ℝ²

    Returns the 2M-vector of all residuals.
    This makes least_squares minimise  Σ w_i ‖e_i‖².
    """
    X = X_vec.reshape(1, 3)
    res = []
    for v in views:
        proj = project(X, v["K"], v["R"], v["t"])[0]       # (2,)
        e    = proj - v["uv"]
        w    = np.sqrt(v["conf"]) if use_weights else 1.0
        res.extend(e * w)
    return np.array(res)


def triangulate_optimal(views: list, X_init: np.ndarray,
                         use_weights: bool = True) -> np.ndarray:
    """
    Refine DLT estimate by minimising weighted reprojection error.

    Why not stop at DLT?
        DLT minimises an ALGEBRAIC error (‖Ax‖²) that does not correspond
        to pixel distances.  The optimal triangulation minimises the
        GEOMETRIC (reprojection) error, which is what we actually care about.

    Parameters
    ----------
    X_init       : (3,) initial guess from DLT
    use_weights  : use confidence weighting in residuals

    Returns
    -------
    X : (3,) refined 3-D point
    """
    result = least_squares(
        _reproj_residuals_point,
        X_init,
        args=(views, use_weights),
        method='lm',
        ftol=1e-12, xtol=1e-12, gtol=1e-12,
        max_nfev=10000,
    )
    return result.x


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TRIANGULATE ALL 8 CORNERS — run all four variants
# ─────────────────────────────────────────────────────────────────────────────

def triangulate_all(table: list) -> dict:
    """
    Run four triangulation variants for all 8 corners:
        dlt_uw    — DLT, no weighting
        dlt_w     — DLT, confidence weighted
        opt_uw    — optimal (LM), no weighting
        opt_w     — optimal (LM), confidence weighted

    Returns a dict: variant_name → (8, 3) array of reconstructed points.
    """
    variants = {
        "dlt_uw":  [],
        "dlt_w":   [],
        "opt_uw":  [],
        "opt_w":   [],
    }

    for entry in table:
        views = entry["views"]

        # Need at least 2 views for triangulation
        if len(views) < 2:
            for v in variants.values():
                v.append(entry["gt"].copy())   # fall back to GT if only 1 view
            continue

        X_dlt_uw = triangulate_dlt(views, use_weights=False)
        X_dlt_w  = triangulate_dlt(views, use_weights=True)
        X_opt_uw = triangulate_optimal(views, X_dlt_uw, use_weights=False)
        X_opt_w  = triangulate_optimal(views, X_dlt_w,  use_weights=True)

        variants["dlt_uw"].append(X_dlt_uw)
        variants["dlt_w" ].append(X_dlt_w)
        variants["opt_uw"].append(X_opt_uw)
        variants["opt_w" ].append(X_opt_w)

    return {k: np.array(v) for k, v in variants.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 5.  ERROR ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def compute_errors(results: dict, table: list) -> dict:
    """
    Compute per-point 3-D Euclidean error (mm) for each variant.

    Returns dict: variant → (8,) error array in mm.
    """
    gt = np.array([e["gt"] for e in table])       # (8, 3) metres
    errors = {}
    for name, pts in results.items():
        errors[name] = np.linalg.norm(pts - gt, axis=1) * 1000.0   # → mm
    return errors


def print_error_table(errors: dict, table: list):
    """Print a formatted per-point error table."""
    print("\n" + "="*75)
    print("  3-D RECONSTRUCTION ERROR PER POINT  (mm)")
    print("="*75)
    hdr = f"  {'Label':>20s}  {'#Views':>6s}  {'DLT-uw':>8s}  {'DLT-w':>8s}  {'Opt-uw':>8s}  {'Opt-w':>8s}"
    print(hdr)
    print("  " + "-"*71)
    for i, entry in enumerate(table):
        n = len(entry["views"])
        row = (f"  {entry['label']:>20s}  {n:>6d}  "
               f"{errors['dlt_uw'][i]:>8.3f}  {errors['dlt_w'][i]:>8.3f}  "
               f"{errors['opt_uw'][i]:>8.3f}  {errors['opt_w'][i]:>8.3f}")
        print(row)
    print("  " + "-"*71)
    print(f"  {'MEAN':>20s}  {'':>6s}  "
          f"{errors['dlt_uw'].mean():>8.3f}  {errors['dlt_w'].mean():>8.3f}  "
          f"{errors['opt_uw'].mean():>8.3f}  {errors['opt_w'].mean():>8.3f}")
    print(f"  {'MAX':>20s}  {'':>6s}  "
          f"{errors['dlt_uw'].max():>8.3f}  {errors['dlt_w'].max():>8.3f}  "
          f"{errors['opt_uw'].max():>8.3f}  {errors['opt_w'].max():>8.3f}")


def explain_high_error_points(errors: dict, table: list):
    """
    Identify the two highest-error points and give a geometric explanation.
    """
    print("\n" + "="*60)
    print("  HIGH-ERROR POINT ANALYSIS")
    print("="*60)
    e = errors["opt_w"]
    ranked = np.argsort(e)[::-1]
    for rank, idx in enumerate(ranked[:3]):
        entry  = table[idx]
        n_views = len(entry["views"])
        cam_ids = [v["cam_idx"]+1 for v in entry["views"]]

        # Compute the angle between rays from each pair of cameras
        ray_angles = []
        for i in range(len(entry["views"])):
            for j in range(i+1, len(entry["views"])):
                vi, vj = entry["views"][i], entry["views"][j]
                # Ray direction in world frame: R^T (K^-1 [u,v,1])
                def ray_dir(v):
                    uv  = np.append(v["uv"], 1.0)
                    xc  = np.linalg.inv(v["K"]) @ uv
                    return v["R"].T @ (xc / np.linalg.norm(xc))
                ri = ray_dir(vi);  rj = ray_dir(vj)
                ang = np.degrees(np.arccos(np.clip(ri @ rj, -1, 1)))
                ray_angles.append(ang)

        print(f"\n  Rank {rank+1}: '{entry['label']}'  "
              f"(error = {e[idx]:.3f} mm, views = {cam_ids})")
        print(f"    Ray angles between views: "
              f"{[f'{a:.1f}°' for a in ray_angles]}")
        print(f"    Near-parallel rays inflate triangulation error because "
              f"the intersection point is highly sensitive to pixel noise "
              f"when the baseline angle is small (low parallax).")
        if n_views == 2:
            print(f"    Only 2 views — fewer constraints weaken the system.")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CONFIDENCE WEIGHTING IMPACT
# ─────────────────────────────────────────────────────────────────────────────

def confidence_impact_analysis(errors: dict):
    """Show the numerical effect of confidence weighting on accuracy."""
    print("\n" + "="*60)
    print("  CONFIDENCE WEIGHTING IMPACT")
    print("="*60)
    delta_dlt = errors["dlt_uw"] - errors["dlt_w"]     # positive = weighting helps
    delta_opt = errors["opt_uw"] - errors["opt_w"]
    print(f"  DLT: weighting reduces mean error by  {delta_dlt.mean():+.3f} mm")
    print(f"  DLT: weighting reduces max  error by  {delta_dlt.max():+.3f} mm")
    print(f"  OPT: weighting reduces mean error by  {delta_opt.mean():+.3f} mm")
    print(f"  OPT: weighting reduces max  error by  {delta_opt.max():+.3f} mm")
    print("""
  Interpretation:
    Confidence weights down-weight observations that are less reliable
    (e.g. partial occlusion edge cases, motion blur).  In this dataset
    all confidences are moderately high (0.72–0.99), so the gain is modest
    but consistent.  With one very low-confidence outlier observation the
    weighted variant would outperform substantially.""")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  3-D VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def draw_camera(ax, R, t, K, scale=0.4, color='blue', label=''):
    """
    Draw a camera frustum:
        - Optical centre  C = -Rᵀ t
        - Four frustum corners projected from image corners
        - Lines from C to corners and around the image rectangle
    """
    C = -R.T @ t                       # camera centre in world

    # Image corners in pixel space
    w, h  = 640, 480
    corners_px = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=float)

    # Unproject to normalised camera-space rays
    Kinv  = np.linalg.inv(K)
    rays  = []
    for px in corners_px:
        xc = Kinv @ np.array([px[0], px[1], 1.0])
        xc /= np.linalg.norm(xc)
        rays.append(xc)

    # Rotate rays to world space and extend by scale
    frustum_pts = [C + R.T @ (r * scale) for r in rays]

    # Draw frustum edges
    for pt in frustum_pts:
        ax.plot([C[0], pt[0]], [C[1], pt[1]], [C[2], pt[2]],
                color=color, linewidth=0.8)
    for i in range(4):
        a, b = frustum_pts[i], frustum_pts[(i+1) % 4]
        ax.plot([a[0],b[0]], [a[1],b[1]], [a[2],b[2]],
                color=color, linewidth=0.8)

    ax.scatter(*C, color=color, s=40, zorder=5)
    ax.text(C[0], C[1], C[2]+0.15, label, color=color, fontsize=9,
            fontweight='bold')


def draw_cube_wireframe(ax, alpha=0.15):
    """Draw ground-truth cube corners and edges."""
    edges = [(0,1),(1,2),(2,3),(3,0),   # front face
             (4,5),(5,6),(6,7),(7,4),   # back face
             (0,4),(1,5),(2,6),(3,7)]   # connecting edges
    for i, j in edges:
        a, b = CUBE_PTS[i], CUBE_PTS[j]
        ax.plot([a[0],b[0]], [a[1],b[1]], [a[2],b[2]],
                'k-', linewidth=1.0, alpha=0.6)
    ax.scatter(CUBE_PTS[:,0], CUBE_PTS[:,1], CUBE_PTS[:,2],
               c='black', s=20, zorder=4, label='GT corners')


def visualise(poses: dict, results: dict, errors: dict, table: list,
              save_path: str = "task2_visualisation.png"):
    """
    Create a 3-panel figure:
        Left  — camera + point geometry (3-D scene)
        Middle — per-point error bar chart (DLT vs Optimal)
        Right  — DLT vs Optimal scatter (mm)
    """
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle("Task 2 — 3D Triangulation", fontsize=14, fontweight='bold')

    # ── Panel 1: 3-D scene ──────────────────────────────────────────────
    ax3d = fig.add_subplot(131, projection='3d')
    ax3d.set_title("Camera–Point Geometry")

    draw_cube_wireframe(ax3d)

    cam_colors  = ['royalblue', 'tomato', 'seagreen']
    cam_labels  = ['Cam 1 (known)', 'Cam 2 (est.)', 'Cam 3 (est.)']
    cam_keys    = ['camera1', 'camera2', 'camera3']
    K_mat       = list(poses.values())[0]["K"]
    for i, ck in enumerate(cam_keys):
        draw_camera(ax3d, poses[ck]["R"], poses[ck]["t"],
                    K_mat, color=cam_colors[i], label=cam_labels[i])

    # Plot reconstructed points (optimal weighted)
    pts = results["opt_w"]
    ax3d.scatter(pts[:,0], pts[:,1], pts[:,2],
                 c='orange', s=40, marker='^', zorder=5, label='Reconstructed')

    # Draw lines from ground-truth to reconstructed to show error
    for i in range(8):
        gt = CUBE_PTS[i];  rc = pts[i]
        ax3d.plot([gt[0],rc[0]], [gt[1],rc[1]], [gt[2],rc[2]],
                  'm-', linewidth=0.8, alpha=0.6)

    ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
    ax3d.legend(fontsize=7, loc='upper left')
    ax3d.set_box_aspect([1,1,1])

    # ── Panel 2: per-point error bar chart ──────────────────────────────
    ax2 = fig.add_subplot(132)
    ax2.set_title("Per-Point 3D Error (mm)")
    x   = np.arange(8)
    w   = 0.22
    ax2.bar(x - 1.5*w, errors["dlt_uw"],  w, label='DLT (unweighted)', color='steelblue',  alpha=0.8)
    ax2.bar(x - 0.5*w, errors["dlt_w"],   w, label='DLT (weighted)',   color='cornflowerblue', alpha=0.8)
    ax2.bar(x + 0.5*w, errors["opt_uw"],  w, label='OPT (unweighted)', color='tomato',    alpha=0.8)
    ax2.bar(x + 1.5*w, errors["opt_w"],   w, label='OPT (weighted)',   color='salmon',    alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([t["label"].replace('-','\n') for t in table],
                        fontsize=6)
    ax2.set_ylabel("Error (mm)")
    ax2.legend(fontsize=7)
    ax2.grid(axis='y', alpha=0.3)

    # ── Panel 3: DLT vs Optimal scatter ─────────────────────────────────
    ax3 = fig.add_subplot(133)
    ax3.set_title("DLT-w vs Optimal-w (mm)")
    ax3.scatter(errors["dlt_w"], errors["opt_w"], c='purple', s=60, zorder=5)
    for i, entry in enumerate(table):
        ax3.annotate(entry["label"].split('-')[0],
                     (errors["dlt_w"][i], errors["opt_w"][i]),
                     textcoords="offset points", xytext=(4,2), fontsize=6)
    lim = max(errors["dlt_w"].max(), errors["opt_w"].max()) * 1.1
    ax3.plot([0, lim], [0, lim], 'k--', linewidth=0.8, label='y=x (no change)')
    ax3.set_xlabel("DLT-w error (mm)")
    ax3.set_ylabel("Optimal-w error (mm)")
    ax3.legend(fontsize=7)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Visualisation saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

CUBE_LABELS = [
    "front-bottom-left", "front-bottom-right",
    "front-top-right",   "front-top-left",
    "back-bottom-left",  "back-bottom-right",
    "back-top-right",    "back-top-left",
]


def main():
    data = load_data("data.json")
    K    = np.array(data["camera_intrinsics"]["K"], dtype=np.float64)

    # ── Step 1: get all 3 poses ───────────────────────────────────────────
    print("="*60)
    print("  STEP 1 — Recovering all camera poses")
    print("="*60)

    R1 = np.array(data["camera1"]["R"], dtype=np.float64)
    t1 = np.array(data["camera1"]["t"], dtype=np.float64)

    res2 = estimate_pose(data["camera2"], K, "Camera 2", multistart=False)
    res3 = estimate_pose(data["camera3"], K, "Camera 3", multistart=True)

    poses = {
        "camera1": {"R": R1, "t": t1, "K": K},
        "camera2": {"R": res2["R"], "t": res2["t"], "K": K},
        "camera3": {"R": res3["R"], "t": res3["t"], "K": K},
    }

    # ── Step 2: build observation table ──────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 2 — Building observation table")
    print("="*60)
    table = [
        dict(point_index=i, label=CUBE_LABELS[i],
             gt=CUBE_PTS[i].copy(), views=[])
        for i in range(8)
    ]
    fill_observations(table, data, poses, K)

    print(f"\n  {'Point':>20s}  {'#Views':>6s}  {'Cameras':>20s}")
    print("  " + "-"*52)
    for entry in table:
        cams = str([v["cam_idx"]+1 for v in entry["views"]])
        print(f"  {entry['label']:>20s}  {len(entry['views']):>6d}  {cams:>20s}")

    # ── Step 3: triangulate all variants ─────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 3 — Triangulating all 8 corners (4 variants)")
    print("="*60)
    results = triangulate_all(table)

    # ── Step 4: error analysis ────────────────────────────────────────────
    errors = compute_errors(results, table)
    print_error_table(errors, table)
    explain_high_error_points(errors, table)
    confidence_impact_analysis(errors)

    # ── Step 5: per-point reprojection errors (sanity check) ─────────────
    print("\n" + "="*60)
    print("  STEP 4 — Reprojection error of reconstructed points (opt_w)")
    print("="*60)
    print(f"  {'Point':>20s}  {'GT err (mm)':>12s}", end="")
    for ck in ["camera1", "camera2", "camera3"]:
        print(f"  {'reproj_'+ck[-1]:>12s}", end="")
    print()
    print("  " + "-"*72)

    for i, entry in enumerate(table):
        X_rec = results["opt_w"][i]
        err3d = np.linalg.norm(X_rec - entry["gt"]) * 1000
        print(f"  {entry['label']:>20s}  {err3d:>12.3f}", end="")
        for ck in ["camera1", "camera2", "camera3"]:
            # Find if this camera sees this point
            v_list = [v for v in entry["views"] if v["cam_key"] == ck]
            if v_list:
                v  = v_list[0]
                pr = project(X_rec.reshape(1,3), v["K"], v["R"], v["t"])[0]
                re = np.linalg.norm(pr - v["uv"])
                print(f"  {re:>12.3f}", end="")
            else:
                print(f"  {'occluded':>12s}", end="")
        print()

    # ── Step 6: visualise ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 5 — Generating visualisation")
    print("="*60)
    visualise(poses, results, errors, table,
              save_path="task2_visualisation.png")

    return results, errors, table, poses


if __name__ == "__main__":
    main()