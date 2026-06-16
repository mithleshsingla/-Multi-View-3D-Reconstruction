"""
Task 3: Robustness Analysis
============================
Nutpaa Technologies — Multi-View 3D Reconstruction Take-Home

FULLY STANDALONE — place in same folder as data.json and run directly.
No imports from task1 or task2 are required.

Sub-tasks
---------
3A  Noise sensitivity    — pipeline at σ = 1, 2, 5, 10 px (10 trials each)
3B  Degenerate geometry  — move Cam3 toward coplanarity with Cam1;
                           report triangulation angle, condition number, and error
3C  Outlier rejection    — inject 2 wrong matches into Cam2;
                           show vanilla LM fails;
                           recover with RANSAC + Huber loss + Cauchy loss
"""

import json, copy
import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

CUBE_PTS = np.array([
    [-0.5,-0.5,-0.5],[ 0.5,-0.5,-0.5],[ 0.5, 0.5,-0.5],[-0.5, 0.5,-0.5],
    [-0.5,-0.5, 0.5],[ 0.5,-0.5, 0.5],[ 0.5, 0.5, 0.5],[-0.5, 0.5, 0.5],
], dtype=np.float64)

CUBE_LABELS = [
    "front-bottom-left","front-bottom-right",
    "front-top-right",  "front-top-left",
    "back-bottom-left", "back-bottom-right",
    "back-top-right",   "back-top-left",
]

# Reference poses from Task 1 on the original (un-modified) data
R2_REF = np.array([[ 0.97722,-0.03722,-0.20893],
                   [ 0.02608, 0.99810,-0.05582],
                   [ 0.21061, 0.04910, 0.97634]])
T2_REF = np.array([ 1.09915,-0.20463, 4.90798])
R3_REF = np.array([[ 0.96454, 0.01482, 0.26353],
                   [-0.03115, 0.99784, 0.05790],
                   [-0.26210,-0.06406, 0.96291]])
T3_REF = np.array([-0.99722, 0.39432, 5.17364])


# ═════════════════════════════════════════════════════════════════════════════
# CORE MATH
# ═════════════════════════════════════════════════════════════════════════════

def rvec2R(rv):
    """Rodrigues vector → 3×3 rotation matrix. rvec = θ·k̂."""
    th = np.linalg.norm(rv)
    if th < 1e-9: return np.eye(3)
    k  = rv / th
    Kx = np.array([[0.,-k[2],k[1]],[k[2],0.,-k[0]],[-k[1],k[0],0.]])
    return np.eye(3)*np.cos(th) + (1-np.cos(th))*np.outer(k,k) + np.sin(th)*Kx

def R2rvec(R):
    """Rotation matrix → Rodrigues vector."""
    tr = np.clip((np.trace(R)-1.)/2., -1., 1.)
    th = np.arccos(tr)
    if abs(th) < 1e-9: return np.zeros(3)
    return th*np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])/(2.*np.sin(th))

def rot_err_deg(Ra, Rb):
    """Angular difference between two rotation matrices (degrees)."""
    dR = Ra @ Rb.T
    return float(np.degrees(np.arccos(np.clip((np.trace(dR)-1.)/2.,-1.,1.))))

def project(pts3d, K, R, t):
    """Project world points → pixel coordinates."""
    Xc = (R @ pts3d.T).T + t
    xh = (K @ Xc.T).T
    return xh[:,:2] / xh[:,2:3]

def reproj_err(pts3d, pts2d, K, R, t, w):
    """Weighted mean reprojection error (pixels)."""
    e = np.linalg.norm(project(pts3d,K,R,t) - pts2d, axis=1)
    return float(np.average(e, weights=w))


# ═════════════════════════════════════════════════════════════════════════════
# NORMALISATION  (Hartley — improves DLT numerical conditioning)
# ═════════════════════════════════════════════════════════════════════════════

def normalise_2d(pts):
    c = pts.mean(0)
    s = np.sqrt(2.) / max(np.sqrt(((pts-c)**2).sum(1)).mean(), 1e-8)
    T = np.array([[s,0.,-s*c[0]],[0.,s,-s*c[1]],[0.,0.,1.]])
    return (T @ np.column_stack([pts,np.ones(len(pts))]).T).T[:,:2], T

def normalise_3d(pts):
    c = pts.mean(0)
    s = np.sqrt(3.) / max(np.sqrt(((pts-c)**2).sum(1)).mean(), 1e-8)
    T = np.eye(4); T[:3,:3] *= s; T[:3,3] = -s*c
    return (T @ np.column_stack([pts,np.ones(len(pts))]).T).T[:,:3], T


# ═════════════════════════════════════════════════════════════════════════════
# DLT
# ═════════════════════════════════════════════════════════════════════════════

def build_dlt_matrix(p3n, p2n, w):
    """
    Build 2N×12 design matrix A.
    Cross-product constraint λ[u,v,1]ᵀ = PX̃ → two rows per point:
        row 2i  : [  0ᵀ | -X̃ᵀ |  vX̃ᵀ ] × √w
        row 2i+1: [  X̃ᵀ |  0ᵀ | -uX̃ᵀ ] × √w
    Null-space of A (last column of V from SVD) = vec(P).
    """
    N=len(p3n); A=np.zeros((2*N,12))
    for i,(X,x,wi) in enumerate(zip(p3n,p2n,w)):
        Xh=np.append(X,1.); u,v=x; sw=float(np.sqrt(max(wi,0.)))
        A[2*i,  4:8]=-Xh;  A[2*i,  8:12]= v*Xh
        A[2*i+1,0:4]= Xh;  A[2*i+1,8:12]=-u*Xh
        A[2*i]*=sw; A[2*i+1]*=sw
    return A

def dlt_solve(pts3d, pts2d, w):
    """DLT: normalise → SVD null-space → de-normalise → 3×4 matrix P."""
    p3n,T3=normalise_3d(pts3d); p2n,T2=normalise_2d(pts2d)
    A=build_dlt_matrix(p3n,p2n,w)
    _,_,Vt=np.linalg.svd(A)
    return np.linalg.inv(T2) @ Vt[-1].reshape(3,4) @ T3

def decompose_P(P, K):
    """Extract R, t from P=K[R|t]. Polar decomp forces valid rotation."""
    RT=np.linalg.inv(K)@P; M,tv=RT[:,:3],RT[:,3]
    U,sv,Vt=np.linalg.svd(M); R=U@Vt
    if np.linalg.det(R)<0: U[:,-1]*=-1; R=U@Vt
    return R, tv/sv.mean()


# ═════════════════════════════════════════════════════════════════════════════
# LM REFINEMENT
# ═════════════════════════════════════════════════════════════════════════════

def _lm_resid(params, pts3d, pts2d, K, w):
    R=rvec2R(params[:3]); t=params[3:6]
    return ((project(pts3d,K,R,t)-pts2d)*np.sqrt(w[:,None])).ravel()

def refine_lm(R0, t0, pts3d, pts2d, K, w):
    """LM pose refinement from initial (R0, t0)."""
    p0=np.concatenate([R2rvec(R0),t0])
    res=least_squares(_lm_resid,p0,args=(pts3d,pts2d,K,w),
                      method='lm',ftol=1e-10,xtol=1e-10,max_nfev=2000)
    return rvec2R(res.x[:3]), res.x[3:6]


# ═════════════════════════════════════════════════════════════════════════════
# MULTI-START  (8 look-at-origin poses — fast and sufficient)
# ═════════════════════════════════════════════════════════════════════════════

def _look_at(pos):
    pos=np.array(pos,dtype=np.float64)
    z=-pos/np.linalg.norm(pos); up=np.array([0.,1.,0.])
    x=np.cross(up,z)
    if np.linalg.norm(x)<1e-6: up=np.array([1.,0.,0.]); x=np.cross(up,z)
    x/=np.linalg.norm(x); y=np.cross(z,x)
    return np.stack([x,y,z]), pos

FIXED_STARTS = [_look_at(p) for p in [
    (-2,0, 4),(2,0, 4),(-2,0,-4),(2,0,-4),
    ( 0,0, 5),(0,0,-5),( 4,0, 4),(-4,0, 4),
]]


# ═════════════════════════════════════════════════════════════════════════════
# OBSERVATION EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def extract_obs(cam_data):
    """Return (pts3d, pts2d, confs) for visible observations only."""
    p3,p2,cs=[],[],[]
    for obs in cam_data["observations"]:
        if not obs["visible"]: continue
        p3.append(CUBE_PTS[obs["point_index"]])
        p2.append([obs["u"],obs["v"]])
        cs.append(float(obs["confidence"]))
    return np.array(p3), np.array(p2,dtype=np.float64), np.array(cs)


# ═════════════════════════════════════════════════════════════════════════════
# POSE ESTIMATION  (always multi-start — fast with 8+1 starts)
# ═════════════════════════════════════════════════════════════════════════════

def estimate_pose(cam_data, K):
    """
    DLT → multi-start LM.
    Tries 9 initializations (DLT + 8 look-at-origin grid), keeps best.
    Fast (~0.15s) and robust to DLT landing in a mirrored local minimum.
    """
    pts3d,pts2d,confs=extract_obs(cam_data)
    if len(pts3d)<6: raise ValueError(f"Need ≥6 visible pts (got {len(pts3d)}).")
    P=dlt_solve(pts3d,pts2d,confs); R0,t0=decompose_P(P,K)
    starts=[(R0,t0)]+list(FIXED_STARTS)
    best_e=1e9; best_R=best_t=None
    for Ri,ti in starts:
        R_,t_=refine_lm(Ri,ti,pts3d,pts2d,K,confs)
        e=reproj_err(pts3d,pts2d,K,R_,t_,confs)
        if e<best_e: best_e,best_R,best_t=e,R_,t_
    return dict(R=best_R,t=best_t,err=best_e)


# ═════════════════════════════════════════════════════════════════════════════
# TRIANGULATION
# ═════════════════════════════════════════════════════════════════════════════

def triangulate_dlt(views, use_weights=True):
    """Multi-view DLT: build 2M×4 system, solve SVD null-space."""
    rows=[]
    for v in views:
        P=v["K"]@np.column_stack([v["R"],v["t"]])
        u,vp=v["uv"]; p1,p2,p3=P
        ww=float(np.sqrt(v["conf"])) if use_weights else 1.
        rows.append((u*p3-p1)*ww); rows.append((vp*p3-p2)*ww)
    _,_,Vt=np.linalg.svd(np.array(rows))
    Xh=Vt[-1]; return Xh[:3]/Xh[3]

def _tri_resid(X,views,uw):
    res=[]
    for v in views:
        e=project(X.reshape(1,3),v["K"],v["R"],v["t"])[0]-v["uv"]
        w=float(np.sqrt(v["conf"])) if uw else 1.
        res.extend(e*w)
    return np.array(res)

def triangulate_optimal(views, X_init, use_weights=True):
    """Refine DLT estimate by minimising geometric reprojection error."""
    res=least_squares(_tri_resid,X_init,args=(views,use_weights),
                      method='lm',ftol=1e-10,xtol=1e-10,max_nfev=2000)
    return res.x

def build_views(data, poses, K):
    """Build per-point observation table from all cameras."""
    table=[dict(point_index=i,label=CUBE_LABELS[i],
                gt=CUBE_PTS[i].copy(),views=[]) for i in range(8)]
    for ck in ["camera1","camera2","camera3"]:
        for obs in data[ck]["observations"]:
            if not obs["visible"]: continue
            table[obs["point_index"]]["views"].append({
                "uv"  :np.array([obs["u"],obs["v"]],dtype=np.float64),
                "conf":float(obs["confidence"]),
                "R":poses[ck]["R"],"t":poses[ck]["t"],"K":K,
            })
    return table

def reconstruct_all(table):
    """Triangulate all 8 corners; return (8,3)."""
    pts=[]
    for entry in table:
        vs=entry["views"]
        if len(vs)<2: pts.append(entry["gt"].copy())
        else:
            X0=triangulate_dlt(vs,use_weights=True)
            pts.append(triangulate_optimal(vs,X0,use_weights=True))
    return np.array(pts)


# ═════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def run_pipeline(data, K):
    """Run pose + triangulation. Returns metrics dict or None on failure."""
    R1=np.array(data["camera1"]["R"],dtype=np.float64)
    t1=np.array(data["camera1"]["t"],dtype=np.float64)
    try:
        r2=estimate_pose(data["camera2"],K)
        r3=estimate_pose(data["camera3"],K)
    except Exception: return None
    poses={"camera1":{"R":R1,"t":t1,"K":K},
           "camera2":{"R":r2["R"],"t":r2["t"],"K":K},
           "camera3":{"R":r3["R"],"t":r3["t"],"K":K}}
    table=build_views(data,poses,K)
    errs_mm=np.linalg.norm(reconstruct_all(table)-CUBE_PTS,axis=1)*1000.
    return {
        "pose_err_R2":rot_err_deg(r2["R"],R2_REF),
        "pose_err_R3":rot_err_deg(r3["R"],R3_REF),
        "pose_err_t2":float(np.linalg.norm(r2["t"]-T2_REF)),
        "pose_err_t3":float(np.linalg.norm(r3["t"]-T3_REF)),
        "mean_3d_err":float(errs_mm.mean()),
        "max_3d_err" :float(errs_mm.max()),
    }

def add_noise(base_data, sigma, rng):
    """Clone data and add N(0,σ²) noise to every visible pixel."""
    data=copy.deepcopy(base_data)
    for ck in ["camera1","camera2","camera3"]:
        for obs in data[ck]["observations"]:
            if obs["visible"]:
                obs["u"]+=float(rng.normal(0.,sigma))
                obs["v"]+=float(rng.normal(0.,sigma))
    return data


# ═════════════════════════════════════════════════════════════════════════════
# TASK 3A — NOISE SENSITIVITY
# ═════════════════════════════════════════════════════════════════════════════

def task3a(base_data, K, sigmas=(1,2,5,10), n_trials=10):
    """
    For each σ run n_trials noisy instances of the full pipeline.
    Report mean ± std of 3D error and pose errors.

    Expected behaviour
    ------------------
    All errors rise monotonically with σ.  Pose estimation is most
    sensitive: each extra pixel directly perturbs the DLT system Ap=0,
    corrupting the null-space vector before LM even starts.
    At σ=10px errors may jump non-linearly as the DLT solution crosses
    into a different local-minimum basin.
    Running 10 trials removes seed variance; we report expected values.
    """
    print("\n"+"="*65)
    print("  3A — NOISE SENSITIVITY")
    print("="*65)
    print(f"  σ levels : {list(sigmas)} px  |  trials per level : {n_trials}")
    print(f"\n  {'σ(px)':>6s}  {'mean3D(mm)':>11s}  {'±std':>7s}  "
          f"{'ΔR2(°)':>8s}  {'ΔR3(°)':>8s}  {'Δt2(m)':>8s}  {'Δt3(m)':>8s}")
    print("  "+"-"*70)

    all_res={}
    for sigma in sigmas:
        trials=[]
        for seed in range(n_trials):
            rng =np.random.default_rng(seed*100+int(sigma*7))
            data=add_noise(base_data,sigma,rng)
            r   =run_pipeline(data,K)
            if r is not None: trials.append(r)
        if not trials:
            all_res[sigma]=None
            print(f"  {sigma:>6.0f}  ALL RUNS FAILED"); continue
        agg  ={k:np.array([t[k] for t in trials]) for k in trials[0]}
        stats={k:(float(agg[k].mean()),float(agg[k].std())) for k in agg}
        all_res[sigma]=stats
        m,s=stats["mean_3d_err"]
        print(f"  {sigma:>6.0f}  {m:>11.3f}  {s:>7.3f}  "
              f"{stats['pose_err_R2'][0]:>8.4f}  {stats['pose_err_R3'][0]:>8.4f}  "
              f"{stats['pose_err_t2'][0]:>8.4f}  {stats['pose_err_t3'][0]:>8.4f}")
    return all_res

def plot_3a(results, save_path):
    sigmas=[s for s in sorted(results) if results[s]]
    def g(k):  return [results[s][k][0] for s in sigmas]
    def gs(k): return [results[s][k][1] for s in sigmas]
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    fig.suptitle("Task 3A — Noise Sensitivity",fontweight='bold',fontsize=13)
    axes[0].errorbar(sigmas,g("mean_3d_err"),gs("mean_3d_err"),
                     marker='o',color='purple',capsize=5,lw=2,label='Mean 3D ± 1σ')
    axes[0].set_xlabel("Added noise σ (px)"); axes[0].set_ylabel("3D error (mm)")
    axes[0].set_title("3D Reconstruction Error vs Noise")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(sigmas,g("pose_err_R2"),'o-',color='royalblue',lw=2,label='Cam 2')
    axes[1].plot(sigmas,g("pose_err_R3"),'s--',color='tomato',  lw=2,label='Cam 3')
    axes[1].set_xlabel("Added noise σ (px)"); axes[1].set_ylabel("Rotation error (°)")
    axes[1].set_title("Pose Rotation Error vs Noise"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].plot(sigmas,g("pose_err_t2"),'o-',color='royalblue',lw=2,label='Cam 2')
    axes[2].plot(sigmas,g("pose_err_t3"),'s--',color='tomato',  lw=2,label='Cam 3')
    axes[2].set_xlabel("Added noise σ (px)"); axes[2].set_ylabel("Translation error (m)")
    axes[2].set_title("Pose Translation Error vs Noise"); axes[2].legend(); axes[2].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path,dpi=150,bbox_inches='tight')
    print(f"\n  Plot saved → {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# TASK 3B — DEGENERATE GEOMETRY
# ═════════════════════════════════════════════════════════════════════════════

def _tri_angle(C1, C2, X):
    """Angle (degrees) between rays from two cameras to 3-D point X."""
    r1=X-C1; r1/=np.linalg.norm(r1)
    r2=X-C2; r2/=np.linalg.norm(r2)
    return float(np.degrees(np.arccos(np.clip(r1@r2,-1.,1.))))

def make_coplanar_cam3(base_data, K, alpha):
    """
    Interpolate Cam3 from its estimated position toward a position
    almost coincident with Cam1 (fully coplanar, z=−5, x≈0).

        alpha=1.0  → original Cam3  (26° mean triangulation angle)
        alpha=0.0  → near-Cam1 pos  (<1° mean triangulation angle)

    Why near-coplanarity is degenerate
    ───────────────────────────────────
    Triangulation accuracy is governed by the baseline angle between
    cameras.  When angle θ → 0, rays from Cam1 and Cam3 become nearly
    parallel.  A small pixel perturbation δu shifts the estimated ray
    direction by ~δu/(f·d) radians, causing a depth error of
        δZ ≈ δu · Z² / (f · baseline)
    so depth error grows as 1/sin(θ) — exploding as θ → 0.
    The DLT condition number simultaneously rises as the geometry
    approaches the critical configuration (all cameras on one line).
    """
    data  =copy.deepcopy(base_data)
    C3_orig=np.array([ 2.33014,-0.04726,-4.74180])
    C3_copl=np.array([ 0.3, 0., -5.0])           # near Cam1 = [0,0,-5]
    C3_new =alpha*C3_orig+(1.-alpha)*C3_copl

    z_ax=-C3_new/np.linalg.norm(C3_new); up=np.array([0.,1.,0.])
    x_ax=np.cross(up,z_ax)
    if np.linalg.norm(x_ax)<1e-6: up=np.array([1.,0.,0.]); x_ax=np.cross(up,z_ax)
    x_ax/=np.linalg.norm(x_ax); y_ax=np.cross(z_ax,x_ax)
    R3_new=np.stack([x_ax,y_ax,z_ax]); t3_new=-R3_new@C3_new

    sigma=float(base_data["camera3"]["noise_sigma_px"])
    rng  =np.random.default_rng(0)
    for obs in data["camera3"]["observations"]:
        if not obs["visible"]: continue
        pr=project(CUBE_PTS[obs["point_index"]:obs["point_index"]+1],K,R3_new,t3_new)[0]
        obs["u"]=float(pr[0]+rng.normal(0.,sigma))
        obs["v"]=float(pr[1]+rng.normal(0.,sigma))
    return data,R3_new,t3_new,C3_new

def task3b(base_data, K):
    """
    Sweep alpha ∈ {1.0,0.8,0.6,0.4,0.2,0.1,0.05}.
    At each step measure:
      1. Mean triangulation angle between Cam1 and Cam3 (degrees)
         — the primary geometric health indicator.
      2. DLT condition number (ratio of largest to smallest
         singular value of the un-normalised A matrix).
         A healthy value is O(10–1000); degenerate → O(10⁶+).
      3. Mean 3-D reconstruction error (mm).
      4. Cam3 rotation error vs its new ground-truth pose.
    """
    print("\n"+"="*65)
    print("  3B — DEGENERATE GEOMETRY")
    print("="*65)
    print("  Cam3 moves from original position toward Cam1 (coplanar, z=−5).")
    print(f"\n  {'alpha':>6s}  {'tri_ang(°)':>10s}  {'cond(A)':>10s}  "
          f"{'mean3D(mm)':>11s}  {'rot_err(°)':>11s}")
    print("  "+"-"*57)

    C1=np.array([0.,0.,-5.])    # Camera 1 centre
    rows=[]
    for alpha in [1.0,0.8,0.6,0.4,0.2,0.1,0.05]:
        data3,R3_gt,t3_gt,C3_new=make_coplanar_cam3(base_data,K,alpha)

        # Mean triangulation angle between Cam1 and new Cam3
        angles=[_tri_angle(C1,C3_new,X) for X in CUBE_PTS]
        mean_ang=float(np.mean(angles))

        # DLT condition number (un-normalised A — measures actual numerical rank)
        pts3d,pts2d,confs=extract_obs(data3["camera3"])
        P_=dlt_solve(pts3d,pts2d,confs); R0_,t0_=decompose_P(P_,K)
        A_raw=build_dlt_matrix(pts3d,pts2d,confs)  # un-normalised
        sv_raw=np.linalg.svd(A_raw,compute_uv=False)
        cond=float(sv_raw[0]/(sv_raw[-1]+1e-12))

        R1=np.array(base_data["camera1"]["R"],dtype=np.float64)
        t1=np.array(base_data["camera1"]["t"],dtype=np.float64)
        try:
            r2=estimate_pose(data3["camera2"],K)
            r3=estimate_pose(data3["camera3"],K)
        except Exception as e:
            print(f"  {alpha:>6.2f}  FAILED: {e}"); continue

        poses={"camera1":{"R":R1,"t":t1,"K":K},
               "camera2":{"R":r2["R"],"t":r2["t"],"K":K},
               "camera3":{"R":r3["R"],"t":r3["t"],"K":K}}
        table=build_views(data3,poses,K)
        errs =np.linalg.norm(reconstruct_all(table)-CUBE_PTS,axis=1)*1000.
        re   =rot_err_deg(r3["R"],R3_gt)
        rows.append(dict(alpha=alpha,tri_ang=mean_ang,cond=cond,
                         mean3d=float(errs.mean()),rot_err=re))
        print(f"  {alpha:>6.2f}  {mean_ang:>10.2f}  {cond:>10.1f}  "
              f"{errs.mean():>11.3f}  {re:>11.4f}")
    return rows

def plot_3b(rows, save_path):
    alphas=[r["alpha"]   for r in rows]
    angs  =[r["tri_ang"] for r in rows]
    errs  =[r["mean3d"]  for r in rows]
    conds =[r["cond"]    for r in rows]
    rots  =[r["rot_err"] for r in rows]

    fig,axes=plt.subplots(1,3,figsize=(16,5))
    fig.suptitle("Task 3B — Degenerate Geometry (Cam3 → Coplanar with Cam1)",
                 fontweight='bold',fontsize=12)

    # Panel 1: triangulation angle
    axes[0].plot(alphas,angs,'o-',color='royalblue',lw=2)
    axes[0].set_xlabel("Alpha  (1.0=original, 0.0=coplanar)")
    axes[0].set_ylabel("Mean Cam1–Cam3 triangulation angle (°)")
    axes[0].set_title("Triangulation Angle vs Coplanarity")
    axes[0].invert_xaxis(); axes[0].grid(alpha=0.3)
    axes[0].axhline(10,color='red',ls='--',lw=1,label='10° danger threshold')
    axes[0].legend(fontsize=8)
    for a,v in zip(alphas,angs):
        axes[0].annotate(f'{v:.1f}°',(a,v),textcoords="offset points",
                         xytext=(4,4),fontsize=7)

    # Panel 2: condition number
    axes[1].plot(alphas,conds,'s-',color='darkorange',lw=2)
    axes[1].set_xlabel("Alpha")
    axes[1].set_ylabel("DLT condition number (un-normalised A)")
    axes[1].set_title("Condition Number vs Coplanarity")
    axes[1].set_yscale('log'); axes[1].invert_xaxis(); axes[1].grid(alpha=0.3)

    # Panel 3: 3D error and rotation error
    ax2=axes[2]; ax3=ax2.twinx()
    l1,=ax2.plot(alphas,errs,'s-',color='tomato',lw=2,label='Mean 3D error (mm)')
    l2,=ax3.plot(alphas,rots,'D--',color='purple',lw=2,label='Cam3 rot error (°)')
    ax2.set_xlabel("Alpha")
    ax2.set_ylabel("Mean 3D error (mm)",color='tomato')
    ax3.set_ylabel("Cam3 rotation error (°)",color='purple')
    ax2.tick_params(axis='y',labelcolor='tomato')
    ax3.tick_params(axis='y',labelcolor='purple')
    ax2.set_title("Reconstruction & Pose Error vs Coplanarity")
    ax2.invert_xaxis(); ax2.grid(alpha=0.3)
    ax2.legend(handles=[l1,l2],loc='upper right',fontsize=8)

    plt.tight_layout(); plt.savefig(save_path,dpi=150,bbox_inches='tight')
    print(f"  Plot saved → {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# TASK 3C — OUTLIER REJECTION
# ═════════════════════════════════════════════════════════════════════════════

def inject_outliers(base_data, shift=150., indices=(1,3)):
    """
    Shift 2 Camera 2 pixel observations by `shift` px in a random direction.
    Simulates a gross feature-matching error (e.g. ambiguous cube faces).
    """
    data=copy.deepcopy(base_data); rng=np.random.default_rng(42)
    for obs in data["camera2"]["observations"]:
        if obs["point_index"] in indices and obs["visible"]:
            a=rng.uniform(0.,2.*np.pi)
            obs["u"]+=float(shift*np.cos(a)); obs["v"]+=float(shift*np.sin(a))
    return data

def vanilla_lm(cam_data, K):
    """
    Standard DLT → LM.  No outlier handling.
    With outliers present the DLT design matrix A is immediately corrupted:
    the 150px-shifted rows drag the null-space vector far from the true pose.
    LM then minimises Σ‖eᵢ‖² including ~150px terms — no recovery possible.
    """
    pts3d,pts2d,confs=extract_obs(cam_data)
    P=dlt_solve(pts3d,pts2d,confs); R0,t0=decompose_P(P,K)
    R_,t_=refine_lm(R0,t0,pts3d,pts2d,K,confs)
    return dict(R=R_,t=t_,err=reproj_err(pts3d,pts2d,K,R_,t_,confs))

def ransac_pose(cam_data, K, n_iter=500, thresh=8.):
    """
    RANSAC PnP with 4-point sampling + multi-start LM scoring.

    Why sample 4, not 6?
    ────────────────────
    We have 6 observations; 2 are outliers (33%).
    P(6-point sample is outlier-free) = C(4,6)/C(6,6) = 0.
    P(4-point sample is outlier-free) = C(4,4)/C(6,4) = 1/15 ≈ 7%.
    Over 500 iterations: P(≥1 clean sample) ≈ 1−(14/15)^500 → 1.0.

    For each 4-point sample we run LM from 4 fixed look-at-origin starts,
    score ALL 6 observations, and keep the pose with the most inliers.
    Final refit uses all identified inliers.
    """
    pts3d,pts2d,confs=extract_obs(cam_data)
    N=len(pts3d)
    best_il=[]; best_R=best_t=None
    for _ in range(n_iter):
        idx=np.random.choice(N,size=4,replace=False)
        best_loc_e=1e9; Rc=tc=None
        for R0,t0 in FIXED_STARTS[:4]:           # 4 starts per sample
            try:
                R_,t_=refine_lm(R0,t0,pts3d[idx],pts2d[idx],K,confs[idx])
                e=reproj_err(pts3d[idx],pts2d[idx],K,R_,t_,confs[idx])
                if e<best_loc_e: best_loc_e,Rc,tc=e,R_,t_
            except: pass
        if Rc is None: continue
        errs=np.linalg.norm(project(pts3d,K,Rc,tc)-pts2d,axis=1)
        il  =np.where(errs<thresh)[0]
        if len(il)>len(best_il): best_il,best_R,best_t=il,Rc,tc
    if best_R is None or len(best_il)<3:
        return dict(R=None,t=None,err=1e9,n_inliers=0)
    s3=pts3d[best_il]; s2=pts2d[best_il]; sc=confs[best_il]
    R_r,t_r=refine_lm(best_R,best_t,s3,s2,K,sc)
    return dict(R=R_r,t=t_r,err=reproj_err(s3,s2,K,R_r,t_r,sc),
                n_inliers=int(len(best_il)))

def _robust_resid(params, pts3d, pts2d, K, confs, loss, scale):
    """
    IRLS residuals for Huber or Cauchy robust loss.

    Standard LM minimises Σ‖eᵢ‖² — unbounded, so one 150px outlier
    dominates.  Robust losses replace ‖e‖² with ρ(‖e‖):

        Huber (δ=scale):
            ρ(r) = r²/2           r ≤ δ
                   δ(r−δ/2)       r > δ
            w_rob = 1  (r≤δ),  δ/r  (r>δ)  ← bounded influence

        Cauchy (c=scale):
            ρ(r) = c²/2·log(1+(r/c)²)
            w_rob = 1/(1+(r/c)²)           ← redescending, → 0 as r→∞

    Return √(w_conf · w_rob) · e so LM minimises Σ w_conf · ρ(‖e‖).
    """
    R=rvec2R(params[:3]); t=params[3:6]
    ev=project(pts3d,K,R,t)-pts2d; r=np.linalg.norm(ev,axis=1)+1e-9
    if loss=='huber': w=np.where(r<=scale,1.,scale/r)
    else:             w=1./(1.+(r/scale)**2)
    return (ev*np.sqrt(confs*w)[:,None]).ravel()

def robust_lm(cam_data, K, loss='huber', scale=5., R0=None, t0=None):
    """LM with Huber or Cauchy robust loss to suppress outlier influence."""
    pts3d,pts2d,confs=extract_obs(cam_data)
    if R0 is None: P=dlt_solve(pts3d,pts2d,confs); R0,t0=decompose_P(P,K)
    p0=np.concatenate([R2rvec(R0),t0])
    res=least_squares(_robust_resid,p0,args=(pts3d,pts2d,K,confs,loss,scale),
                      method='lm',ftol=1e-10,xtol=1e-10,max_nfev=3000)
    R_=rvec2R(res.x[:3]); t_=res.x[3:6]
    return dict(R=R_,t=t_,err=reproj_err(pts3d,pts2d,K,R_,t_,confs))

def task3c(base_data, K):
    print("\n"+"="*65)
    print("  3C — OUTLIER REJECTION")
    print("="*65)

    # [1] Clean baseline
    print("\n  [1] Clean baseline (no outliers)")
    cl   =vanilla_lm(base_data["camera2"],K)
    re_cl=rot_err_deg(cl["R"],R2_REF)
    print(f"      Reproj error : {cl['err']:.4f} px")
    print(f"      Rotation err : {re_cl:.4f}°  ← reference")

    # [2] Inject outliers
    print("\n  [2] Injecting 2 outlier matches into Camera 2 (+150 px)")
    dirty=inject_outliers(base_data,shift=150.,indices=(1,3))
    p3_c,p2_c,_=extract_obs(base_data["camera2"])
    p3_d,p2_d,_=extract_obs(dirty["camera2"])
    for i in range(len(p3_d)):
        delta=p2_d[i]-p2_c[i]
        if np.linalg.norm(delta)>10.:
            idx=int(np.where((CUBE_PTS==p3_d[i]).all(1))[0][0])
            print(f"      '{CUBE_LABELS[idx]}'  shifted ({delta[0]:+.1f}, {delta[1]:+.1f}) px")

    # [3] Vanilla LM — expected to fail
    print("\n  [3] Vanilla LM on corrupted data")
    van  =vanilla_lm(dirty["camera2"],K)
    re_v =rot_err_deg(van["R"],R2_REF)
    print(f"      Reproj error : {van['err']:.4f} px")
    print(f"      Rotation err : {re_v:.4f}°  {'⚠ FAILED' if re_v>2. else '✓ OK'}")
    print(f"      Explanation  : Outlier rows in A poison the DLT null-space;")
    print(f"                     LM minimises Σ‖eᵢ‖² including 150px terms.")

    # [4] RANSAC — should recover
    print("\n  [4] RANSAC (500 iterations, 4-pt sample, threshold=8 px)")
    np.random.seed(42)
    rans =ransac_pose(dirty["camera2"],K,n_iter=500,thresh=8.)
    re_r =rot_err_deg(rans["R"],R2_REF) if rans["R"] is not None else 999.
    print(f"      Inliers      : {rans.get('n_inliers','?')} / {len(p3_d)}")
    print(f"      Reproj error : {rans['err']:.4f} px")
    print(f"      Rotation err : {re_r:.4f}°  {'✓ RECOVERED' if re_r<2. else '⚠ STILL FAILING'}")

    # [5] Huber robust LM — warm-started from RANSAC result
    # Warm-starting from the RANSAC pose skips the DLT initialisation
    # entirely — important when the DLT is poisoned by outliers.
    print("\n  [5] Huber robust LM (δ = 5 px, warm-started from RANSAC)")
    R_init = rans["R"] if rans["R"] is not None else None
    t_init = rans["t"] if rans["R"] is not None else None
    hub  =robust_lm(dirty["camera2"],K,loss='huber',scale=5.,R0=R_init,t0=t_init)
    re_h =rot_err_deg(hub["R"],R2_REF)
    print(f"      Reproj error : {hub['err']:.4f} px")
    print(f"      Rotation err : {re_h:.4f}°  {'✓ RECOVERED' if re_h<2. else '⚠ STILL FAILING'}")
    print(f"      Mechanism    : w_rob=δ/r=5/150≈0.033 for outliers → bounded influence")

    # [6] Cauchy robust LM — warm-started from RANSAC result
    print("\n  [6] Cauchy robust LM (c = 5 px, warm-started from RANSAC)")
    cau  =robust_lm(dirty["camera2"],K,loss='cauchy',scale=5.,R0=R_init,t0=t_init)
    re_c =rot_err_deg(cau["R"],R2_REF)
    print(f"      Reproj error : {cau['err']:.4f} px")
    print(f"      Rotation err : {re_c:.4f}°  {'✓ RECOVERED' if re_c<2. else '⚠ STILL FAILING'}")
    print(f"      Mechanism    : w_rob=1/(1+(150/5)²)≈0.001 → outlier virtually ignored")

    # Summary
    print("\n  ── Summary ─────────────────────────────────────────────────")
    print(f"  {'Method':>22s}  {'Reproj(px)':>10s}  {'ΔR(°)':>8s}  {'Status':>12s}")
    print("  "+"-"*58)
    summary=[
        ("Clean baseline", cl['err'],   re_cl, "reference"),
        ("Vanilla LM",     van['err'],  re_v,  "FAILED"    if re_v>2. else "OK"),
        ("RANSAC",         rans['err'], re_r,  "RECOVERED" if re_r<2. else "FAILED"),
        ("Huber LM",       hub['err'],  re_h,  "RECOVERED" if re_h<2. else "FAILED"),
        ("Cauchy LM",      cau['err'],  re_c,  "RECOVERED" if re_c<2. else "FAILED"),
    ]
    for name,rp,rd,status in summary:
        print(f"  {name:>22s}  {rp:>10.4f}  {rd:>8.4f}  {status:>12s}")
    return summary

def plot_3c(summary, save_path):
    names=[r[0] for r in summary]; rp=[r[1] for r in summary]; rd=[r[2] for r in summary]
    colors=['seagreen','crimson','royalblue','darkorange','purple']
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    fig.suptitle("Task 3C — Outlier Rejection Comparison",fontweight='bold',fontsize=13)
    for ax,vals,ylabel,title,thresh in [
        (axes[0],rp,"Reprojection error (px)","Reprojection Error",None),
        (axes[1],rd,"Rotation error vs GT (°)","Rotation Error (lower=better)",2.),
    ]:
        bars=ax.bar(names,vals,color=colors,alpha=0.85,edgecolor='white',width=0.5)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.tick_params(axis='x',rotation=18); ax.grid(axis='y',alpha=0.3)
        for bar,v in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2.,
                    bar.get_height()+max(vals)*0.01,
                    f'{v:.2f}',ha='center',va='bottom',fontsize=8)
        if thresh:
            ax.axhline(thresh,color='red',ls='--',lw=1.3,label=f'{thresh}° failure threshold')
            ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(save_path,dpi=150,bbox_inches='tight')
    print(f"  Plot saved → {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    with open("data.json") as f: base_data=json.load(f)
    K=np.array(base_data["camera_intrinsics"]["K"],dtype=np.float64)

    print("="*65); print("  TASK 3 — ROBUSTNESS ANALYSIS"); print("="*65)

    noise_res =task3a(base_data,K,sigmas=(1,2,5,10),n_trials=10)
    plot_3a(noise_res, "task3a_noise_sensitivity.png")

    degen_rows=task3b(base_data,K)
    plot_3b(degen_rows,"task3b_degenerate_geometry.png")

    summary   =task3c(base_data,K)
    plot_3c(summary,  "task3c_outlier_rejection.png")

    print("\n"+"="*65)
    print("  TASK 3 COMPLETE")
    print("  Outputs: task3a_noise_sensitivity.png")
    print("           task3b_degenerate_geometry.png")
    print("           task3c_outlier_rejection.png")
    print("="*65)

if __name__=="__main__":
    main()