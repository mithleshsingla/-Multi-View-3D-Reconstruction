"""
Task 4: Profiling — Actual Runtime Measurement
================================================
Nutpaa Technologies — Multi-View 3D Reconstruction Take-Home

Measures real wall-clock time for every stage of the pipeline
using timeit for micro-benchmarks and time.perf_counter for
end-to-end stages. Also runs cProfile to identify hotspot functions.

Usage:
    python task4_profiling.py
"""

import json, io, cProfile, pstats, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import task3_robustness as t3

# ─────────────────────────────────────────────────────────────────────────────
# TIMING HELPER
# ─────────────────────────────────────────────────────────────────────────────

def measure(fn, *args, n_runs=10, **kwargs):
    """
    Run fn(*args, **kwargs) n_runs times.
    Returns (mean_ms, std_ms, last_result).
    Uses time.perf_counter (highest-resolution clock available).
    Discards the first call (JIT warm-up / import side-effects).
    """
    result = fn(*args, **kwargs)          # warm-up call (discarded)
    times  = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        times.append((time.perf_counter() - t0) * 1e3)  # → ms
    return float(np.mean(times)), float(np.std(times)), result


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA ONCE
# ─────────────────────────────────────────────────────────────────────────────

with open("data.json") as f:
    BASE_DATA = json.load(f)

K = np.array(BASE_DATA["camera_intrinsics"]["K"], dtype=np.float64)
R1 = np.array(BASE_DATA["camera1"]["R"], dtype=np.float64)
t1 = np.array(BASE_DATA["camera1"]["t"], dtype=np.float64)

# Pre-extract observations so stage timings are isolated
pts3d_c2, pts2d_c2, confs_c2 = t3.extract_obs(BASE_DATA["camera2"])
pts3d_c3, pts2d_c3, confs_c3 = t3.extract_obs(BASE_DATA["camera3"])

# Pre-compute normalised arrays for DLT-matrix stage
p3n_c2, T3_c2 = t3.normalise_3d(pts3d_c2)
p2n_c2, T2_c2 = t3.normalise_2d(pts2d_c2)

# Pre-compute DLT matrix and P for downstream stages
A_c2    = t3.build_dlt_matrix(p3n_c2, p2n_c2, confs_c2)
P_c2    = t3.dlt_solve(pts3d_c2, pts2d_c2, confs_c2)
R0_c2, t0_c2 = t3.decompose_P(P_c2, K)

# Pre-compute full poses for triangulation stages
r2 = t3.estimate_pose(BASE_DATA["camera2"], K)
r3 = t3.estimate_pose(BASE_DATA["camera3"], K)
poses_full = {
    "camera1": {"R": R1,       "t": t1,       "K": K},
    "camera2": {"R": r2["R"],  "t": r2["t"],  "K": K},
    "camera3": {"R": r3["R"],  "t": r3["t"],  "K": K},
}
table_full = t3.build_views(BASE_DATA, poses_full, K)
views_3    = table_full[0]["views"]   # front-bottom-left — 3 cameras visible
views_2    = table_full[5]["views"]   # back-bottom-right — 2 cameras visible
X0_3       = t3.triangulate_dlt(views_3)
X0_2       = t3.triangulate_dlt(views_2)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE-BY-STAGE BENCHMARKS
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmarks():
    results = {}

    print("\n" + "="*70)
    print("  STAGE-BY-STAGE TIMING  (mean ± std over N repetitions)")
    print("="*70)
    print(f"  {'Stage':45s}  {'N':>4s}  {'mean(ms)':>9s}  {'std(ms)':>9s}")
    print("  " + "-"*70)

    def record(label, fn, *args, n=50, **kw):
        m, s, r = measure(fn, *args, n_runs=n, **kw)
        results[label] = dict(mean=m, std=s, n=n)
        print(f"  {label:45s}  {n:>4d}  {m:>9.4f}  {s:>9.4f}")
        return r

    # ── Data I/O ─────────────────────────────────────────────────────────
    record("Data loading (json.load)",
           lambda: json.load(open("data.json")), n=100)

    record("extract_obs (cam2, 6 visible pts)",
           t3.extract_obs, BASE_DATA["camera2"], n=1000)

    # ── Normalisation ─────────────────────────────────────────────────────
    record("normalise_3d (6 pts)",
           t3.normalise_3d, pts3d_c2, n=5000)

    record("normalise_2d (6 pts)",
           t3.normalise_2d, pts2d_c2, n=5000)

    # ── DLT ──────────────────────────────────────────────────────────────
    record("build_dlt_matrix (12×12 A)",
           t3.build_dlt_matrix, p3n_c2, p2n_c2, confs_c2, n=2000)

    record("np.linalg.svd  (12×12 A — DLT solve core)",
           lambda: np.linalg.svd(A_c2), n=2000)

    record("dlt_solve  (normalise + SVD + de-normalise)",
           t3.dlt_solve, pts3d_c2, pts2d_c2, confs_c2, n=500)

    record("decompose_P  (polar decomp → R, t)",
           t3.decompose_P, P_c2, K, n=2000)

    # ── LM refinement ────────────────────────────────────────────────────
    record("refine_lm  (single LM run, cam2)",
           t3.refine_lm, R0_c2, t0_c2, pts3d_c2, pts2d_c2, K, confs_c2, n=50)

    record("estimate_pose  (9 starts, cam2)",
           t3.estimate_pose, BASE_DATA["camera2"], K, n=10)

    record("estimate_pose  (9 starts, cam3 — harder)",
           t3.estimate_pose, BASE_DATA["camera3"], K, n=10)

    # ── Projection & reprojection ─────────────────────────────────────────
    record("project  (6 pts, 1 camera)",
           t3.project, pts3d_c2, K, r2["R"], r2["t"], n=10000)

    record("reproj_err  (6 pts, 1 camera)",
           t3.reproj_err, pts3d_c2, pts2d_c2, K, r2["R"], r2["t"], confs_c2,
           n=5000)

    # ── Triangulation ────────────────────────────────────────────────────
    record("triangulate_dlt   (3 views — front-BL)",
           t3.triangulate_dlt, views_3, True, n=500)

    record("triangulate_dlt   (2 views — back-BR)",
           t3.triangulate_dlt, views_2, True, n=500)

    record("triangulate_optimal  (3 views — front-BL)",
           t3.triangulate_optimal, views_3, X0_3, True, n=100)

    record("triangulate_optimal  (2 views — back-BR)",
           t3.triangulate_optimal, views_2, X0_2, True, n=100)

    record("reconstruct_all  (8 corners, all views)",
           t3.reconstruct_all, table_full, n=20)

    # ── Full pipeline ────────────────────────────────────────────────────
    record("run_pipeline  (pose est. + triangulation)",
           t3.run_pipeline, BASE_DATA, K, n=5)

    print()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# TASK-LEVEL TIMING  (3A, 3B, 3C as whole units)
# ─────────────────────────────────────────────────────────────────────────────

def time_tasks():
    print("="*70)
    print("  TASK-LEVEL TIMING  (single run each)")
    print("="*70)
    task_times = {}

    for label, fn, kwargs in [
        ("Task 3A (4σ × 10 trials)",
         t3.task3a,
         dict(base_data=BASE_DATA, K=K, sigmas=(1,2,5,10), n_trials=10)),
        ("Task 3B (7 alpha steps)",
         t3.task3b,
         dict(base_data=BASE_DATA, K=K)),
        ("Task 3C (outlier rejection, all methods)",
         t3.task3c,
         dict(base_data=BASE_DATA, K=K)),
    ]:
        t0 = time.perf_counter()
        fn(**kwargs)
        elapsed = (time.perf_counter() - t0)
        task_times[label] = elapsed
        print(f"  {label:45s}  {elapsed:>8.3f} s")

    print()
    return task_times


# ─────────────────────────────────────────────────────────────────────────────
# CPROFILE  — top hotspot functions
# ─────────────────────────────────────────────────────────────────────────────

def run_cprofile():
    print("="*70)
    print("  CPROFILE — TOP 15 CUMULATIVE HOTSPOTS  (one full pipeline run)")
    print("="*70)

    pr = cProfile.Profile()
    pr.enable()
    # Profile the most expensive realistic workload: 3A with 2 trials
    t3.task3a(BASE_DATA, K, sigmas=(2, 5), n_trials=2)
    pr.disable()

    stream = io.StringIO()
    ps    = pstats.Stats(pr, stream=stream).sort_stats("cumulative")
    ps.print_stats(15)
    lines = stream.getvalue().split("\n")

    # Print the header + data lines
    for line in lines:
        stripped = line.strip()
        if stripped:
            print("  " + line)

    print()


# ─────────────────────────────────────────────────────────────────────────────
# BOTTLENECK ANALYSIS & OPTIMISATION PROPOSAL
# ─────────────────────────────────────────────────────────────────────────────

def print_analysis(stage_results, task_times):
    print("="*70)
    print("  BOTTLENECK ANALYSIS & PROPOSED OPTIMISATION")
    print("="*70)

    # Find top-3 slowest stages
    sorted_stages = sorted(stage_results.items(),
                           key=lambda x: x[1]["mean"], reverse=True)
    print("\n  Top-3 slowest stages:")
    for i, (name, v) in enumerate(sorted_stages[:3], 1):
        print(f"    {i}. '{name}'  →  {v['mean']:.4f} ms")

    bottleneck, bv = sorted_stages[0]
    total_pipeline = stage_results.get("run_pipeline  (pose est. + triangulation)",
                                        {"mean": 1})["mean"]
    pct = bv["mean"] / total_pipeline * 100

    print(f"\n  Primary bottleneck : '{bottleneck}'")
    print(f"  Share of pipeline  : {pct:.1f}%")
    print(f"  Task 3A total time : {task_times.get('Task 3A (4σ × 10 trials)', 0):.2f} s")

    single_lm_ms = stage_results.get(
        "refine_lm  (single LM run, cam2)", {"mean": 0})["mean"]
    epnp_est_ms  = 0.1   # literature value for EPnP on 6 pts

    print(f"""
  PROPOSED OPTIMISATION: Replace multi-start DLT+LM with EPnP
  ─────────────────────────────────────────────────────────────
  Current  : 9 LM starts × {single_lm_ms:.2f} ms/start
           = {9*single_lm_ms:.2f} ms per camera pose
           = {2*9*single_lm_ms:.2f} ms for both cameras

  EPnP     : O(n) single eigenvalue solve of a 12×12 matrix
           ≈ {epnp_est_ms:.1f} ms per camera (Lepetit et al., IJCV 2009)
           = {2*epnp_est_ms:.1f} ms for both cameras

  Speedup  : {(9*single_lm_ms) / epnp_est_ms:.0f}× per camera

  Task 3A with EPnP: ~{task_times.get('Task 3A (4σ × 10 trials)',120) * epnp_est_ms/(9*single_lm_ms):.1f} s  (vs {task_times.get('Task 3A (4σ × 10 trials)',120):.1f} s measured)

  EPnP expresses the N world points as weighted sums of 4 virtual
  control points, transforms the projection equation to control-point
  form, and solves the resulting 12×12 linear system via eigendecomposition.
  It handles noise as well as iterative LM for n ≥ 6 points but is
  non-iterative — making it suitable for real-time and batch applications.
""")


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def make_plots(stage_results, task_times, save_path="task4_profiling.png"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Task 4 — Actual Pipeline Profiling",
                 fontweight='bold', fontsize=13)

    # ── Panel 1: per-stage horizontal bar (log scale) ────────────────────
    ax   = axes[0]
    names = list(stage_results.keys())
    means = [stage_results[n]["mean"] for n in names]
    stds  = [stage_results[n]["std"]  for n in names]
    colors = ['#e74c3c' if m == max(means) else
              '#e67e22' if m >= sorted(means)[-3] else
              '#3498db'
              for m in means]

    y_pos = range(len(names))
    ax.barh(y_pos, means, xerr=stds, color=colors, alpha=0.85,
            capsize=3, error_kw=dict(elinewidth=0.8))
    ax.set_yticks(y_pos)
    ax.set_yticklabels([n.split('(')[0].strip() for n in names], fontsize=7)
    ax.set_xlabel("Mean runtime (ms)  [log scale]")
    ax.set_title("Per-Stage Runtime\n(red = bottleneck, orange = top-3)")
    ax.set_xscale('log')
    ax.grid(axis='x', alpha=0.3)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m * 1.1, i, f'{m:.3f}', va='center', fontsize=6)

    # ── Panel 2: pie chart of full pipeline breakdown ────────────────────
    ax2 = axes[1]
    def ms(key):
        return stage_results.get(key, {"mean": 0})["mean"]

    lm_c2    = ms("estimate_pose  (9 starts, cam2)")
    lm_c3    = ms("estimate_pose  (9 starts, cam3 — harder)")
    tri_dlt  = ms("triangulate_dlt   (3 views — front-BL)") * 8
    tri_opt  = ms("triangulate_optimal  (3 views — front-BL)") * 8
    total_pipe = ms("run_pipeline  (pose est. + triangulation)")
    other    = max(0, total_pipe - lm_c2 - lm_c3 - tri_dlt - tri_opt)

    slices = {
        f"Cam2 pose\n({lm_c2:.1f} ms)": lm_c2,
        f"Cam3 pose\n({lm_c3:.1f} ms)": lm_c3,
        f"Tri DLT ×8\n({tri_dlt:.1f} ms)": tri_dlt,
        f"Tri OPT ×8\n({tri_opt:.1f} ms)": tri_opt,
        f"Other\n({other:.1f} ms)": max(other, 0.001),
    }
    pie_colors = ['#e74c3c','#e67e22','#3498db','#2ecc71','#95a5a6']
    wedges, texts, autotexts = ax2.pie(
        list(slices.values()),
        labels=list(slices.keys()),
        colors=pie_colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={"fontsize": 7},
    )
    ax2.set_title(f"Full Pipeline Breakdown\n(total: {total_pipe:.1f} ms)")

    # ── Panel 3: task-level bar chart ────────────────────────────────────
    ax3 = axes[2]
    task_names  = list(task_times.keys())
    task_secs   = list(task_times.values())
    task_colors = ['#9b59b6','#1abc9c','#e74c3c']
    bars = ax3.bar(range(len(task_names)), task_secs,
                   color=task_colors, alpha=0.85, width=0.5)
    ax3.set_xticks(range(len(task_names)))
    ax3.set_xticklabels([n.split('(')[0].strip() for n in task_names],
                         fontsize=9, rotation=10)
    ax3.set_ylabel("Wall-clock time (s)")
    ax3.set_title("End-to-End Task Runtimes")
    ax3.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, task_secs):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(task_secs)*0.01,
                 f'{v:.2f}s', ha='center', va='bottom', fontsize=9,
                 fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Plot saved → {save_path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*70)
    print("  TASK 4 — PIPELINE PROFILING  (actual measurements)")
    print("="*70)
    print("  All times measured with time.perf_counter.")
    print("  Each micro-benchmark discards 1 warm-up call, then averages N runs.")

    stage_results = run_benchmarks()
    task_times    = time_tasks()
    run_cprofile()
    print_analysis(stage_results, task_times)
    make_plots(stage_results, task_times, "task4_profiling.png")

    # ── Print final summary table ─────────────────────────────────────────
    print("="*70)
    print("  FINAL TIMING SUMMARY")
    print("="*70)
    print(f"  {'Stage / Task':50s}  {'Time':>12s}")
    print("  " + "-"*65)
    for name, v in stage_results.items():
        print(f"  {name:50s}  {v['mean']:>9.4f} ms")
    print()
    for name, v in task_times.items():
        print(f"  {name:50s}  {v:>9.3f}  s")
    print()


if __name__ == "__main__":
    main()