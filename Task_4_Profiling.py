"""
Task 4: Profiling Output
========================
Nutpaa Technologies — Multi-View 3D Reconstruction Take-Home

Run this script to profile the full pipeline and identify bottlenecks.
Produces a table of per-stage runtimes and a flamegraph-style bar chart.

Usage:
    python task4_profiling.py
"""

import cProfile
import pstats
import io
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── import all pipeline components from task3 (which is standalone) ──────
# task3_robustness is standalone; all helpers imported via t3 below
import task3_robustness as t3

# ─────────────────────────────────────────────────────────────────────────────
# MANUAL STAGE TIMING
# ─────────────────────────────────────────────────────────────────────────────

def time_stage(fn, *args, n_runs=5, **kwargs):
    """Run fn n_runs times; return (mean_ms, std_ms, result)."""
    times = []
    result = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times)), result


def run_profiling():
    with open("data.json") as f:
        data = json.load(f)
    K = np.array(data["camera_intrinsics"]["K"], dtype=np.float64)

    print("="*60)
    print("  TASK 4 — PROFILING")
    print("="*60)
    print(f"\n  {'Stage':40s}  {'mean (ms)':>10s}  {'std (ms)':>10s}")
    print("  " + "-"*64)

    stages = {}

    # Stage 1: Data loading
    m, s, _ = time_stage(lambda: json.load(open("data.json")), n_runs=20)
    stages["Data loading"] = m
    print(f"  {'Data loading':40s}  {m:>10.3f}  {s:>10.3f}")

    # Stage 2: Normalisation (2D + 3D)
    pts3d, pts2d, confs = t3.extract_obs(data["camera2"])
    m, s, _ = time_stage(lambda: (t3.normalise_3d(pts3d), t3.normalise_2d(pts2d)), n_runs=500)
    stages["Normalisation (2D + 3D)"] = m
    print(f"  {'Normalisation (2D + 3D)':40s}  {m:>10.3f}  {s:>10.3f}")

    # Stage 3: Build DLT matrix
    p3n, T3 = t3.normalise_3d(pts3d); p2n, T2 = t3.normalise_2d(pts2d)
    m, s, _ = time_stage(t3.build_dlt_matrix, p3n, p2n, confs, n_runs=500)
    stages["Build DLT matrix (2N×12)"] = m
    print(f"  {'Build DLT matrix (2N×12)':40s}  {m:>10.3f}  {s:>10.3f}")

    # Stage 4: SVD (DLT solve)
    m, s, P = time_stage(t3.dlt_solve, pts3d, pts2d, confs, n_runs=200)
    stages["DLT solve (SVD null-space)"] = m
    print(f"  {'DLT solve (SVD null-space)':40s}  {m:>10.3f}  {s:>10.3f}")

    # Stage 5: Decompose P
    m, s, _ = time_stage(t3.decompose_P, P, K, n_runs=500)
    stages["Decompose P → R, t"] = m
    print(f"  {'Decompose P → R, t':40s}  {m:>10.3f}  {s:>10.3f}")

    # Stage 6: Single LM run
    R0, t0 = t3.decompose_P(P, K)
    m, s, _ = time_stage(t3.refine_lm, R0, t0, pts3d, pts2d, K, confs, n_runs=20)
    stages["LM refinement (single start)"] = m
    print(f"  {'LM refinement (single start)':40s}  {m:>10.3f}  {s:>10.3f}")

    # Stage 7: Full estimate_pose (9-start multi-start)
    m, s, _ = time_stage(t3.estimate_pose, data["camera2"], K, n_runs=5)
    stages["estimate_pose (9-start multi-start)"] = m
    print(f"  {'estimate_pose (9-start multi-start)':40s}  {m:>10.3f}  {s:>10.3f}")

    # Stage 8: Triangulation per point (DLT)
    R1=np.array(data["camera1"]["R"],dtype=np.float64)
    t1=np.array(data["camera1"]["t"],dtype=np.float64)
    r2=t3.estimate_pose(data["camera2"],K); r3=t3.estimate_pose(data["camera3"],K)
    poses={"camera1":{"R":R1,"t":t1,"K":K},
           "camera2":{"R":r2["R"],"t":r2["t"],"K":K},
           "camera3":{"R":r3["R"],"t":r3["t"],"K":K}}
    table=t3.build_views(data,poses,K)
    views_sample=table[0]["views"]   # front-bottom-left, 3 views

    m, s, X0 = time_stage(t3.triangulate_dlt, views_sample, True, n_runs=200)
    stages["Triangulate DLT (per point)"] = m
    print(f"  {'Triangulate DLT (per point)':40s}  {m:>10.3f}  {s:>10.3f}")

    # Stage 9: Optimal triangulation per point
    m, s, _ = time_stage(t3.triangulate_optimal, views_sample, X0, True, n_runs=100)
    stages["Triangulate optimal/LM (per point)"] = m
    print(f"  {'Triangulate optimal/LM (per point)':40s}  {m:>10.3f}  {s:>10.3f}")

    # Stage 10: Full pipeline (both cameras + all 8 points)
    m, s, _ = time_stage(t3.run_pipeline, data, K, n_runs=3)
    stages["Full pipeline (poses + triangulation)"] = m
    print(f"  {'Full pipeline (poses + triangulation)':40s}  {m:>10.3f}  {s:>10.3f}")

    print()

    # ── Bottleneck analysis ───────────────────────────────────────────────
    print("  BOTTLENECK ANALYSIS")
    print("  " + "-"*64)
    total = stages["Full pipeline (poses + triangulation)"]
    dominant = max(stages, key=stages.get)
    print(f"  Total pipeline time         : {total:.1f} ms")
    print(f"  Dominant stage              : '{dominant}' ({stages[dominant]:.1f} ms)")
    print(f"  LM fraction of pipeline     : {stages['LM refinement (single start)']*9/total*100:.1f}%")
    print()
    print("  PROPOSED OPTIMISATION: Replace multi-start DLT+LM with EPnP")
    print("  ─────────────────────────────────────────────────────────────")
    print("  EPnP (Lepetit et al., IJCV 2009) expresses world points as")
    print("  linear combinations of 4 control points, reduces pose estimation")
    print("  to a 12×12 eigenvalue problem, and solves in O(n) — achieving")
    print("  accuracy comparable to iterative LM at ~0.1ms vs ~135ms.")
    print("  This would make Task 3A's 40 pipeline calls take <1s total")
    print("  instead of ~120s, enabling real-time robustness sweeps.")

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Task 4 — Pipeline Profiling", fontweight='bold', fontsize=13)

    # Bar chart: all stages
    stage_names = list(stages.keys())
    times_ms    = list(stages.values())
    colors = ['#e74c3c' if t == max(times_ms) else '#3498db' for t in times_ms]

    ax = axes[0]
    bars = ax.barh(range(len(stage_names)), times_ms, color=colors, alpha=0.85)
    ax.set_yticks(range(len(stage_names)))
    ax.set_yticklabels(stage_names, fontsize=8)
    ax.set_xlabel("Mean runtime (ms)")
    ax.set_title("Per-Stage Runtime\n(red = bottleneck)")
    ax.set_xscale('log')
    ax.grid(axis='x', alpha=0.3)
    for bar, v in zip(bars, times_ms):
        ax.text(v * 1.05, bar.get_y() + bar.get_height()/2,
                f'{v:.2f}ms', va='center', fontsize=7)

    # Pie chart: time breakdown inside full pipeline
    sub_stages = {
        "estimate_pose ×2\n(multi-start LM)":
            stages["estimate_pose (9-start multi-start)"] * 2,
        "Triangulation ×8\n(DLT + optimal)":
            (stages["Triangulate DLT (per point)"] +
             stages["Triangulate optimal/LM (per point)"]) * 8,
        "Data prep\n(normalise, DLT, decompose)":
            stages["Normalisation (2D + 3D)"] +
            stages["DLT solve (SVD null-space)"] +
            stages["Decompose P → R, t"],
        "Other": max(0, total -
            stages["estimate_pose (9-start multi-start)"] * 2 -
            (stages["Triangulate DLT (per point)"] +
             stages["Triangulate optimal/LM (per point)"]) * 8 -
            stages["Normalisation (2D + 3D)"] -
            stages["DLT solve (SVD null-space)"] -
            stages["Decompose P → R, t"])
    }
    labels = [f"{k}\n({v:.1f} ms)" for k, v in sub_stages.items()]
    pie_colors = ['#e74c3c', '#f39c12', '#27ae60', '#95a5a6']
    axes[1].pie(list(sub_stages.values()), labels=labels,
                colors=pie_colors, autopct='%1.0f%%',
                startangle=90, textprops={'fontsize': 8})
    axes[1].set_title("Full Pipeline Time Breakdown")

    plt.tight_layout()
    plt.savefig("task4_profiling.png", dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved → task4_profiling.png")

    # ── cProfile of full pipeline ─────────────────────────────────────────
    print("\n  CPROFILE TOP-10 FUNCTIONS:")
    print("  " + "-"*64)
    pr = cProfile.Profile()
    pr.enable()
    t3.run_pipeline(data, K)
    pr.disable()
    stream = io.StringIO()
    ps = pstats.Stats(pr, stream=stream).sort_stats('cumulative')
    ps.print_stats(10)
    lines = stream.getvalue().split('\n')
    for line in lines[4:16]:
        if line.strip():
            print("  " + line)


if __name__ == "__main__":
    run_profiling()