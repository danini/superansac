"""Compact summary of a saved dev_ablate run: per problem/matcher, AUC delta vs base (paired bootstrap CI) and runtime.

Usage: python tests/dev_summary.py run.pkl [base_config_name]   (default base = first config)
"""
import os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation import pose_auc

AUCS = {"F": ([5.0, 10.0, 20.0], 1), "E": ([5.0, 10.0, 20.0], 1), "H": ([2.5, 5.0, 10.0], 1), "PnP": ([2.0, 5.0, 10.0], 1), "Rigid": ([2.0, 5.0, 10.0], 1)}


def boot(a, b, aucs, idx, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    d0 = 100 * (pose_auc(a, aucs)[idx] - pose_auc(b, aucs)[idx])
    ds = []
    for _ in range(n):
        s = rng.integers(0, len(a), len(a))
        ds.append(100 * (pose_auc(a[s], aucs)[idx] - pose_auc(b[s], aucs)[idx]))
    return d0, np.percentile(ds, 2.5), np.percentile(ds, 97.5)


d = pickle.load(open(sys.argv[1], "rb"))
names = list(d["configs"].keys())
base = sys.argv[2] if len(sys.argv) > 2 else names[0]  # optional: name of the reference config
for (prob, det), rows in d["results"].items():
    aucs, idx = AUCS[prob]
    errs = {c: np.array([r[c][0] for _, _, r in rows]) for c in names + ["PoseLib"]}
    ms = {c: np.mean([r[c][1] for _, _, r in rows]) for c in names + ["PoseLib"]}
    pl = 100 * pose_auc(errs["PoseLib"], aucs)[idx]
    print(f"\n[{prob} {det}] n={len(rows)}  PoseLib AUC {pl:.2f}  {ms['PoseLib']:.1f} ms")
    for c in names:
        a = 100 * pose_auc(errs[c], aucs)[idx]
        line = f"  {c:16s} AUC {a:6.2f} (vs PL {a - pl:+5.2f})  {ms[c]:7.1f} ms ({ms[c] / ms['PoseLib']:.2f}x PL)"
        if c != base:
            dd, lo, hi = boot(errs[c], errs[base], aucs, idx)
            line += f"  | vs {base}: {dd:+5.2f} [{lo:+5.2f},{hi:+5.2f}]  time {ms[c] / ms[base]:.2f}x"
        print(line)

