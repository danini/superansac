"""Per-pair diagnostic: where does SupeRANSAC lose to PoseLib? (dev split)"""
import os, sys, json
from concurrent.futures import ProcessPoolExecutor
import numpy as np

TESTS = "/usr/local/google/home/dbarath/ransac/superansac/tests"
sys.path.insert(0, TESTS)
os.environ.setdefault("OMP_NUM_THREADS", "1")
import dev_ablate as da
from datasets.benchmark_pairs import DEFAULT_DATA_DIR, canonical_detector_name, ensure_cached_matches, load_calibrated_pairs

problem = sys.argv[1]          # F or E
det = canonical_detector_name(sys.argv[2])
cfg = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {"final_bundle_rounds": 2}
pairs = load_calibrated_pairs(DEFAULT_DATA_DIR, split="dev", verbose=False)
cache = ensure_cached_matches(pairs, [det], data_dir=DEFAULT_DATA_DIR)
fn = da._f if problem == "F" else da._e
tasks = [(p, cache[p["cache_key"]][det][0], cache[p["cache_key"]][det][1], {"su": cfg}) for p in pairs]
with ProcessPoolExecutor(max_workers=24) as ex:
    res = list(ex.map(fn, tasks))
rows = []
for p, (ds, r) in zip(pairs, res):
    n = len(cache[p["cache_key"]][det][0])
    rows.append((ds, p["scene"][:40], n, r["su"][0], r["PoseLib"][0]))
rows.sort(key=lambda x: -(x[3] - x[4]))
print(f"{'dataset':14s} {'scene':40s} {'N':>5s} {'SU':>8s} {'PL':>8s} {'diff':>8s}")
for ds, sc, n, su, pl in rows[:15]:
    print(f"{ds:14s} {sc:40s} {n:5d} {su:8.2f} {pl:8.2f} {su-pl:+8.2f}")
print("...")
for ds, sc, n, su, pl in rows[-8:]:
    print(f"{ds:14s} {sc:40s} {n:5d} {su:8.2f} {pl:8.2f} {su-pl:+8.2f}")
d = np.array([r[3] - r[4] for r in rows])
both_ok = np.array([(r[3] < 10 and r[4] < 10) for r in rows])
print(f"\nmedian diff (both < 10 deg): {np.median(d[both_ok]):+.3f} deg over {both_ok.sum()} pairs; "
      f"SU fails(>10) where PL ok: {int(np.sum([(r[3] >= 10 and r[4] < 10) for r in rows]))}; "
      f"PL fails where SU ok: {int(np.sum([(r[4] >= 10 and r[3] < 10) for r in rows]))}")
for ds in sorted({r[0] for r in rows}):
    sel = [r for r in rows if r[0] == ds]
    dd = np.array([r[3] - r[4] for r in sel]); ok = np.array([(r[3] < 10 and r[4] < 10) for r in sel])
    print(f"  {ds:14s} median diff (both ok) {np.median(dd[ok]) if ok.any() else float('nan'):+.3f}  n_ok={ok.sum()}  "
          f"SU-only-fail={int(np.sum([(r[3]>=10 and r[4]<10) for r in sel]))} PL-only-fail={int(np.sum([(r[4]>=10 and r[3]<10) for r in sel]))}")

