"""Compare two full test-suite logs (before / after) cell by cell: AUC and runtime."""
import re, sys

def parse(path):
    L = open(path).read().splitlines()
    out = {}
    sec = None
    for line in L:
        if line.startswith("### "):
            sec = line[4:].strip()
            continue
        if not line.startswith("|") or sec is None:
            continue
        c = [x.strip().replace("**", "") for x in line.strip("|").split("|")]
        try:
            if sec.startswith("Fundamental Matrix (F) -- real") or sec.startswith("Essential Matrix (E) -- real") or sec.startswith("Homography (H) -- real"):
                prob = sec[0]
                feat, ds, n, meth = c[0], c[1], c[2], c[3]
                if ds != "Overall":
                    continue
                out[(prob, feat, meth)] = (float(c[5].rstrip("%")), float(c[8]))
            elif "7 matchers" in sec:
                prob = sec[0]
                m = c[0]
                if m in ("Matcher", "---") or m.startswith("Average") or m.startswith("**Average"):
                    continue
                pl = [float(x) for x in c[2].split(" / ")]; su = [float(x) for x in c[3].split(" / ")]
                tpl, tsu = [float(x) for x in c[6].split(" / ")]
                out[("det" + prob, m, "PoseLib")] = (pl[1], tpl)
                out[("det" + prob, m, "SupeRANSAC")] = (su[1], tsu)
            elif sec.startswith("Fundamental matrix -- classic") or sec.startswith("Homography -- classic"):
                prob = "clsF" if sec.startswith("Fundamental") else "clsH"
                ds, meth = c[0], c[2]
                if ds != "Overall":
                    continue
                out[(prob, ds, meth)] = (float(c[4].rstrip("%")), float(c[7]))
            elif sec.startswith("6D Absolute Pose (PnP) -- by dataset") or sec.startswith("6D Rigid Registration (SE(3)) -- by dataset"):
                prob = "PnP" if "Absolute" in sec else "Rigid"
                ds, meth = c[0], c[2]
                out[(prob, ds, meth)] = (float(c[4].rstrip("%")), float(c[8]))
        except (ValueError, IndexError):
            continue
    return out

a, b = parse(sys.argv[1]), parse(sys.argv[2])
keys = [k for k in a if k in b]
print(f"{'cell':44s} {'AUC before':>10s} {'AUC after':>10s} {'dAUC':>7s} {'ms before':>10s} {'ms after':>9s} {'speedup':>8s}")
for k in keys:
    (a1, t1), (a2, t2) = a[k], b[k]
    tag = " ".join(k)
    print(f"{tag[:44]:44s} {a1:10.2f} {a2:10.2f} {a2-a1:+7.2f} {t1:10.1f} {t2:9.1f} {t1/max(t2,1e-9):7.2f}x")

