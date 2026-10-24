"""Generate the README results tables from the JSON outputs in data/ (so numbers are never typed by hand)."""
import json, os, sys
import numpy as np

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
rel = json.load(open(os.path.join(D, "relative_pose_results.json")))
hom = json.load(open(os.path.join(D, "homography_results.json")))
det = json.load(open(os.path.join(D, "detector_results.json")))
cls = json.load(open(os.path.join(D, "classic_labelled_results.json")))
p6d = json.load(open(os.path.join(D, "pose_6d_results.json")))


def b(v, best):
    return f"**{v:.1f}**" if abs(v - best) < 1e-9 else f"{v:.1f}"


def rel_table(problem, idx=1):
    s = rel[problem]["summary"]
    out = ["| Matcher | Dataset | Pairs | PoseLib | SupeRANSAC | Δ | ms PoseLib / SupeRANSAC |", "|---|---|---:|---:|---:|---:|---:|"]
    for m in ("LoMa", "RoMa", "DISK+LG"):
        for ds in ("PhotoTourism", "ETH3D", "7Scenes", "Overall"):
            d = s[m][ds]; pl, su = d["PoseLib"]["auc"][idx], d["SupeRANSAC"]["auc"][idx]
            best = max(pl, su)
            name = "**Overall**" if ds == "Overall" else ds
            out.append(f"| {m} | {name} | {d['num_pairs']} | {b(pl, best)} | {b(su, best)} | {su-pl:+.1f} | {d['PoseLib']['time_ms']:.0f} / {d['SupeRANSAC']['time_ms']:.0f} |")
    return "\n".join(out)


def hom_table(idx=1):
    s = hom["summary"]
    out = ["| Matcher | Dataset | Pairs | PoseLib | SupeRANSAC | Δ | ms PoseLib / SupeRANSAC |", "|---|---|---:|---:|---:|---:|---:|"]
    for m in ("LoMa", "RoMa", "DISK+LG"):
        for ds in [k for k in s[m] if k != "Overall"] + ["Overall"]:
            d = s[m][ds]; pl, su = d["PoseLib"]["auc"][idx], d["SupeRANSAC"]["auc"][idx]
            best = max(pl, su)
            name = "**Overall**" if ds == "Overall" else ds
            out.append(f"| {m} | {name} | {d['num_pairs']} | {b(pl, best)} | {b(su, best)} | {su-pl:+.1f} | {d['PoseLib']['time_ms']:.0f} / {d['SupeRANSAC']['time_ms']:.0f} |")
    return "\n".join(out)


def det_table():
    out = ["| Matcher | $F$ AUC@10° PoseLib / SupeRANSAC | $E$ AUC@10° PoseLib / SupeRANSAC | $H$ AUC@5px PoseLib / SupeRANSAC |", "|---|---:|---:|---:|"]
    avg = {p: [[], []] for p in ("fundamental", "essential", "homography")}
    for m in ("LoMa", "RoMa", "LoFTR", "DISK+LG", "RootSIFT", "SIFT", "ORB"):
        cells = []
        for p in ("fundamental", "essential", "homography"):
            o = det[p][m]["Overall"]; pl, su = o["PoseLib"]["auc"][1], o["SupeRANSAC"]["auc"][1]
            avg[p][0].append(pl); avg[p][1].append(su)
            best = max(pl, su)
            cells.append(f"{b(pl, best)} / {b(su, best)} ({su-pl:+.1f})")
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    cells = []
    for p in ("fundamental", "essential", "homography"):
        pl, su = np.mean(avg[p][0]), np.mean(avg[p][1]); best = max(pl, su)
        cells.append(f"{b(pl, best)} / {b(su, best)} ({su-pl:+.1f})")
    out.append("| Average | " + " | ".join(cells) + " |")
    return "\n".join(out)


def det_ms():
    rows = []
    for m in ("LoMa", "RoMa", "LoFTR", "DISK+LG", "RootSIFT", "SIFT", "ORB"):
        f, e, h = (det[p][m]["Overall"] for p in ("fundamental", "essential", "homography"))
        rows.append(f"| {m} | {f['PoseLib']['time_ms']:.0f} / {f['SupeRANSAC']['time_ms']:.0f} | {e['PoseLib']['time_ms']:.0f} / {e['SupeRANSAC']['time_ms']:.0f} | {h['PoseLib']['time_ms']:.0f} / {h['SupeRANSAC']['time_ms']:.0f} |")
    return "\n".join(["| Matcher | $F$ ms PoseLib / SupeRANSAC | $E$ ms | $H$ ms |", "|---|---:|---:|---:|"] + rows)


def cls_table():
    out = ["| Problem | Scenes × repeats | Method | AUC@1px / 2px / 5px ($F$) — AUC@2.5 / 5 / 10 px ($H$) | Med err (px) | Mean ms |", "|---|---:|---|---|---:|---:|"]
    for p, lab in (("fundamental", "$F$"), ("homography", "$H$")):
        s = cls[p]
        for ds in [k for k in s if k != "Overall"] + ["Overall"]:
            d = s[ds]; name = f"{lab} **Overall**" if ds == "Overall" else f"{lab} {ds}"
            best = [max(d["PoseLib"]["auc"][i], d["SupeRANSAC"]["auc"][i]) for i in range(3)]
            for m in ("PoseLib", "SupeRANSAC"):
                a = d[m]["auc"]
                out.append(f"| {name} | {d['num_problems']} × {d['num_runs'] // d['num_problems']} | {m} | {b(a[0], best[0])} / {b(a[1], best[1])} / {b(a[2], best[2])} | {d[m]['med_err']:.2f} | {d[m]['time_ms']:.0f} |")
    return "\n".join(out)


def p6d_table():
    out = ["| Problem | Data | Method | AUC@(2°,4cm) | AUC@(5°,10cm) | AUC@(10°,20cm) | Med rot / trans | Mean ms |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for key, lab, methods in (("absolute_pose", "PnP", ("SupeRANSAC", "PoseLib", "OpenCV RANSAC")), ("rigid_transform", "Rigid", ("SupeRANSAC", "Procrustes RANSAC"))):
        s = p6d[key]["by_dataset"]
        for ds in [k for k in s if k != "Overall"]:
            d = s[ds]; best = [max(d[m]["auc"][i] for m in methods) for i in range(3)]
            name = ds.replace(" (synthetic outliers)", " synthetic") + (" synthetic" if ds == "rigid_pose_example" else "")
            for m in methods:
                a = d[m]["auc"]
                out.append(f"| {lab} | {name}, {d['num_problems']} | {m} | {b(a[0], best[0])} | {b(a[1], best[1])} | {b(a[2], best[2])} | {d[m]['med_rot']:.2f}° / {d[m]['med_trans']:.1f} cm | {d[m]['time_ms']:.0f} |")
    return "\n".join(out)


def summary_row_feh():
    out = []
    for p, lab, src in (("fundamental", "$F$", det), ("essential", "$E$", det)):
        for grp, ms in (("LoMa / RoMa / LoFTR / DISK+LG", ("LoMa", "RoMa", "LoFTR", "DISK+LG")), ("RootSIFT / SIFT / ORB", ("RootSIFT", "SIFT", "ORB"))):
            pl = [src[p][m]["Overall"]["PoseLib"]["auc"][1] for m in ms]; su = [src[p][m]["Overall"]["SupeRANSAC"]["auc"][1] for m in ms]
            tpl = [src[p][m]["Overall"]["PoseLib"]["time_ms"] for m in ms]; tsu = [src[p][m]["Overall"]["SupeRANSAC"]["time_ms"] for m in ms]
            out.append(f"| {lab}, {grp} | 300 calibrated pairs | " + " / ".join(f"{v:.1f}" for v in pl) + " (PoseLib) | " + " / ".join(f"{v:.1f}" for v in su) + " | " + " / ".join(f"{s-q:+.1f}" for s, q in zip(su, pl)) + " | " + ", ".join(f"{a:.0f} / {q:.0f}" for a, q in zip(tsu, tpl)) + " ms |")
    for grp, ms in (("LoMa / RoMa / LoFTR / DISK+LG", ("LoMa", "RoMa", "LoFTR", "DISK+LG")), ("RootSIFT / SIFT / ORB", ("RootSIFT", "SIFT", "ORB"))):
        pl = [det["homography"][m]["Overall"]["PoseLib"]["auc"][1] for m in ms]; su = [det["homography"][m]["Overall"]["SupeRANSAC"]["auc"][1] for m in ms]
        tpl = [det["homography"][m]["Overall"]["PoseLib"]["time_ms"] for m in ms]; tsu = [det["homography"][m]["Overall"]["SupeRANSAC"]["time_ms"] for m in ms]
        out.append(f"| $H$, {grp} | 16 Homogr + EVD pairs | " + " / ".join(f"{v:.1f}" for v in pl) + " (PoseLib) | " + " / ".join(f"{v:.1f}" for v in su) + " | " + " / ".join(f"{s-q:+.1f}" for s, q in zip(su, pl)) + " | " + ", ".join(f"{a:.0f} / {q:.0f}" for a, q in zip(tsu, tpl)) + " ms |")
    f = cls["fundamental"]["Overall"]; h = cls["homography"]["Overall"]
    out.append(f"| $F$, labelled tentative matches | 30 kusvod2 + AdelaideRMF scenes × 5 | {f['PoseLib']['auc'][1]:.1f} @2px (PoseLib) | {f['SupeRANSAC']['auc'][1]:.1f} | {f['SupeRANSAC']['auc'][1]-f['PoseLib']['auc'][1]:+.1f} | {f['SupeRANSAC']['time_ms']:.0f} / {f['PoseLib']['time_ms']:.0f} ms |")
    out.append(f"| $H$, labelled tentative matches | 35 Homogr + EVD + AdelaideRMF scenes × 5 | {h['PoseLib']['auc'][1]:.1f} @5px (PoseLib) | {h['SupeRANSAC']['auc'][1]:.1f} | {h['SupeRANSAC']['auc'][1]-h['PoseLib']['auc'][1]:+.1f} | {h['SupeRANSAC']['time_ms']:.0f} / {h['PoseLib']['time_ms']:.0f} ms |")
    pn = p6d["absolute_pose"]["by_dataset"]["7Scenes RGB-D"]; rg = p6d["rigid_transform"]["by_dataset"]["7Scenes RGB-D"]
    out.append(f"| PnP, 6 matchers | {pn['num_problems']} real 7Scenes RGB-D problems | {pn['PoseLib']['auc'][1]:.1f} (PoseLib), {pn['OpenCV RANSAC']['auc'][1]:.1f} (OpenCV) | {pn['SupeRANSAC']['auc'][1]:.1f} | {pn['SupeRANSAC']['auc'][1]-pn['PoseLib']['auc'][1]:+.1f} / {pn['SupeRANSAC']['auc'][1]-pn['OpenCV RANSAC']['auc'][1]:+.1f} | {pn['SupeRANSAC']['time_ms']:.0f} / {pn['PoseLib']['time_ms']:.0f} / {pn['OpenCV RANSAC']['time_ms']:.0f} ms |")
    out.append(f"| Rigid, 6 matchers | {rg['num_problems']} real 7Scenes RGB-D problems | {rg['Procrustes RANSAC']['auc'][1]:.1f} (Procrustes RANSAC) | {rg['SupeRANSAC']['auc'][1]:.1f} | {rg['SupeRANSAC']['auc'][1]-rg['Procrustes RANSAC']['auc'][1]:+.1f} | {rg['SupeRANSAC']['time_ms']:.0f} / {rg['Procrustes RANSAC']['time_ms']:.0f} ms |")
    return "\n".join(["| Task | Data | Baseline | SupeRANSAC | Δ | Runtime SupeRANSAC / baseline |", "|---|---|---:|---:|---:|---:|"] + out)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    sections = {
        "summary": summary_row_feh, "F": lambda: rel_table("fundamental"), "E": lambda: rel_table("essential"),
        "H": hom_table, "det": det_table, "det_ms": det_ms, "classic": cls_table, "p6d": p6d_table,
    }
    for k, fn in sections.items():
        if which in ("all", k):
            print(f"\n<!-- {k} -->\n{fn()}\n")
