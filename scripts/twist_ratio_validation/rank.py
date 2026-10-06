"""Ranking items of the candidate comparison (AB_result.md) for RUN at K: rank.py RUN K [RUN K ...]"""
import json, sys, os
SPT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SPT)
from rules_walk import evaluate
args = sys.argv[1:]
for run, k in zip(args[::2], args[1::2]):
    nd, pd, pure, wn, wp, ch = evaluate(json.load(open(f"{SPT}/probes/w_{run}_{k}.json")))
    angles = [v["angle"] for v in nd["per"].values()] + [v["angle"] for v in pd["per"].values()]
    p = pure["per"]
    norm = {"F": (0, 0.2), "BK": (0, 0.2), "LL": (1, 0.1), "LR": (1, 0.1), "YL": (2, 0.5), "YR": (2, 0.5)}
    smin = min(p[c]["signed"][ax] / m for c, (ax, m) in norm.items())
    w25 = sum(ch[x] for x in ("W2", "W3", "W4", "W5"))
    print(f"{run} {k}: W1 {'ok' if ch['W1'] else 'NG'}, W2-W5 {w25}/4, none worst {wn:.1f}, p30 worst {wp:.1f}, "
          f"mean12 {sum(angles)/len(angles):.1f}, p30 falls {pd['falls']}, single min {smin:.2f}")
