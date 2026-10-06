"""Append one AB_result.md row: append_one.py RUN K [NOTE]."""
import json, os, sys
SPT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SPT)
from rules_walk import evaluate
run, k = sys.argv[1], sys.argv[2]
note = sys.argv[3] if len(sys.argv) > 3 else ""
nd, pd, pure, wn, wp, checks = evaluate(json.load(open(f"{SPT}/probes/w_{run}_{k}.json")))
p = pure["per"]
single = ", ".join(f"{p[c]['signed'][ax]:.3f}" for c, ax in (("F", 0), ("BK", 0), ("LL", 1), ("LR", 1), ("YL", 2), ("YR", 2)))
ok = " ".join(f"{n}{'○' if v else '×'}" for n, v in checks.items())
row = (f"| {run} model_{k} | {wn:.0f} / {wp:.0f} 度 | {nd['speed']:.2f} / {pd['speed']:.2f} | "
       f"{nd['falls']}/90 / {pd['falls']}/90 | {single}{note} | {ok} |")
with open(f"{SPT}/AB_result.md", "a") as f:
    f.write(row + "\n")
print(row)
