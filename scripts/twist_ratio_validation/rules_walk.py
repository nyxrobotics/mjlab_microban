"""Apply the walking checkpoint rules W1-W5 (walk_status.txt, fixed before training) to a vprobe_walk.py JSON.

usage: rules_walk.py PROBE.json [--line]
Prints the per-command table and PASS/FAIL per rule (or one summary line with --line);
exit 0 iff W1-W5 all hold.
"""
import json
import math
import sys

SCALE = (0.7, 0.3, 1.5)
DIAG = ["A", "M", "B", "MB", "K", "MK"]
PURE_MIN = {"F": (0, 0.08), "BK": (0, 0.04), "LL": (1, 0.02), "LR": (1, 0.02), "YL": (2, 0.20), "YR": (2, 0.20)}
AXIS_MIN = (0.03, 0.02, 0.15)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def angle(cmd, v):
    c = [cmd[i] / SCALE[i] for i in range(3)]
    w = [v[i] / SCALE[i] for i in range(3)]
    n = math.sqrt(sum(x * x for x in c))
    vn = math.sqrt(sum(x * x for x in w))
    if n == 0.0:
        return float("nan")
    if vn < 1e-9:
        return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0, sum(c[i] * w[i] for i in range(3)) / (n * vn)))))


def cell(rows, push, cmds):
    sel = [r for r in rows if r["push"] == push and r["cmd"] in cmds]
    per = {}
    for c in cmds:
        rr = [r for r in sel if r["cmd"] == c]
        if not rr:
            continue
        twist = rr[0]["twist"]
        m = [mean([r["mean"][i] for r in rr]) for i in range(3)]
        signed = [m[i] * (1 if twist[i] > 0 else -1 if twist[i] < 0 else 0) for i in range(3)]
        per[c] = {"n": len(rr), "falls": sum(r["fell"] for r in rr), "mean": m, "signed": signed,
                  "angle": angle(twist, m), "speed": mean([r["speed"] for r in rr if r["speed"] is not None])}
    return {"n": len(sel), "falls": sum(r["fell"] for r in sel),
            "speed": mean([r["speed"] for r in sel if r["speed"] is not None]), "per": per}


def evaluate(data):
    rows = data["rows"]
    nd = cell(rows, "none", DIAG)
    pd = cell(rows, "p30_15", DIAG)
    pure = cell(rows, "none", list(PURE_MIN) + ["S"])
    wn = max(v["angle"] for v in nd["per"].values())
    wp = max(v["angle"] for v in pd["per"].values())

    def signs(c):
        return all(all(v["signed"][i] >= AXIS_MIN[i] for i in range(3)) for v in c["per"].values())

    s = pure["per"]["S"]["mean"]
    w5 = all(pure["per"][c]["signed"][ax] >= lo for c, (ax, lo) in PURE_MIN.items()) and (
        abs(s[0]) <= 0.05 and abs(s[1]) <= 0.05 and abs(s[2]) <= 0.2)
    checks = {
        "W1": wn <= 20.0 and wp <= 30.0,
        "W2": signs(nd) and signs(pd),
        "W3": nd["speed"] >= 0.25 and pd["speed"] >= 0.20,
        "W4": nd["falls"] <= 1 and pd["falls"] <= 5 and pure["falls"] == 0,
        "W5": w5,
    }
    return nd, pd, pure, wn, wp, checks


def main():
    data = json.load(open(sys.argv[1]))
    nd, pd, pure, wn, wp, checks = evaluate(data)
    ok = all(checks.values())
    tag = " ".join(f"{k}={'ok' if v else 'NG'}" for k, v in checks.items())
    line = (f"it {data['iteration']}: {'PASS' if ok else 'FAIL'} [{tag}] angle none {wn:.0f}/p30 {wp:.0f} deg; "
            f"speed none {nd['speed']:.2f}/p30 {pd['speed']:.2f}; falls none {nd['falls']}/90 p30 {pd['falls']}/90 "
            f"single {pure['falls']}; F {pure['per']['F']['signed'][0]:+.3f} BK {pure['per']['BK']['signed'][0]:+.3f} "
            f"LL {pure['per']['LL']['signed'][1]:+.3f} LR {pure['per']['LR']['signed'][1]:+.3f} "
            f"YL {pure['per']['YL']['signed'][2]:+.2f} YR {pure['per']['YR']['signed'][2]:+.2f}")
    if "--line" in sys.argv:
        print(line)
    else:
        def fmt(c):
            return " ".join(
                f"{k}[{v['mean'][0]:+.3f},{v['mean'][1]:+.3f},{v['mean'][2]:+.2f} ang{v['angle']:.0f} s{v['speed']:.2f} f{v['falls']}]"
                for k, v in c["per"].items())
        print(line)
        print("none  ", fmt(nd))
        print("p30   ", fmt(pd))
        print("single", fmt(pure))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
