"""Apply the pre-declared twist-ratio validation rules (status.txt) to a vprobe_tr.py JSON.

usage: rules_tr.py PROBE.json STAGE   (STAGE: 8000 = early warning, 9500 = validation)
Prints one summary line, the per-cell table and PASS/FAIL per rule; exit 0 iff all rules hold.
"""
import json
import math
import sys

SCALE = (0.7, 0.3, 1.5)
DIAG = ["A", "M", "B", "MB", "K", "MK"]
PURE_MIN = {"F": (0, 0.08), "BK": (0, 0.04), "LL": (1, 0.02), "LR": (1, 0.02), "YL": (2, 0.20), "YR": (2, 0.20)}
AXIS_MIN = (0.03, 0.02, 0.15)  # signed minimum per commanded axis (m/s, m/s, rad/s)


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


def cell(rows, push, targets, cmds):
    sel = [r for r in rows if r["push"] == push and r["targets"] == targets and r["cmd"] in cmds]
    per = {}
    for c in cmds:
        rr = [r for r in sel if r["cmd"] == c]
        if not rr:
            continue
        twist = rr[0]["twist"]
        m = [mean([r["mean"][i] for r in rr]) for i in range(3)]
        signed = [m[i] * (1 if twist[i] > 0 else -1 if twist[i] < 0 else 0) for i in range(3)]
        per[c] = {"n": len(rr), "falls": sum(r["fell"] for r in rr), "mean": m, "signed": signed,
                  "angle_of_mean": angle(twist, m), "speed": mean([r["speed"] for r in rr if r["speed"] is not None]),
                  "angle_rollouts": mean([r["angle_deg"] for r in rr if r["angle_deg"] is not None])}
    return {"n": len(sel), "falls": sum(r["fell"] for r in sel),
            "speed": mean([r["speed"] for r in sel if r["speed"] is not None]), "per": per}


def main():
    data = json.load(open(sys.argv[1]))
    stage = int(sys.argv[2])
    rows = data["rows"]
    pushes = {r["push"] for r in rows}
    nf = cell(rows, "none", "full", DIAG)
    nh = cell(rows, "none", "hands_off", DIAG)
    pf = cell(rows, "p30_15", "full", DIAG) if "p30_15" in pushes else None
    pure = cell(rows, "none", "off", list(PURE_MIN) + ["S"])
    checks = {}
    final = stage >= 9500
    # V1 ratio (validation only; reported at 8000)
    worst_nf = max(v["angle_of_mean"] for v in nf["per"].values())
    worst_pf = max(v["angle_of_mean"] for v in pf["per"].values()) if pf else float("nan")
    if final:
        checks["V1 ratio none/full every cmd angle<=20deg"] = worst_nf <= 20.0
        checks["V1 ratio p30/full every cmd angle<=30deg"] = pf is not None and worst_pf <= 30.0
    # V2 sign-correct on every commanded axis (validation only)
    def signs_ok(c):
        return all(all(v["signed"][i] >= AXIS_MIN[i] for i in range(3)) for v in c["per"].values())
    if final:
        checks["V2 sign none/full every cmd every axis"] = signs_ok(nf)
        checks["V2 sign p30/full every cmd every axis"] = pf is not None and signs_ok(pf)
    # V3 speed fraction
    checks[f"V3 speed none/full >= {0.20 if final else 0.12}"] = nf["speed"] >= (0.20 if final else 0.12)
    if final:
        checks["V3 speed p30/full >= 0.15"] = pf is not None and pf["speed"] >= 0.15
    # V4 falls
    checks["V4 falls none/full <= 2"] = nf["falls"] <= 2
    checks["V4 falls none/hands_off <= 2"] = nh["falls"] <= 2
    checks["V4 falls p30/full <= 9"] = pf is not None and pf["falls"] <= 9
    # V5 single axis and standing
    ok = pure["falls"] == 0
    for c, (axis, lo) in PURE_MIN.items():
        ok = ok and pure["per"][c]["signed"][axis] >= lo
    s = pure["per"]["S"]["mean"]
    ok = ok and abs(s[0]) <= 0.05 and abs(s[1]) <= 0.05 and abs(s[2]) <= 0.2
    checks["V5 single-axis minimums + standing + no falls"] = ok

    def fmt(c):
        return " ".join(
            f"{k}[{v['mean'][0]:+.3f},{v['mean'][1]:+.3f},{v['mean'][2]:+.2f} ang{v['angle_of_mean']:.0f} s{v['speed']:.2f} f{v['falls']}]"
            for k, v in c["per"].items())
    print(f"iteration {data['iteration']} stage {stage}")
    print(f"none/full   n={nf['n']} falls={nf['falls']} speed={nf['speed']:.3f} worst_angle={worst_nf:.1f}  {fmt(nf)}")
    print(f"none/hoff   n={nh['n']} falls={nh['falls']} speed={nh['speed']:.3f}  {fmt(nh)}")
    if pf:
        print(f"p30/full    n={pf['n']} falls={pf['falls']} speed={pf['speed']:.3f} worst_angle={worst_pf:.1f}  {fmt(pf)}")
    print(f"single-axis n={pure['n']} falls={pure['falls']}  {fmt(pure)}")
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    allok = all(checks.values())
    print("OVERALL", "PASS" if allok else "FAIL")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
