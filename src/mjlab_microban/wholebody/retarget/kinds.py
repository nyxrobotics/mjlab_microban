"""CMU motion kinds (the training side takes a clip's leading letters as its kind), from the words of the
index's description (cmu-mocap-index-text.txt).  The first rule that matches wins, in this order: motions on
platforms, playground or seats, on the floor (lying, crawling, hands down), dances, sports and fighting,
jumps, runs, walks.  Anything else (gestures, acting) and clips without a description are misc.  Clip names
are cmu<kind>_<subject>_<take>.
"""
from __future__ import annotations

import re

RULES = [
    ("climb", r"\b(climb\w*|playground|ladder|stairs?|step ?stools?|stools?|uneven terrain|ledge|rock|boulder\w*|hang\w*)\b"),
    ("sit", r"\b(sit|sits|sitting|seated|chair|bench|couch)\b"),
    ("ground", (r"\b(lie|lies|lying|laying|lay|crawl\w*|roll\w*|floor|ground|get up|gets up|getting up|fall\w*|push ?ups?|sit ?ups?|"
                r"break ?danc\w*|cartwheel\w*|handstand\w*|kneel\w*|prone|supine)\b")),
    ("dance", r"\b(danc\w*|salsa|ballet|lindy\w*|hip ?hop|disco|charleston|waltz|tango|arabesque|pirouette|jive|lambada|swing dancing|indian)\b"),
    ("sport", (r"\b(boxing|boxer|punch\w*|kick\w*|fight\w*|martial|karate|sword\w*|kendo|tai chi|basketball|soccer|golf|baseball|football|"
               r"tennis|frisbee|volleyball|throw\w*|bowl\w*|dribbl\w*|shoot\w*|bat|racquet\w*|jab|strike|block\w*|swordplay)\b")),
    ("jump", r"\b(jump\w*|hop\w*|leap\w*|jacks|skip\w*)\b"),
    ("run", r"\b(run\w*|jog\w*|sprint\w*)\b"),
    ("walk", r"\b(walk\w*|march\w*|strid\w*|steps?|stepping|navigate|sneak\w*|stroll\w*|veer\w*|sidestep\w*|turn\w*|wander\w*)\b"),
]


def load_index(path):
    out = {}
    with open(path, encoding="utf-8", errors="replace") as stream:
        lines = stream.read().splitlines()
    for line in lines:
        m = re.match(r"^(\d+)_(\d+)\t(.*)$", line.rstrip("\n"))
        if m:
            out[f"{int(m.group(1)):02d}_{int(m.group(2)):02d}"] = m.group(3).strip()
    return out


def kind(desc: str) -> str:
    d = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", desc).lower()  # WalkStraight -> walk straight
    for name, pat in RULES:
        if re.search(pat, d):
            return name
    return "misc"
