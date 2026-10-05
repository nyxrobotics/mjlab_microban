"""Rewrite the robot repository's run- and HOME-specific pins (stdlib only).

Used by scripts/retrain_all_for_home.py after new policies are installed in
the robot repository (microban):

* run-specific pins: the frozen walking source of PICO v12
  (``EXPECTED_V12_LEGACY_SOURCE_CHECKPOINT_SHA256`` / ``_ITERATION`` /
  ``EXPECTED_V12_LEGACY_PROBE_SHA256`` in src/moves/pico_hybrid.py) and the
  walk fallback (``EXPECTED_WALK_FALLBACK_SHA256`` in
  tools/validate_pico_policy.py);
* the HOME pins the robot tests keep on purpose: the reviewed degree table
  ``TRAINING_HOME_DEG`` (tests/test_shared_home.py) is regenerated and
  ``PACKAGER_V12_HOME_POSE_JSON`` (tests/test_pico_hybrid.py) is set from the
  packaged ONNX.  Every other HOME-bound value of the robot tests comes from
  config/home_pose.yaml (robot home-config), so a branch of any HOME needs no
  other test edit.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping

PICO_HYBRID = "src/moves/pico_hybrid.py"
VALIDATOR = "tools/validate_pico_policy.py"

# ---------------------------------------------------------------- robot yaml
def parse_robot_home_yaml(text: str) -> dict:
    """Parse the strict YAML subset of the robot's config/home_pose.yaml."""

    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        key, _, value = raw.strip().partition(":")
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        value = value.strip()
        if value == "":
            child: dict = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = json.loads(value)
    return root


def signed_degree_token(value_deg: float) -> str:
    """Same as mjlab_microban.robot.home_pose.signed_degree_token."""

    sign = "minus" if value_deg < 0 else "plus"
    text = repr(abs(float(value_deg)))
    if text.endswith(".0"):
        text = text[:-2]
    return sign + text.replace(".", "p")


def set_training_home_deg(text: str, joint_pos_deg: Mapping[str, float]) -> tuple[str, bool]:
    """Regenerate the TRAINING_HOME_DEG table of tests/test_shared_home.py."""

    match = re.search(r"^TRAINING_HOME_DEG = \{\n(.*?)\n\}", text, flags=re.M | re.S)
    if match is None:
        return text, False
    names = re.findall(r'^\s*"([a-z_]+)":', match.group(1), flags=re.M)
    if set(names) != set(joint_pos_deg):
        names = list(joint_pos_deg)
    body = "\n".join(f'    "{name}": {float(joint_pos_deg[name])!r},' for name in names)
    new = text[: match.start(1)] + body + text[match.end(1):]
    return new, new != text


def set_packager_home_json(text: str, home_json: str, width: int = 72) -> tuple[str, bool]:
    """Set PACKAGER_V12_HOME_POSE_JSON (tests/test_pico_hybrid.py) to the ONNX value."""

    match = re.search(r"^PACKAGER_V12_HOME_POSE_JSON = \(\n(.*?)\n\)", text, flags=re.M | re.S)
    if match is None:
        return text, False
    try:
        if ast.literal_eval("(\n" + match.group(1) + "\n)") == home_json:
            return text, False
    except (SyntaxError, ValueError):
        pass
    chunks, current = [], ""
    for piece in re.split(r"(?<=,)", home_json):
        if current and len(current) + len(piece) > width:
            chunks.append(current)
            current = ""
        current += piece
    if current:
        chunks.append(current)
    # repr() of a chunk holding double quotes is a single-quoted literal.
    body = "\n".join(f"    {chunk!r}" for chunk in chunks)
    new = text[: match.start(1)] + body + text[match.end(1):]
    return new, new != text


# ---------------------------------------------------------- run-specific pins
def set_hex_pin(text: str, name: str, value: str) -> tuple[str, bool]:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name}: not a lowercase SHA-256: {value!r}")
    pattern = re.compile(rf'(^{name} = \(\s*")([0-9a-f]{{64}})("\s*\))', flags=re.M)
    if not pattern.search(text):
        raise ValueError(f"pin {name} not found")
    new = pattern.sub(lambda m: m.group(1) + value + m.group(3), text, count=1)
    return new, new != text


def set_int_pin(text: str, name: str, value: int) -> tuple[str, bool]:
    pattern = re.compile(rf"^({name} = )[0-9_]+$", flags=re.M)
    if not pattern.search(text):
        raise ValueError(f"pin {name} not found")
    new = pattern.sub(lambda m: m.group(1) + f"{int(value):_}", text, count=1)
    return new, new != text


def set_source_comment(text: str, *, tag: str, source_path: str, iteration: int,
                       probe_path: str) -> tuple[str, bool]:
    """Rewrite the comment above EXPECTED_V12_LEGACY_SOURCE_CHECKPOINT_SHA256."""

    lines = text.split("\n")
    try:
        pin = next(i for i, line in enumerate(lines)
                   if line.startswith("EXPECTED_V12_LEGACY_SOURCE_CHECKPOINT_SHA256 = ("))
    except StopIteration:
        return text, False
    start = pin
    while start > 0 and lines[start - 1].startswith("#"):
        start -= 1
        if re.match(r"# (Centered-HOME|HOME) chain", lines[start]):
            break
    else:
        return text, False
    if not re.match(r"# (Centered-HOME|HOME) chain", lines[start]):
        return text, False
    comment = [
        f"# HOME chain (mjlab_microban config/home_pose.yaml, HOME tag {tag},",
        "# scripts/retrain_all_for_home.py): the frozen source is the walking checkpoint",
        f"#   {source_path}",
        f'# (bootstrap provenance schema 2 records its SHA-256 and its saved "iter" {iteration}),',
        "# probed by",
        f"#   {probe_path}.",
        "# The deployed walk.onnx is that same checkpoint's export.",
    ]
    new_lines = lines[:start] + comment + lines[pin:]
    new = "\n".join(new_lines)
    return new, new != text


def set_quoted_values(text: str, values: Mapping[str, str]) -> tuple[str, bool]:
    """Set every ``"key": "value"`` (or ``"key": (\\n "value"\\n)``) literal of ``values``.

    Used for the packager metadata the robot tests pin for the installed
    package (tests/test_pico_hybrid.py: the frozen walking source and probe).
    """

    new = text
    for key, value in values.items():
        pattern = re.compile(rf'("{re.escape(key)}":\s*(?:\(\s*)?)"[^"\n]*"')
        new = pattern.sub(lambda m, v=value: m.group(1) + json.dumps(v), new)
    return new, new != text
