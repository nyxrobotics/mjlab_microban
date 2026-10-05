"""Rewrite the robot repository's run- and HOME-specific pins (stdlib only).

Used by scripts/retrain_all_for_home.py after new policies are installed in
the robot repository (microban):

* run-specific pins: the frozen walking source of PICO v12
  (``EXPECTED_V12_LEGACY_SOURCE_CHECKPOINT_SHA256`` / ``_ITERATION`` /
  ``EXPECTED_V12_LEGACY_PROBE_SHA256`` in src/moves/pico_hybrid.py) and the
  walk fallback (``EXPECTED_WALK_FALLBACK_SHA256`` in
  tools/validate_pico_policy.py);
* HOME literals the robot tests pin on purpose (tests/test_shared_home.py,
  tests/test_home_pose_config.py, tests/test_pico_hybrid.py): the
  ``TRAINING_HOME_DEG`` table is regenerated, ``PACKAGER_V12_HOME_POSE_JSON``
  is set from the packaged ONNX, and distinctive tokens of the old HOME (the
  HOME tag, revision tokens such as ``plus1p198384259489``, full-precision
  angles and the root height) are replaced token by token with the new
  HOME's.  Short or ambiguous tokens are left alone and reported; the robot
  test suite is the judge of what is still stale.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping

HOME_TEST_FILES = (
    "tests/test_shared_home.py",
    "tests/test_home_pose_config.py",
    "tests/test_pico_hybrid.py",
)
PICO_HYBRID = "src/moves/pico_hybrid.py"
VALIDATOR = "tools/validate_pico_policy.py"

_NUMBER = re.compile(r"(?<![\w.])-?\d+\.\d+(?:e[-+]?\d+)?(?![\w.])")
_DEGREE_TOKEN = re.compile(r"(?<![A-Za-z0-9])(?:plus|minus)\d+(?:p\d+)?(?![A-Za-z0-9])")
MIN_NUMBER_TOKEN_CHARS = 8


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


def _flat_numbers(home: Mapping) -> dict[str, float]:
    out: dict[str, float] = {}
    for unit in ("joint_pos_deg", "joint_pos_rad"):
        for name, value in (home.get(unit) or {}).items():
            out[f"{unit}.{name}"] = float(value)
    root = home.get("root_pos_m")
    if isinstance(root, list) and len(root) == 3:
        out["root_z"] = float(root[2])
    return out


def build_home_token_map(old: Mapping, new: Mapping) -> tuple[dict[str, str], list[str]]:
    """Token map old HOME -> new HOME and the tokens skipped as ambiguous."""

    pairs: dict[str, set[str]] = {}
    unsigned: dict[str, set[str]] = {}

    def add(old_token: str, new_token: str, table: dict[str, set[str]] | None = None) -> None:
        if old_token != new_token:
            (pairs if table is None else table).setdefault(old_token, set()).add(new_token)

    old_numbers, new_numbers = _flat_numbers(old), _flat_numbers(new)
    for key, old_value in old_numbers.items():
        if key not in new_numbers:
            continue
        old_text, new_text = repr(old_value), repr(new_numbers[key])
        if len(old_text.lstrip("-")) >= MIN_NUMBER_TOKEN_CHARS:
            add(old_text, new_text)
            # Unsigned spelling (e.g. -math.radians(1.19...)) of a negative value;
            # used only where no value spells that token directly.
            if old_text.startswith("-") and new_text.startswith("-"):
                add(old_text[1:], new_text[1:], unsigned)
    for token, targets in unsigned.items():
        if token not in pairs:
            pairs[token] = targets
    for joint in ("left_hip_pitch", "left_ankle_pitch"):
        try:
            add(
                signed_degree_token(old["joint_pos_deg"][joint]),
                signed_degree_token(new["joint_pos_deg"][joint]),
            )
        except (KeyError, TypeError):
            pass
    ambiguous = sorted(token for token, targets in pairs.items() if len(targets) > 1)
    mapping = {token: next(iter(targets)) for token, targets in pairs.items() if len(targets) == 1}
    old_tag, new_tag = old.get("tag"), new.get("tag")
    if isinstance(old_tag, str) and isinstance(new_tag, str) and old_tag != new_tag:
        mapping["@tag"] = new_tag
        mapping["@old_tag"] = old_tag
    return mapping, ambiguous


def substitute_home_tokens(text: str, mapping: Mapping[str, str]) -> tuple[str, int]:
    """Apply a build_home_token_map mapping once; return (text, replacements)."""

    count = 0

    def number(match: re.Match) -> str:
        nonlocal count
        token = match.group(0)
        if token in mapping:
            count += 1
            return mapping[token]
        return token

    def degree(match: re.Match) -> str:
        nonlocal count
        token = match.group(0)
        if token in mapping:
            count += 1
            return mapping[token]
        return token

    def negated(match: re.Match) -> str:
        nonlocal count
        token = "-" + match.group(1)
        if token in mapping:
            count += 1
            return f"math.radians({mapping[token]})"
        return match.group(0)

    # "-math.radians(1.19...)" spells the negative value -1.19...
    text = re.sub(r"-math\.radians\((\d+\.\d+)\)", negated, text)
    text = _NUMBER.sub(number, text)
    text = _DEGREE_TOKEN.sub(degree, text)
    old_tag, new_tag = mapping.get("@old_tag"), mapping.get("@tag")
    if old_tag and new_tag:
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(old_tag)}(?![A-Za-z0-9])")
        lines = text.split("\n")
        for index, line in enumerate(lines):
            if line.lstrip().startswith("def "):
                continue  # test names keep their wording
            lines[index], tag_count = pattern.subn(new_tag, line)
            count += tag_count
        text = "\n".join(lines)
    return text, count


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
