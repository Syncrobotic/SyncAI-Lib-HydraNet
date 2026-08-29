"""PLAN §5's prohibitions, for the three that a test can actually hold.

§5 lists six things this product must never do. Five of the six have lived only in that
paragraph since it was written, and this repository has paid for prose-only invariants
often enough to have a name for it: the two-package boundary was a comment until
`test_package_boundaries.py`, the assets allowlist was a comment until
`test_assets_allowlist.py`, and the release workflow's header stated "no CHANGELOG.md in
the tree at all" fifteen lines above the step that read it unconditionally.

**The list with the legal exposure was the one with no test**, which is the wrong way
round. §5 item 5 bans inferring age or gender on customers, face recognition, and cross-store
tracking. Two of those three are import questions, so they are checkable, and this file
checks them.

WHICH OF THE SIX THIS FILE HOLDS, AND WHICH IT CANNOT
-----------------------------------------------------
Stated rather than left to be discovered, because a guard file named after a
six-item list invites the reading that it covers six items.

* §5 item 1 *an answer in the weights that belongs in a config* -- **not checkable here.** It
  is a judgement about what a head learned, and the instrument is review, not `ast`.
* §5 item 2 *the rule layer reaching into the model* -- **checked.** The event layer importing
  `models` or `engine` is the mechanical form of it.
* §5 item 3 *a per-class number resting on one camera* -- **not checkable here.** It is a claim
  about how a figure is reported; `analytics/staff.py` enforces its own version by
  refusing to license a camera on a single-class held-out set.
* §5 item 4 *coverage bought with more frames from annotated cameras* -- **not checkable here**,
  it is a property of a dataset build.
* §5 item 5 *age/gender, face recognition, cross-store tracking* -- **two of three checked**,
  see below.
* §5 item 6 *SKU-level checkout, new hardware, per-frame dense scene understanding* -- **not
  checkable here.** The dense-head half is a config question (which heads a config names),
  not an import one.

WHY §5 item 5's FIRST CLAUSE NEEDS A GUARD TODAY RATHER THAN LATER
------------------------------------------------------------------
The capability §5 item 5 forbids **is built, tested, and sitting inside the serving package.**
`data/attributes.py` loads PA-100K and its label vocabulary opens `Female` (45.62% of
80,000 crops) and `AgeOver60`; `analytics/track_attributes.py` pools per-frame gender
logits into one verdict per shopper and carries an `age_band()`. Neither is reachable
from a serving process today -- `tracker.py` and `staff.py` name `track_attributes` in
comments and import nothing from it -- so this guard passes on the day it is written,
which is the point. It is one `from . import track_attributes` away from being false, and
that import would be an ordinary-looking line in a diff about something else.

The modules are **not deleted**, and that is deliberate rather than an oversight: their
measurements are cited by `analytics/staff.py` and by §7.15, and the finding that the same
staff member is labelled `F` and `M` in adjacent frames is *why* the shipped classifier
reads torso colour instead. Keeping the evidence and banning the wiring is the honest
arrangement. `staff/customer` stays in, exactly as §5 item 5 says: it is a uniform, not an
identity, and nothing here touches `analytics/staff.py`.

**Cross-store tracking has no test here, and that is a gap rather than a decision.**
§7a-5 puts single-store re-linking in scope and bans the cross-store form, and the
machinery either would use is the same appearance code, so an import ban would forbid
something planned. The thing that would carry it is `WorldFrame.space`, which is
`camera_floor(<camera_id>)` today and by §2.3.1 must gain a store frame later; the guard
worth writing is that no `space` value ever spans two stores, and it cannot be written
until a second value exists. Recorded here so the absence is visible rather than assumed.

An `ast` walk, matching `test_package_boundaries.py`, so the check does not need torch
importable -- and so a *comment* discussing these measurements never trips a guard aimed
at code. Several serving-path modules discuss them at length and must stay free to.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Mirrors `test_package_boundaries.SERVING_PATH` rather than importing it: these are two
# statements of what a serving process loads, and one changing is a reason to look at the
# other rather than to have moved it silently.
SERVING_PATH = ("serving", "analytics", "models", "engine", "geometry")

# The quarantine. Present, cited, tested -- and unreachable from a serving process.
ATTRIBUTE_MODULES = (
    "syncai_hydranet.data.attributes",
    "syncai_hydranet.analytics.track_attributes",
)
QUARANTINED_FILES = (
    REPO / "src/syncai_hydranet/data/attributes.py",
    REPO / "src/syncai_hydranet/analytics/track_attributes.py",
)

# Libraries whose only purpose is recognising a face, as opposed to finding or blurring
# one. `utils/face_blur.py` is the opposite of this list and imports numpy and PIL: it
# exists to make a face unreadable, which is the obligation, not the breach.
FACE_ID_PACKAGES = (
    "face_recognition",
    "insightface",
    "facenet",
    "facenet_pytorch",
    "deepface",
    "dlib",
    "arcface",
)

# Read as data, these name a person's protected attributes. Read in a comment they name a
# measurement this repository made and should keep citing, which is why the check below
# looks at string constants only.
ATTRIBUTE_VOCABULARY = (
    "Female",
    "Male",
    "AgeOver60",
    "AgeLess16",
    "Age18-60",
    "Age17-30",
    "Age31-45",
)


def _imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module)
            found |= {f"{node.module}.{a.name}" for a in node.names}
    return found


def _serving_files() -> list[Path]:
    files: list[Path] = []
    for part in SERVING_PATH:
        files += sorted(REPO.glob(f"src/syncai_hydranet/{part}/**/*.py"))
    return [f for f in files if f not in QUARANTINED_FILES]


def test_the_serving_path_infers_no_age_or_gender():
    """PLAN §5 item 5. The ban is on the wiring, not on the evidence.

    `track_attributes.py` lives in `analytics/`, which is itself on the serving path, so
    the module is excluded from its own scan -- the question is whether anything a
    serving process loads pulls it in, not whether it exists.
    """
    offenders = []
    for f in _serving_files():
        hits = {m for m in _imports_of(f) if m.startswith(ATTRIBUTE_MODULES)}
        if hits:
            offenders.append(f"{f.relative_to(REPO)} -> {sorted(hits)}")
    assert not offenders, (
        "PLAN §5 item 5: 'Never infer age/gender on customers.' A serving-path module imports "
        "the attribute stack:\n  "
        + "\n  ".join(offenders)
        + "\n\nThese modules are kept for the measurement that produced them -- the same "
        "staff member labelled `F` and `M` in adjacent frames, which is why the shipped "
        "staff classifier reads torso colour. They are evidence, not a feature. If a "
        "customer-facing attribute is genuinely wanted, §5 is the file to change first, "
        "and changing it is a decision about what this product is."
    )


def test_no_serving_module_carries_an_attribute_vocabulary():
    """The second way in, and the one an import ban alone would miss.

    A head could learn these classes and a serving module hold its label list without
    importing anything from `data/`. String constants only: the modules that discuss the
    gender-flicker measurement in prose are doing the right thing and must not be
    penalised for saying so.
    """
    offenders = []
    for f in _serving_files():
        tree = ast.parse(f.read_text(encoding="utf-8"))
        hits = sorted(
            {
                node.value
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in ATTRIBUTE_VOCABULARY
            }
        )
        if hits:
            offenders.append(f"{f.relative_to(REPO)} -> {hits}")
    assert not offenders, (
        "PLAN §5 item 5: a serving-path module holds protected-attribute class names as "
        "data:\n  " + "\n  ".join(offenders)
    )


def test_nothing_in_the_tree_does_face_recognition():
    """PLAN §5 item 5, and the whole tree rather than the serving path.

    Wider than the other guards on purpose. A commissioning tool or a one-off script that
    identified a face would breach this as squarely as a serving module, and the
    difference between finding a face and recognising one is the entire distance between
    `utils/face_blur.py` and a breach.
    """
    offenders = []
    for root in ("src", "tools", "scripts"):
        for f in sorted((REPO / root).glob("**/*.py")):
            hits = {m for m in _imports_of(f) if m.split(".")[0] in FACE_ID_PACKAGES}
            if hits:
                offenders.append(f"{f.relative_to(REPO)} -> {sorted(hits)}")
    assert not offenders, (
        "PLAN §5 item 5: 'no face recognition.' A face-identification library is imported:\n  "
        + "\n  ".join(offenders)
    )


def test_the_rule_layer_does_not_reach_into_the_model():
    """PLAN §5 item 2, and §2.3's 'nothing crosses from L3/L4 down into L0'.

    The event layer reads tracks in metres and seconds. A threshold is an argument, so
    `analytics/events/` needs no access to weights, logits or the engine -- and an event
    module that imported them would be the beginning of a rule that cannot be changed
    without a retrain, which is the arrangement §5 item 1 and §5 item 2 exist together to prevent.
    """
    banned = ("syncai_hydranet.models", "syncai_hydranet.engine", "syncai_hydranet.serving")
    offenders = []
    for f in sorted(REPO.glob("src/syncai_hydranet/analytics/events/**/*.py")):
        hits = {m for m in _imports_of(f) if m.startswith(banned)}
        if hits:
            offenders.append(f"{f.relative_to(REPO)} -> {sorted(hits)}")
    assert not offenders, (
        "PLAN §5 item 2: the rule layer reaches into the model. '4 minutes is loitering' is an "
        "argument, and an event module that imports the model stack has stopped being "
        "one:\n  " + "\n  ".join(offenders)
    )


def test_the_quarantined_modules_still_exist():
    """The guards above are vacuous if the files they name are gone.

    A ban on importing something that no longer exists passes for the wrong reason, and
    silently: the next person to add a gender attribute would add it at a fresh path, and
    `ATTRIBUTE_MODULES` would point at nothing. If these modules are ever deleted, delete
    the guard deliberately rather than letting it decay into a test of nothing.
    """
    missing = [str(p.relative_to(REPO)) for p in QUARANTINED_FILES if not p.is_file()]
    assert not missing, (
        "the modules the §5 item 5 import ban names have moved or been deleted, so that ban "
        "now checks nothing:\n  " + "\n  ".join(missing)
    )
