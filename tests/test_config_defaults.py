"""What a shipped config does when nobody overrides it.

Every run in `runs/` was launched with `--set train.batch_size=48 train.lr=6.0e-4
train.amp_dtype=bfloat16 ...`, so the values in the YAML files were never the values
that trained anything. That is how `hydranet_indoor.yaml` came to ask for twice the
effective learning rate of every run this project has produced -- it lowered
`batch_size` to 8 and inherited a `lr` the base had set for 16 -- and how the default
`amp_dtype` stayed at float16 while all nine runs used bfloat16.

Neither is visible to anyone who overrides both, which is everyone who has ever run
this. So the defaults get their own tests.

pytest tests/test_config_defaults.py -v
"""

from pathlib import Path

import pytest

from syncai_hydranet.config import load_config

CONFIGS = sorted(p for p in (Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"))

# Held by 48 -> 6.0e-4, 32 -> 4.0e-4 and 16 -> 2.0e-4, which is every from-scratch run
# in runs/. Not a law of nature -- a convention this project has followed without
# writing down, which is the reason it could be broken without anyone noticing.
LR_PER_SAMPLE = 1.25e-5

# A config is allowed off the line if it says why here. Fine-tuning is the real case:
# it wants a rate well below the from-scratch schedule regardless of batch size.
DECLARED_DEVIATIONS = {
    "hydranet_retail_cctv.yaml": (
        "fine-tunes from hydranet_retail_base/best.pt at 3.1e-6 per sample, a quarter "
        "of the from-scratch rate; runs/hydranet_retail_cctv used exactly this"
    ),
    "hydranet_retail_cocostuff.yaml": (
        "inherits the CCTV fine-tune's rate on purpose: its control is "
        "runs/hydranet_retail_cctv_noself, which is this recipe minus COCO-Stuff, and "
        "moving lr onto the from-scratch line would make that comparison meaningless"
    ),
}


def _ids(paths):
    return [p.name for p in paths]


@pytest.mark.parametrize("path", CONFIGS, ids=_ids(CONFIGS))
def test_lr_matches_batch_size(path):
    cfg = load_config(str(path), [])
    train = cfg["train"]
    ratio = train["lr"] / train["batch_size"]

    if path.name in DECLARED_DEVIATIONS:
        assert ratio != pytest.approx(LR_PER_SAMPLE, rel=1e-3), (
            f"{path.name} is listed in DECLARED_DEVIATIONS but now sits on the line; "
            "remove the entry rather than leaving a stale reason next to it"
        )
        return

    assert ratio == pytest.approx(LR_PER_SAMPLE, rel=1e-3), (
        f"{path.name} asks for lr={train['lr']:.2e} at batch_size="
        f"{train['batch_size']}, i.e. {ratio:.3e} per sample against the "
        f"{LR_PER_SAMPLE:.3e} every run used. Either move lr with batch_size or add "
        "an entry to DECLARED_DEVIATIONS saying why this one differs."
    )


@pytest.mark.parametrize("path", CONFIGS, ids=_ids(CONFIGS))
def test_amp_dtype_is_stated_not_inherited_from_code(path):
    """The code default is a last resort, not a specification.

    A config that leaves `amp_dtype` unset trains in whatever the trainer's fallback
    happens to be that month. Since the value materially changes the numerics -- and
    the detection loss has a logged crash that only fp16/bf16 autocast reaches -- the
    config has to say it.
    """
    cfg = load_config(str(path), [])
    assert cfg["train"].get("amp_dtype") == "bfloat16", (
        f"{path.name} does not resolve to an explicit amp_dtype: bfloat16"
    )


@pytest.mark.parametrize("path", CONFIGS, ids=_ids(CONFIGS))
def test_defaults_are_internally_consistent(path):
    """The cheap structural checks a config gets only when someone runs it."""
    from syncai_hydranet.config_schema import check_config

    cfg = load_config(str(path), [])
    check_config(cfg)  # raises ConfigError on anything structurally wrong

    train = cfg["train"]
    assert train["batch_size"] > 0
    assert train["epochs"] > 0
    assert 0 < train["lr"] < 1
    # A warm-up longer than the run means the schedule never leaves its ramp, and the
    # only symptom is a run that looks like it did not learn.
    assert train.get("warmup_iters", 0) < train["epochs"] * 1000
