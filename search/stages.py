"""
The stage calls of one search trial.

Each stage first writes its resolved config to <trial>/configs/ (so the trial folder records what ran) and is
then called as a function in the worker process. The driver keys (seed, paths, compile) are set here and only
here; a spec override may not set them, so trial.json cannot state another seed or path than the one used.
"""
import dataclasses
import json
from pathlib import Path

import generate_semids
import test_seq2seq
import train_quantizer
import train_rqkmeans
import train_seq2seq
from search.config_io import write_resolved_config
from search.spec import TrialSpec

REPO_ROOT = Path(__file__).resolve().parent.parent

# (dataset, quantizer) -> (Stage-1 base config, Stage-2 base config), relative to the repo root. The Stage-2
# bases differ only in codebook_size: 288 for the FSQ-type quantizers, 256 for the others.
BASE_CONFIGS = {
    ("toys", "rqvae"): ("config/rq-vae/amazon_toy_tiger_gts.yaml", "config/recommender/amazon_toy_grid_gts.yaml"),
    ("toys", "rqkmeans"): ("config/rq-kmeans/amazon_toy_rqkmeans_gts.yaml",
                           "config/recommender/amazon_toy_grid_gts.yaml"),
    ("toys", "fsq"): ("config/fsq/amazon_toy_gts.yaml", "config/recommender/amazon_toy_grid_fsq_gts.yaml"),
    ("toys", "rfsq"): ("config/fsq/amazon_toy_rfsq_fixed_gts.yaml", "config/recommender/amazon_toy_grid_fsq_gts.yaml"),
    ("music4all_onion", "rqvae"): ("config/rq-vae/onion_tiger_gts.yaml", "config/recommender/onion_grid_gts.yaml"),
    ("music4all_onion", "rqkmeans"): ("config/rq-kmeans/onion_rqkmeans_gts.yaml",
                                      "config/recommender/onion_grid_gts.yaml"),
    ("music4all_onion", "fsq"): ("config/fsq/onion_gts.yaml", "config/recommender/onion_grid_fsq_gts.yaml"),
}

DRIVER_KEYS = ("general.seed", "general.model_path", "general.output_path", "general.meta_json",
               "general.run_name", "general.recommender_model_dir", "general.wandb_dir", "general.compile")


@dataclasses.dataclass(frozen=True)
class Stage1Result:
    semids: Path
    quantizer_meta: dict
    generate_meta: dict | None  # None for RQ-KMeans, which writes its SIDs while fitting


@dataclasses.dataclass(frozen=True)
class Stage2Result:
    best_checkpoint: Path
    train_meta: dict


def base_configs(dataset: str, quantizer: str) -> tuple[Path, Path]:
    """The (Stage-1, Stage-2) base config paths of a dataset and quantizer."""
    try:
        stage1_base, stage2_base = BASE_CONFIGS[(dataset, quantizer)]
    except KeyError:
        raise ValueError(f"no base configs for dataset {dataset!r} and quantizer {quantizer!r}") from None
    return REPO_ROOT / stage1_base, REPO_ROOT / stage2_base


def stage1(spec: TrialSpec, folder: str | Path, compile: bool) -> Stage1Result:
    """
    Fit the quantizer of `spec` and write the SIDs to <folder>/semids.pt.

    Neural quantizers train, then generate the SIDs from the same config; RQ-KMeans writes its SIDs itself.
    """
    if spec.semids_from is not None:
        raise ValueError(f"Stage 1 does not run when semids_from is set ({spec.semids_from})")
    _reject_driver_keys("stage1_overrides", spec.stage1_overrides)
    folder = Path(folder)
    base, _ = base_configs(spec.dataset, spec.quantizer)
    semids = folder / "semids.pt"
    quantizer_meta = folder / "meta" / "quantizer.json"
    driver = {
        "general.seed": spec.stage1_seed,
        "general.model_path": str(folder / "checkpoints" / "quantizer.pt"),
        "general.meta_json": str(quantizer_meta),
        "general.wandb_dir": str(folder / "wandb"),
        "general.compile": compile,
    }

    if spec.quantizer == "rqkmeans":
        driver["general.output_path"] = str(semids)
        config = _write(base, spec.stage1_overrides, driver, folder / "configs" / "stage1.yaml")
        train_rqkmeans.run_training(str(config))
        return Stage1Result(semids, _read_meta(quantizer_meta), None)

    config = _write(base, spec.stage1_overrides, driver, folder / "configs" / "stage1.yaml")
    train_quantizer.run_training(str(config))
    quantizer_meta_data = _read_meta(quantizer_meta)

    # the SIDs come from the config the quantizer was trained with; only the meta file differs
    generate_meta = folder / "meta" / "generate.json"
    generate_config = _write(base, spec.stage1_overrides, {**driver, "general.meta_json": str(generate_meta)},
                             folder / "configs" / "generate.yaml")
    generate_semids.run_generation(str(generate_config), model_path=driver["general.model_path"],
                                   output_path=str(semids))
    return Stage1Result(semids, quantizer_meta_data, _read_meta(generate_meta))


def semids_path(spec: TrialSpec, folder: str | Path) -> Path:
    """The SIDs Stage 2 trains on: the source trial's when semids_from is set, else this trial's own."""
    if spec.semids_from is not None:
        return Path(spec.semids_from) / "semids.pt"
    return Path(folder) / "semids.pt"


def stage2(spec: TrialSpec, folder: str | Path, compile: bool, on_validation) -> Stage2Result:
    """
    Train the recommender of `spec` on its SIDs; `on_validation` is passed through to train_seq2seq.run_training.
    The best checkpoint is <folder>/checkpoints/stage2_best.pt, as named in the train meta.
    """
    _reject_driver_keys("stage2_overrides", spec.stage2_overrides)
    folder = Path(folder)
    _, base = base_configs(spec.dataset, spec.quantizer)
    train_meta = folder / "meta" / "train.json"
    driver = {
        "general.seed": spec.stage2_seed,
        "general.meta_json": str(train_meta),
        "general.run_name": "stage2",
        "general.recommender_model_dir": str(folder / "checkpoints"),
        "general.wandb_dir": str(folder / "wandb"),
        "general.compile": compile,
    }
    config = _write(base, spec.stage2_overrides, driver, folder / "configs" / "stage2.yaml")
    train_seq2seq.run_training(str(config), str(semids_path(spec, folder)), on_validation=on_validation)
    meta = _read_meta(train_meta)
    return Stage2Result(Path(meta["best_model_path"]), meta)


def test(spec: TrialSpec, folder: str | Path, best_checkpoint: str | Path) -> dict:
    """Evaluate `best_checkpoint` on the test split with the Stage-2 config of this trial; returns the test meta."""
    folder = Path(folder)
    test_meta = folder / "meta" / "test.json"
    # the trained config itself, so the model is tested exactly as it was trained; only the meta file differs
    config = write_resolved_config(folder / "configs" / "stage2.yaml", {"general.meta_json": str(test_meta)},
                                   folder / "configs" / "test.yaml")
    test_seq2seq.run_testing(str(config), str(semids_path(spec, folder)), model_path=str(best_checkpoint))
    return _read_meta(test_meta)


def _write(base: Path, overrides: dict, driver: dict, dest: Path) -> Path:
    return write_resolved_config(base, {**overrides, **driver}, dest, allow_new=DRIVER_KEYS)


def _reject_driver_keys(name: str, overrides: dict) -> None:
    driver_keys = sorted(key for key in overrides if key in DRIVER_KEYS)
    if driver_keys:
        raise ValueError(f"{name} must not set driver keys: {driver_keys}")


def _read_meta(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"the stage finished without writing its meta file {path}")
    with open(path) as f:
        return json.load(f)
