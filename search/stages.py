"""
The stage calls of one search trial.

Each stage first writes its resolved config to <trial>/configs/ (so the trial folder records what ran) and is
then called as a function in the worker process. The driver keys (seed, paths, compile, W&B names) are set here
and only here; a spec override may not set them, so trial.json cannot state another seed or path than the one used.

W&B: one project per dataset and stage, sid-search-<dataset>-quantizer and -recommender (each with its own
metrics). The group is the folder's parent (the study of a search trial, the source trial of a rerun) and the run name
<group>-<folder>; both are the same in the two projects, so a trial's two runs carry one name. RQ-KMeans fits
without W&B, so the W&B keys of its quantizer stage stay unused.
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

# (dataset, quantizer) -> (quantizer base config, recommender base config), relative to the repo root. The
# recommender bases differ only in codebook_size: 288 for the FSQ-type quantizers, 256 for the others.
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
               "general.run_name", "general.recommender_model_dir", "general.wandb_dir", "general.compile",
               "general.wandb_project", "general.wandb_group", "general.wandb_run_name", "general.wandb_tags")


@dataclasses.dataclass(frozen=True)
class QuantizerResult:
    semids: Path
    quantizer_meta: dict
    generate_meta: dict | None  # None for RQ-KMeans, which writes its SIDs while fitting


@dataclasses.dataclass(frozen=True)
class RecommenderResult:
    best_checkpoint: Path
    recommender_meta: dict


def base_configs(dataset: str, quantizer: str) -> tuple[Path, Path]:
    """The (quantizer, recommender) base config paths of a dataset and quantizer."""
    try:
        quantizer_base, recommender_base = BASE_CONFIGS[(dataset, quantizer)]
    except KeyError:
        raise ValueError(f"no base configs for dataset {dataset!r} and quantizer {quantizer!r}") from None
    return REPO_ROOT / quantizer_base, REPO_ROOT / recommender_base


def quantizer(spec: TrialSpec, folder: str | Path, compile: bool) -> QuantizerResult:
    """
    Fit the quantizer of `spec` and write the SIDs to <folder>/semids.pt.

    Neural quantizers train, then generate the SIDs from the same config; RQ-KMeans writes its SIDs itself.
    """
    if spec.semids_from is not None:
        raise ValueError(f"the quantizer stage does not run when semids_from is set ({spec.semids_from})")
    _reject_driver_keys("quantizer_overrides", spec.quantizer_overrides)
    folder = Path(folder)
    base, _ = base_configs(spec.dataset, spec.quantizer)
    semids = folder / "semids.pt"
    quantizer_meta = folder / "meta" / "quantizer.json"
    driver = {
        "general.seed": spec.quantizer_seed,
        "general.model_path": str(folder / "checkpoints" / "quantizer.pt"),
        "general.meta_json": str(quantizer_meta),
        "general.wandb_dir": str(folder / "wandb"),
        "general.compile": compile,
        **_wandb_keys(spec, folder, compile, "quantizer"),
    }

    if spec.quantizer == "rqkmeans":
        driver["general.output_path"] = str(semids)
        config = _write(base, spec.quantizer_overrides, driver, folder / "configs" / "quantizer.yaml")
        train_rqkmeans.run_training(str(config))
        return QuantizerResult(semids, _read_meta(quantizer_meta), None)

    config = _write(base, spec.quantizer_overrides, driver, folder / "configs" / "quantizer.yaml")
    train_quantizer.run_training(str(config))
    quantizer_meta_data = _read_meta(quantizer_meta)

    # the SIDs come from the config the quantizer was trained with; only the meta file differs
    generate_meta = folder / "meta" / "generate.json"
    generate_config = _write(base, spec.quantizer_overrides, {**driver, "general.meta_json": str(generate_meta)},
                             folder / "configs" / "generate.yaml")
    generate_semids.run_generation(str(generate_config), model_path=driver["general.model_path"],
                                   output_path=str(semids))
    return QuantizerResult(semids, quantizer_meta_data, _read_meta(generate_meta))


def semids_path(spec: TrialSpec, folder: str | Path) -> Path:
    """The SIDs the recommender trains on: the source trial's when semids_from is set, else this trial's own."""
    if spec.semids_from is not None:
        return Path(spec.semids_from) / "semids.pt"
    return Path(folder) / "semids.pt"


def recommender_config(spec: TrialSpec, folder: str | Path, compile: bool) -> Path:
    """Write the resolved recommender config of `spec` to <folder>/configs/recommender.yaml and return its path."""
    _reject_driver_keys("recommender_overrides", spec.recommender_overrides)
    folder = Path(folder)
    _, base = base_configs(spec.dataset, spec.quantizer)
    driver = {
        "general.seed": spec.recommender_seed,
        "general.meta_json": str(_recommender_meta(folder)),
        "general.run_name": "recommender",
        "general.recommender_model_dir": str(folder / "checkpoints"),
        "general.wandb_dir": str(folder / "wandb"),
        "general.compile": compile,
        **_wandb_keys(spec, folder, compile, "recommender"),
    }
    return _write(base, spec.recommender_overrides, driver, folder / "configs" / "recommender.yaml")


def recommender(spec: TrialSpec, folder: str | Path, config: str | Path, on_validation) -> RecommenderResult:
    """
    Train the recommender with `config` (from recommender_config) on the SIDs of `spec`; `on_validation` is passed
    through to train_seq2seq.run_training. The best checkpoint is <folder>/checkpoints/recommender_best.pt, as named in
    the recommender meta.
    """
    train_seq2seq.run_training(str(config), str(semids_path(spec, folder)), on_validation=on_validation)
    meta = _read_meta(_recommender_meta(Path(folder)))
    return RecommenderResult(Path(meta["best_model_path"]), meta)


def test(spec: TrialSpec, folder: str | Path, best_checkpoint: str | Path) -> dict:
    """Evaluate `best_checkpoint` on the test split with the recommender config of this trial; returns the test meta."""
    folder = Path(folder)
    test_meta = folder / "meta" / "test.json"
    # the trained config itself, so the model is tested exactly as it was trained; only the meta file differs
    config = write_resolved_config(folder / "configs" / "recommender.yaml", {"general.meta_json": str(test_meta)},
                                   folder / "configs" / "test.yaml")
    test_seq2seq.run_testing(str(config), str(semids_path(spec, folder)), model_path=str(best_checkpoint))
    return _read_meta(test_meta)


def _write(base: Path, overrides: dict, driver: dict, dest: Path) -> Path:
    return write_resolved_config(base, {**overrides, **driver}, dest, allow_new=DRIVER_KEYS)


def _wandb_keys(spec: TrialSpec, folder: Path, compile: bool, stage: str) -> dict:
    folder = folder.absolute()  # both names from the same path, also for a relative folder
    group = folder.parent.name
    return {"general.wandb_project": f"sid-search-{spec.dataset}-{stage}",
            "general.wandb_group": group,
            "general.wandb_run_name": f"{group}-{folder.name}",  # no "/": it also names W&B artifacts
            "general.wandb_tags": [spec.phase, spec.quantizer, "compile" if compile else "eager"]}


def _recommender_meta(folder: Path) -> Path:
    return folder / "meta" / "recommender.json"


def _reject_driver_keys(name: str, overrides: dict) -> None:
    driver_keys = sorted(key for key in overrides if key in DRIVER_KEYS)
    if driver_keys:
        raise ValueError(f"{name} must not set driver keys: {driver_keys}")


def _read_meta(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"the stage finished without writing its meta file {path}")
    with open(path) as f:
        return json.load(f)
