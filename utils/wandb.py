import os
import logging
import wandb
from dotenv import load_dotenv
from omegaconf import OmegaConf
from utils.model_id_generation import generate_model_id

logger = logging.getLogger(__name__)


def wandb_init(config, project: str, run_name: str = None, group: str = None, job_type: str = None, tags: list = None):
    """
    Initialise a WandB run.

    Args:
        config   : OmegaConf config object.
        project  : WandB project name (caller decides which stage project to use).
        run_name : Optional custom run name.
        group    : Optional custom group name.
        job_type : Optional custom job type name.
        tags     : Optional list of tags.

    general.wandb_project, general.wandb_run_name, general.wandb_group and general.wandb_tags, when set, win over
    project, run_name, group and tags: the search driver names its runs through the config.
    """
    wandb_dir = config.general.get("wandb_dir", os.path.join("outputs", "wandb", "runs"))
    os.makedirs(wandb_dir, exist_ok=True)
    wandb.init(
        project=_from_config(config, "wandb_project", project),
        name=_from_config(config, "wandb_run_name", run_name),
        group=_from_config(config, "wandb_group", group),
        job_type=job_type,
        tags=_from_config(config, "wandb_tags", tags),
        entity=config.general.wandb_entity,
        config=OmegaConf.to_container(config, resolve=True),
        resume="never",
        dir=wandb_dir,
    )


def prefix_run_name(config, prefix: str) -> None:
    """Put `prefix` in front of the active run's generated name, unless the config names the run."""
    if wandb.run is None or config.general.get("wandb_run_name") is not None:
        return
    if not wandb.run.name.startswith(f"{prefix}-"):
        wandb.run.name = f"{prefix}-{wandb.run.name}"


def _from_config(config, key: str, default):
    # tags may stay an OmegaConf list: wandb.init turns them into a tuple itself
    value = config.general.get(key)
    return default if value is None else value


def get_run_name(fallback: str) -> str:
    """
    Return the current WandB run name when a run is active, otherwise the fallback.

    Usage:
        model_id = get_run_name(fallback=generate_model_id(config))
        # → "sunny-energy-67"  (when wandb active)
        # → "amazon-beauty--bs256-..."  (when wandb disabled / offline)
    """
    if wandb.run is not None:
        return wandb.run.name
    return fallback


def log_model_artifact(model_path: str, run_name: str, artifact_type: str, metadata: dict = None, config = None):
    """
    Log a model file as a WandB Artifact so it can be retrieved by run name.

    Args:
        model_path    : Local path to the .pt file.
        run_name      : Artifact name (typically the WandB run name).
        artifact_type : e.g. "rqvae-tokenizer" or "tiger-recommender".
        metadata      : Optional dict of extra metadata to attach.
        config        : Optional config object containing wandb_log_artifacts setting.
    """
    if wandb.run is None:
        return

    # logging artifacts defaults to False
    log_artifacts = False
    if config is not None:
        try:
            general_cfg = config.general
        except (AttributeError, KeyError):
            general_cfg = None

        if general_cfg is not None:
            if hasattr(general_cfg, 'get'):
                log_artifacts = general_cfg.get('wandb_log_artifacts', False)
            else:
                log_artifacts = getattr(general_cfg, 'wandb_log_artifacts', False)

    if not log_artifacts:
        logger.info("WandB model artifact logging is disabled in the configuration.")
        return

    logger.info(f"Uploading model artifact to WandB: {run_name} (type: {artifact_type})...")
    artifact = wandb.Artifact(
        name=run_name,
        type=artifact_type,
        metadata=metadata or {},
    )
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)


def wandb_login():
    load_dotenv()
    key = os.getenv("wandb_key")
    wandb.login(key=key)


if __name__ == "__main__":
    wandb_login()
    print("WandB login successful.")
