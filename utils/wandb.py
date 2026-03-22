import os
import wandb
from dotenv import load_dotenv
from omegaconf import OmegaConf
from utils.model_id_generation import generate_model_id


def wandb_init(config, project: str):
    """
    Initialise a WandB run.

    Args:
        config  : OmegaConf config object.
        project : WandB project name (caller decides which stage project to use).
    """
    wb_conf = config.model.copy()
    wb_conf.update(config.data)
    wb_conf.update(config.train)
    wb_conf.update({
        "model_id": generate_model_id(config),
    })

    wandb.init(
        project=project,
        entity=config.general.wandb_entity,
        config=OmegaConf.to_container(wb_conf, resolve=True),
        resume="never",
        dir=os.path.join("outputs", "wandb", "runs"),
    )


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


def log_model_artifact(model_path: str, run_name: str, artifact_type: str, metadata: dict = None):
    """
    Log a model file as a WandB Artifact so it can be retrieved by run name.

    Args:
        model_path    : Local path to the .pt file.
        run_name      : Artifact name (typically the WandB run name).
        artifact_type : e.g. "rqvae-tokenizer" or "tiger-recommender".
        metadata      : Optional dict of extra metadata to attach.
    """
    if wandb.run is None:
        return
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
