from typing import Dict
from omegaconf import OmegaConf

def generate_model_id(config: Dict) -> str:
    dataset = str(config.data.dataset) + ("-" + str(config.data.category) if hasattr(config.data, 'category') and str(config.data.category) != '' else "")

    # check if this is a Recommender config
    if hasattr(config, 'seq2seq'):
        s = config.seq2seq
        d_model = s.get('d_model', 128)
        layers = s.get('num_layers', 4)
        heads = s.get('num_heads', 6)
        lr = s.get('learning_rate', 5e-4)
        bs = s.get('batch_size', 256)
        return f"recommender-{dataset}-dm{d_model}-l{layers}-h{heads}-lr{lr}-bs{bs}"

    # check if this is an RQ-VAE config
    elif hasattr(config, 'model') and hasattr(config, 'train'):
        m = config.model
        t = config.train
        d = config.data

        dimension = d.get('embedding_dimension', '')
        batch_size = d.get('batch_size', '')
        normalize_data = d.get('normalize_data', False)
        hidden_dimension = m.get('hidden_dimensions', [])
        latent_dimension = m.get('latent_dimension', '')
        num_codebook_layers = m.get('num_codebook_layers', '')
        codebook_clusters = m.get('codebook_clusters', '')
        commitment_weight = m.get('commitment_weight', '')
        learning_rate = t.get('learning_rate', '')
        weight_decay = t.get('weight_decay', '')
        num_epochs = t.get('num_epochs', '')

        # create the model ID string
        model_id = (
            f"rqvae-{dataset}-{dimension}-bs{batch_size}-norm{str(normalize_data)[0]}-"
            f"hd{'_'.join(map(str, hidden_dimension))}-ld{latent_dimension}-"
            f"cb{num_codebook_layers}x{codebook_clusters}-cw{commitment_weight}-"
            f"lr{learning_rate}-wd{weight_decay}-ep{num_epochs}"
        )
        return model_id

    return f"model-{dataset}-unknown"