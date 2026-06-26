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

    # check if this is a quantizer config (FSQ / RQ-VAE)
    elif hasattr(config, 'model') and hasattr(config, 'train'):
        m = config.model
        t = config.train
        d = config.data

        quantizer_type = m.get('quantizer_type')
        if quantizer_type == 'residual_fsq':
            prefix = 'rfsq'
        elif quantizer_type == 'fsq':
            prefix = 'fsq'
        elif quantizer_type == 'rqvae':
            prefix = 'rqvae'
        else:
            prefix = 'quantizer'

        dimension = d.get('embedding_dimension', '')
        dim_str = f"-{dimension}" if dimension else ""
        batch_size = d.get('batch_size', '')
        normalize_data = d.get('normalize_data', False)
        hidden_dimension = m.get('hidden_dimensions', [])
        latent_dimension = m.get('latent_dimension', '')
        codebook_layers = m.get('codebook_layers', '')
 
        # FSQ/ResidualFSQ specific model ID
        if quantizer_type in ['fsq', 'residual_fsq']:
            level_list = m.get('level_list', [])
            levels_str = '_'.join(map(str, level_list))
            proj_type = m.get('projection_type', 'none')
            inner_dim = m.get('inner_dim', '')
            
            model_id = (
                f"{prefix}-{dataset}{dim_str}-bs{batch_size}-norm{str(normalize_data)[0]}-"
                f"hd{'_'.join(map(str, hidden_dimension))}-ld{latent_dimension}-"
                f"l{codebook_layers}-levels{levels_str}-proj_{proj_type}_{inner_dim}-"
                f"lr{t.get('learning_rate', '')}-wd{t.get('weight_decay', '')}-ep{t.get('num_epochs', '')}"
            )
        # RQ-VAE specific model ID
        else:
            codebook_size = m.get('codebook_size', '')
            commitment_weight = m.get('commitment_weight', '')
            
            model_id = (
                f"{prefix}-{dataset}{dim_str}-bs{batch_size}-norm{str(normalize_data)[0]}-"
                f"hd{'_'.join(map(str, hidden_dimension))}-ld{latent_dimension}-"
                f"cb{codebook_layers}x{codebook_size}-cw{commitment_weight}-"
                f"lr{t.get('learning_rate', '')}-wd{t.get('weight_decay', '')}-ep{t.get('num_epochs', '')}"
            )
        return model_id

    return f"model-{dataset}-unknown"