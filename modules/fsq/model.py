from modules.vae.model import QuantizedAutoEncoder
from modules.rqvae.networks import Encoder, Decoder
from modules.fsq.quantization import FSQ, ResidualFSQ

def FSQ_AutoEncoder(input_dim, codebook_layers, hidden_dims, latent_dim, level_list, loss_type="mse", normalize=True, projection_type=None, inner_dim=None):
    encoder = Encoder(input_dim=input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim)
    decoder = Decoder(output_dim=input_dim, hidden_dims=hidden_dims[::-1], latent_dim=latent_dim)
    quantizer = FSQ(dim=latent_dim, codebook_layers=codebook_layers, level_list=level_list, projection_type=projection_type, inner_dim=inner_dim)
    return QuantizedAutoEncoder(encoder=encoder, decoder=decoder, quantizer=quantizer, input_dim=input_dim, latent_dim=latent_dim, loss_type=loss_type, normalize=normalize)

def ResidualFSQ_AutoEncoder(input_dim, codebook_layers, hidden_dims, latent_dim, level_list, loss_type="mse", normalize=True, projection_type="mlp_1_hidden", inner_dim=128):
    encoder = Encoder(input_dim=input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim)
    decoder = Decoder(output_dim=input_dim, hidden_dims=hidden_dims[::-1], latent_dim=latent_dim)
    quantizer = ResidualFSQ(dim=latent_dim, level_list=level_list, codebook_layers=codebook_layers, inner_dim=inner_dim, projection_type=projection_type)
    return QuantizedAutoEncoder(encoder=encoder, decoder=decoder, quantizer=quantizer, input_dim=input_dim, latent_dim=latent_dim, loss_type=loss_type, normalize=normalize)
