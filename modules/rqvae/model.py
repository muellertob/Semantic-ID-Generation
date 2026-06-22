from modules.vae.model import QuantizedAutoEncoder
from modules.rqvae.networks import Encoder, Decoder
from modules.rqvae.quantization import ResidualVectorQuantizer
from schemas.quantization import QuantizeForwardMode, QuantizeDistance

def RQ_VAE(
        input_dim, 
        latent_dim, 
        hidden_dims, 
        codebook_size, 
        codebook_kmeans_init=True, 
        codebook_sim_vq=False, 
        n_quantization_layers=3, 
        commitment_weight=0.25, 
        quantization_method=QuantizeForwardMode.STE, 
        distance_mode=QuantizeDistance.L2
    ) -> QuantizedAutoEncoder:
        encoder = Encoder(
            input_dim=input_dim, 
            hidden_dims=hidden_dims, 
            latent_dim=latent_dim
        )
        decoder = Decoder(
            output_dim=input_dim, 
            hidden_dims=hidden_dims[::-1], 
            latent_dim=latent_dim
        )
        quantizer = ResidualVectorQuantizer(
            n_quantization_layers=n_quantization_layers,
            latent_dim=latent_dim,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
            do_kmeans_init=codebook_kmeans_init,
            sim_vq=codebook_sim_vq,
            forward_mode=quantization_method,
            distance_mode=distance_mode,
        )
        return QuantizedAutoEncoder(
            encoder=encoder, 
            decoder=decoder, 
            quantizer=quantizer, 
            input_dim=input_dim, 
            latent_dim=latent_dim
        )