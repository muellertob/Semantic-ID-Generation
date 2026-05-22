import pytest
import torch
from modules.rqvae.model import RQ_VAE
from schemas.quantization import QuantizeForwardMode, QuantizeDistance

@pytest.mark.filterwarnings("ignore:Full backward hook is firing")
@pytest.mark.filterwarnings("ignore:Number of distinct clusters")
def test_full_pipeline_precision():
    """
    Ensures that every tensor (inputs, parameters, intermediate activations, outputs, gradients)
    throughout the entire RQ-VAE pipeline remains strictly torch.float32.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    # initialize model matching the amazon beauty setup
    model = RQ_VAE(
        input_dim=768,
        hidden_dims=[768, 256, 128],
        latent_dim=64,
        n_quantization_layers=3,
        codebook_size=128,
        commitment_weight=0.25,
        quantization_method=QuantizeForwardMode.STE,
        distance_mode=QuantizeDistance.L2,
        codebook_kmeans_init=False  # explicitly called
    ).to(device)
    
    for name, param in model.named_parameters():
        assert param.dtype == torch.float32, f"Parameter {name} has dtype {param.dtype} on init"
        
    # register hooks to check all forward and backward passes dynamically
    def forward_hook(module, args, output):
        def check_tensor(t, name):
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                assert t.dtype == torch.float32, f"module {module.__class__.__name__} {name} has dtype {t.dtype}"

        for i, arg in enumerate(args):
            check_tensor(arg, f"received input {i}")
            
        if isinstance(output, torch.Tensor):
            check_tensor(output, "produced output")
        elif hasattr(output, 'embeddings'):  # check custom schema output types
            check_tensor(output.embeddings, "produced embeddings")
            check_tensor(output.loss, "produced loss")
        elif isinstance(output, (list, tuple)):
            for i, item in enumerate(output):
                check_tensor(item, f"produced output item {i}")

    def backward_hook(module, grad_input, grad_output):
        def check_grad(g, name):
            if g is not None and g.is_floating_point():
                assert g.dtype == torch.float32, f"module {module.__class__.__name__} {name} has dtype {g.dtype}"

        for i, grad in enumerate(grad_input):
            check_grad(grad, f"grad_input {i}")
        for i, grad in enumerate(grad_output):
            check_grad(grad, f"grad_output {i}")

    hooks = []
    try:
        for module in model.modules():
            hooks.append(module.register_forward_hook(forward_hook))
            hooks.append(module.register_full_backward_hook(backward_hook))
            
        batch_size = 256
        data = torch.randn(batch_size, 768, device=device)
        assert data.dtype == torch.float32
        
        # test k-means initialization specifically, which runs the encoder and the k-means algorithm
        model.kmeans_init_codebooks(data)
        for name, param in model.named_parameters():
            assert param.dtype == torch.float32, f"Parameter {name} became {param.dtype} after kmeans init"

        model.train()
        out = model(data)
        assert out.loss.dtype == torch.float32, f"Final loss has dtype {out.loss.dtype}"
        
        out.loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert param.grad.dtype == torch.float32, f"Gradient for {name} has dtype {param.grad.dtype}"
                
        optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-3)
        optimizer.step()
        
        for name, param in model.named_parameters():
            assert param.dtype == torch.float32, f"Parameter {name} became {param.dtype} after optimizer step"
            
    finally:
        for h in hooks:
            h.remove()
