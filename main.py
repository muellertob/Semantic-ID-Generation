"""
Main CLI entry point for the Semantic-ID Generation project.
Supports multiple commands:
- train-rqvae: Train the RQ-VAE model
- train-seq2seq: Train the Seq2Seq (Transformer) model
- generate-ids: Generate Semantic IDs using a trained RQ-VAE
"""

import argparse
import logging
from train_rq_vae import run_training as run_rqvae
from train_seq2seq import run_training as run_seq2seq
from generate_semids import run_generation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Semantic ID Generation CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # RQ-VAE Training
    rqvae_parser = subparsers.add_parser('train-rqvae', help='Train RQ-VAE model')
    rqvae_parser.add_argument('--config', type=str, required=True, help='Path to configuration file')

    # Seq2Seq Training (TIGER Transformer)
    seq2seq_parser = subparsers.add_parser('train-seq2seq', help='Train Seq2Seq (Transformer) model')
    seq2seq_parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    seq2seq_parser.add_argument('--semids', type=str, required=True, help='Path to generated Semantic IDs (.pt file)')

    # ID Generation
    gen_parser = subparsers.add_parser('generate-ids', help='Generate Semantic IDs')
    gen_parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    gen_parser.add_argument('--model_path', type=str, required=True, help='Path to trained RQ-VAE model')
    gen_parser.add_argument('--output_path', type=str, default='outputs/semids.pt', help='Path to save output')
    gen_parser.add_argument('--temperature', type=float, default=0.5, help='Sampling temperature')
    gen_parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    args = parser.parse_args()

    if args.command == 'train-rqvae':
        run_rqvae(args.config)
    elif args.command == 'train-seq2seq':
        run_seq2seq(args.config, args.semids)
    elif args.command == 'generate-ids':
        run_generation(
            config_path=args.config,
            model_path=args.model_path,
            output_path=args.output_path,
            temperature=args.temperature,
            batch_size=args.batch_size
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()