"""
Main CLI entry point for the Semantic-ID Generation project.
Supports multiple commands:
- train-rqvae: Train the RQ-VAE model
- train-seq2seq: Train the Seq2Seq (Transformer) model
- test-seq2seq: Test a trained Seq2Seq model on the test set
- generate-ids: Generate Semantic IDs using a trained RQ-VAE
"""

import argparse
import logging
from train_rq_vae import run_training as run_rqvae
from train_seq2seq import run_training as run_seq2seq
from test_seq2seq import run_testing as test_seq2seq
from generate_semids import run_generation
from train_rqkmeans import run_training as run_rqkmeans

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
    seq2seq_parser.add_argument('--resume', type=str, help='Path to checkpoint file to resume from')
    seq2seq_parser.add_argument('--warmup_steps', type=int, help='Override warmup_steps from config')

    # Seq2Seq Testing
    test_seq2seq_parser = subparsers.add_parser('test-seq2seq', help='Test Seq2Seq (Transformer) model on test set')
    test_seq2seq_parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    test_seq2seq_parser.add_argument('--semids', type=str, required=True, help='Path to generated Semantic IDs (.pt file)')
    test_seq2seq_parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')

    # RQ-KMeans Training
    rqkmeans_parser = subparsers.add_parser('train-rqkmeans', help='Train RQ-KMeans and generate semantic IDs')
    rqkmeans_parser.add_argument('--config', type=str, required=True, help='Path to configuration file')

    # ID Generation
    gen_parser = subparsers.add_parser('generate-ids', help='Generate Semantic IDs')
    gen_parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    gen_parser.add_argument('--model_path', type=str, required=True, help='Path to trained RQ-VAE model')
    gen_parser.add_argument('--output_path', type=str, default='outputs/semids.pt', help='Path to save output')
    gen_parser.add_argument('--temperature', type=float, default=0.5, help='Sampling temperature')
    gen_parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    # SASRec benchmark
    sasrec_parser = subparsers.add_parser("train-sasrec", help="Train SASRec benchmark model")
    sasrec_parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    sasrec_parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    args = parser.parse_args()

    if args.command == 'train-rqkmeans':
        run_rqkmeans(args.config)
    elif args.command == 'train-rqvae':
        run_rqvae(args.config)
    elif args.command == 'train-seq2seq':
        run_seq2seq(args.config, args.semids, args.resume, args.warmup_steps)
    elif args.command == 'test-seq2seq':
        test_seq2seq(args.config, args.semids, args.model_path)
    elif args.command == 'train-sasrec':
        from train_sasrec import run_training as run_sasrec
        run_sasrec(args.config, resume_path=args.resume)
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