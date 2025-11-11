#!/usr/bin/env python3
"""
Brain CLI - Main Entry Point

Command-line interface for the Brain framework.

Usage:
    brain --help
    brain train --config config.yaml
    brain evaluate --model checkpoint.pt --dataset test
    brain serve --port 8000
"""

import sys
import argparse
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser"""

    parser = argparse.ArgumentParser(
        prog="brain",
        description="Brain Framework - Comprehensive AI toolkit with 46+ SOTA architectures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  brain train --config configs/bert.yaml --dataset glue/sst2

  # Evaluate a model
  brain evaluate --model checkpoints/model.pt --dataset test.json

  # Start API server
  brain serve --host 0.0.0.0 --port 8000

  # Run inference
  brain predict --model bert-base --text "Hello world"

For more information, visit: https://github.com/yourusername/Brain
        """
    )

    parser.add_argument(
        "--version",
        action="version",
        version="Brain Framework 1.0.0"
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands"
    )

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train a model",
        description="Train a machine learning model"
    )
    train_parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (YAML)"
    )
    train_parser.add_argument(
        "--model",
        type=str,
        default="bert-base-uncased",
        help="Model architecture name"
    )
    train_parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name or path"
    )
    train_parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints"
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size"
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    train_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for training"
    )
    train_parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    train_parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLflow logging"
    )

    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a model",
        description="Evaluate a trained model on a dataset"
    )
    eval_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint or model name"
    )
    eval_parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Evaluation dataset"
    )
    eval_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Evaluation batch size"
    )
    eval_parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["accuracy"],
        help="Metrics to compute"
    )
    eval_parser.add_argument(
        "--output",
        type=str,
        default="./results.json",
        help="Output file for results"
    )
    eval_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use"
    )

    # Predict command
    predict_parser = subparsers.add_parser(
        "predict",
        help="Run inference",
        description="Run inference on input data"
    )
    predict_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or checkpoint path"
    )
    predict_parser.add_argument(
        "--text",
        type=str,
        help="Input text (for NLP tasks)"
    )
    predict_parser.add_argument(
        "--image",
        type=str,
        help="Input image path (for vision tasks)"
    )
    predict_parser.add_argument(
        "--audio",
        type=str,
        help="Input audio path (for audio tasks)"
    )
    predict_parser.add_argument(
        "--input-file",
        type=str,
        help="Input file with multiple examples (JSON/CSV)"
    )
    predict_parser.add_argument(
        "--output-file",
        type=str,
        help="Output file for predictions"
    )
    predict_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use"
    )

    # Serve command
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start API server",
        description="Start the Brain API server"
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address"
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number"
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development mode)"
    )
    serve_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes"
    )

    # List command
    list_parser = subparsers.add_parser(
        "list",
        help="List available models",
        description="List all available models and architectures"
    )
    list_parser.add_argument(
        "--category",
        type=str,
        choices=["all", "transformers", "vision", "audio", "multimodal"],
        default="all",
        help="Filter by category"
    )

    # Download command
    download_parser = subparsers.add_parser(
        "download",
        help="Download pretrained models",
        description="Download pretrained models or datasets"
    )
    download_parser.add_argument(
        "name",
        type=str,
        help="Model or dataset name"
    )
    download_parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Output directory"
    )

    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show system information",
        description="Display system and framework information"
    )

    return parser


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()

    # If no command specified, show help
    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Import command modules (lazy loading)
    try:
        if args.command == "train":
            from .commands.train import train_command
            train_command(args)

        elif args.command == "evaluate":
            from .commands.evaluate import evaluate_command
            evaluate_command(args)

        elif args.command == "predict":
            from .commands.predict import predict_command
            predict_command(args)

        elif args.command == "serve":
            from .commands.serve import serve_command
            serve_command(args)

        elif args.command == "list":
            from .commands.list_models import list_command
            list_command(args)

        elif args.command == "download":
            from .commands.download import download_command
            download_command(args)

        elif args.command == "info":
            from .commands.info import info_command
            info_command(args)

        else:
            parser.print_help()
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
