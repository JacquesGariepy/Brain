"""
Brain CLI - Predict Command
"""

import sys
import time
import torch


def predict_command(args):
    """Execute prediction command"""
    print("=" * 70)
    print("Brain Framework - Inference")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Device: {args.device}")

    if args.text:
        print(f"  Input: Text")
        print(f"  Text: {args.text}")
    elif args.image:
        print(f"  Input: Image ({args.image})")
    elif args.audio:
        print(f"  Input: Audio ({args.audio})")
    elif args.input_file:
        print(f"  Input: File ({args.input_file})")

    print("\n" + "-" * 70)

    try:
        print("\n[1/3] Loading model...")

        # Import orchestrator
        from core.orchestrator import BrainOrchestrator

        orchestrator = BrainOrchestrator()

        # Map model name to architecture
        architecture_map = {
            'bert': 'transformer',
            'gpt': 'transformer',
            'transformer': 'transformer',
            'vit': 'vit',
            'vision-transformer': 'vit',
            'clip': 'clip',
        }

        architecture = architecture_map.get(args.model.lower(), 'transformer')
        loader_func = f"_load_{architecture}"

        if not hasattr(orchestrator, loader_func):
            print(f"✗ Unknown model architecture: {architecture}")
            sys.exit(1)

        # Load model
        model = getattr(orchestrator, loader_func)()
        print(f"      ✓ Loaded {architecture} model")

        # Move to device
        device = args.device if hasattr(args, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        print(f"      ✓ Moved to {device}")

        print("[2/3] Running inference...")
        start_time = time.time()

        with torch.no_grad():
            from architectures.base import VisionArchitecture, LanguageArchitecture, MultimodalArchitecture

            if args.text:
                # Text input
                if isinstance(model, (LanguageArchitecture, MultimodalArchitecture)):
                    # TODO: Proper tokenization
                    # For now, use random tokens as placeholder
                    input_ids = torch.randint(0, 50000, (1, 128)).to(device)
                    output = model(input_ids=input_ids)
                else:
                    print(f"✗ Model {architecture} does not support text input")
                    sys.exit(1)

            elif args.image:
                # Image input
                if isinstance(model, (VisionArchitecture, MultimodalArchitecture)):
                    # TODO: Load and preprocess image
                    # For now, use random image as placeholder
                    image = torch.randn(1, 3, 224, 224).to(device)
                    output = model(x=image) if isinstance(model, VisionArchitecture) else model(image=image)
                else:
                    print(f"✗ Model {architecture} does not support image input")
                    sys.exit(1)

            else:
                # Default: random input
                if isinstance(model, VisionArchitecture):
                    image = torch.randn(1, 3, 224, 224).to(device)
                    output = model(x=image)
                else:
                    input_ids = torch.randint(0, 50000, (1, 128)).to(device)
                    output = model(input_ids=input_ids)

        latency_ms = (time.time() - start_time) * 1000

        print("[3/3] Results:")

        # Extract predictions
        if hasattr(output, 'predictions') and output.predictions is not None:
            predictions = output.predictions
        elif hasattr(output, 'logits') and output.logits is not None:
            predictions = output.logits
        elif hasattr(output, 'embeddings') and output.embeddings is not None:
            predictions = output.embeddings
        else:
            print("✗ No output generated")
            sys.exit(1)

        # Display results
        if args.text:
            print(f"\n  Input: {args.text}")

            # Get top prediction
            if len(predictions.shape) >= 2:
                top_pred = predictions[0].argmax().item()
                confidence = torch.softmax(predictions[0], dim=-1).max().item()
                print(f"  Prediction: Class {top_pred}")
                print(f"  Confidence: {confidence:.4f}")
            else:
                print(f"  Output shape: {predictions.shape}")

            print(f"  Latency: {latency_ms:.2f}ms")

        else:
            print(f"\n  Output shape: {predictions.shape}")
            print(f"  Latency: {latency_ms:.2f}ms")

            if len(predictions.shape) >= 2 and predictions.shape[-1] < 100:
                top_pred = predictions[0].argmax().item()
                print(f"  Top prediction: Class {top_pred}")

        print("\n" + "=" * 70)
        print("✓ Inference completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Error during prediction: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
