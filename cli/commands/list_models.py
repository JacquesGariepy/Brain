"""
Brain CLI - List Models Command
"""


def list_command(args):
    """List available models"""
    print("=" * 70)
    print("Brain Framework - Available Models")
    print("=" * 70)

    models_by_category = {
        "Transformers": [
            "bert-base-uncased",
            "gpt-2",
            "t5-base",
            "roberta-base",
        ],
        "Vision": [
            "clip-vit-base",
            "yolov8",
            "sam-vit-h",
            "dinov2-base",
        ],
        "Audio": [
            "whisper-base",
            "wav2vec2-base",
            "musicgen-small",
        ],
        "Multimodal": [
            "blip2",
            "llava-1.5",
            "flamingo",
        ],
    }

    category_filter = args.category

    for category, models in models_by_category.items():
        if category_filter != "all" and category.lower() != category_filter:
            continue

        print(f"\n{category}:")
        print("-" * 70)
        for model in models:
            print(f"  • {model}")

    print("\n" + "=" * 70)
    print(f"Total: {sum(len(m) for m in models_by_category.values())} models")
    print("=" * 70)
