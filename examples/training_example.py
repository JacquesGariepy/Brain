"""
Complete Training Example for SOTA Brain

Shows how to:
- Load pre-trained models
- Create datasets
- Train models
- Evaluate models
- Save checkpoints
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sota_brain import SOTABrain, SOTABrainConfig
from architectures.optimization.optimizers import AdamW, Lion
from tqdm import tqdm


class DummyTextDataset(Dataset):
    """
    Example text dataset for language modeling.
    Replace with your actual dataset.
    """

    def __init__(self, num_samples=1000, seq_len=128, vocab_size=1000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random data (replace with real data)
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        labels = torch.randint(0, self.vocab_size, (self.seq_len,))
        return input_ids, labels


def train_language_model():
    """
    Complete example: Train a language model from scratch.
    """
    print("=" * 80)
    print("Training Language Model Example")
    print("=" * 80)

    # 1. Configuration
    config = SOTABrainConfig(
        use_language=True,
        use_vision=False,
        language_model="transformer",  # or "mamba"
        d_model=256,
        num_layers=6,
        num_heads=8,
        vocab_size=1000,
        max_seq_len=128,
        optimizer="adamw",
        learning_rate=3e-4,
        use_flash_attention=False,  # Set True if PyTorch 2.0+
        gradient_checkpointing=False,
        use_lora=False  # Set True for efficient fine-tuning
    )

    print(f"\nConfiguration:")
    print(f"- Model: {config.language_model}")
    print(f"- d_model: {config.d_model}")
    print(f"- Layers: {config.num_layers}")
    print(f"- Heads: {config.num_heads}")
    print(f"- Vocab size: {config.vocab_size}")
    print(f"- Optimizer: {config.optimizer}")

    # 2. Create model
    print("\nInitializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    brain = SOTABrain(config)
    brain = brain.to(device)

    total_params = sum(p.numel() for p in brain.parameters())
    trainable_params = sum(p.numel() for p in brain.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 3. Create dataset and dataloader
    print("\nCreating dataset...")
    train_dataset = DummyTextDataset(
        num_samples=1000,
        seq_len=config.max_seq_len,
        vocab_size=config.vocab_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )

    # 4. Optimizer and scheduler
    optimizer = brain.configure_optimizer()

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader) * 10  # 10 epochs
    )

    # 5. Training loop
    print("\nStarting training...")
    num_epochs = 3  # For demo
    brain.train()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Training")

        for batch_idx, (input_ids, labels) in enumerate(progress_bar):
            # Move to device
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            logits = brain(text_input=input_ids)

            # Compute loss
            loss = nn.functional.cross_entropy(
                logits.view(-1, config.vocab_size),
                labels.view(-1)
            )

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(brain.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            # Track loss
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

            # Log every N steps
            if (batch_idx + 1) % 50 == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                lr = optimizer.param_groups[0]['lr']
                print(f"\nStep {batch_idx + 1}: Loss = {avg_loss:.4f}, LR = {lr:.6f}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch + 1} Average Loss: {avg_epoch_loss:.4f}")

    # 6. Save model
    print("\nSaving model...")
    save_path = "checkpoints/sota_brain_lm.pt"
    os.makedirs("checkpoints", exist_ok=True)

    torch.save({
        'epoch': num_epochs,
        'model_state_dict': brain.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'loss': avg_epoch_loss,
    }, save_path)

    print(f"Model saved to {save_path}")

    # 7. Evaluation example
    print("\nEvaluation example...")
    brain.eval()

    with torch.no_grad():
        test_input = torch.randint(0, config.vocab_size, (1, 20)).to(device)
        output = brain(text_input=test_input)
        predictions = output.argmax(dim=-1)

        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Predictions: {predictions[0, :10].tolist()}")

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)


def load_pretrained_huggingface():
    """
    Example: Load a pre-trained model from Hugging Face and fine-tune.
    """
    print("\n" + "=" * 80)
    print("Loading Pre-trained Model Example")
    print("=" * 80)

    try:
        from transformers import AutoModel, AutoTokenizer

        print("\nLoading GPT-2 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        print("Loading GPT-2 model...")
        pretrained_model = AutoModel.from_pretrained("gpt2")

        print(f"\nPre-trained model loaded!")
        print(f"Model type: {type(pretrained_model)}")
        print(f"Total parameters: {sum(p.numel() for p in pretrained_model.parameters()):,}")

        # You can now:
        # 1. Use it directly
        # 2. Extract embeddings
        # 3. Fine-tune on your task

        print("\nExample tokenization:")
        text = "Hello, how are you?"
        tokens = tokenizer.encode(text)
        print(f"Text: {text}")
        print(f"Tokens: {tokens}")

    except ImportError:
        print("\nTransformers library not installed.")
        print("Install with: pip install transformers")


def fine_tune_with_lora():
    """
    Example: Fine-tune with LoRA (efficient adaptation).
    """
    print("\n" + "=" * 80)
    print("Fine-tuning with LoRA Example")
    print("=" * 80)

    # Config with LoRA
    config = SOTABrainConfig(
        use_language=True,
        language_model="transformer",
        d_model=256,
        num_layers=4,
        num_heads=8,
        vocab_size=1000,
        use_lora=True,  # Enable LoRA
        lora_rank=8,    # Low rank
        optimizer="adamw"
    )

    brain = SOTABrain(config)

    # Freeze base model, only train LoRA parameters
    for name, param in brain.named_parameters():
        if 'lora' not in name.lower():
            param.requires_grad = False

    trainable = sum(p.numel() for p in brain.parameters() if p.requires_grad)
    total = sum(p.numel() for p in brain.parameters())

    print(f"\nLoRA Fine-tuning:")
    print(f"Total parameters: {total:,}")
    print(f"Trainable (LoRA) parameters: {trainable:,}")
    print(f"Reduction: {(1 - trainable/total) * 100:.1f}%")

    # Training would proceed normally, but much faster!


if __name__ == "__main__":
    # Run examples
    print("\nSOTA Brain Training Examples\n")

    # Example 1: Train from scratch
    train_language_model()

    # Example 2: Load pre-trained (if transformers available)
    # load_pretrained_huggingface()

    # Example 3: LoRA fine-tuning
    # fine_tune_with_lora()

    print("\n✓ All examples completed successfully!")
