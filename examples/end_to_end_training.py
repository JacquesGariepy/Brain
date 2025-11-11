"""
Brain Framework - End-to-End Training Example

Demonstrates complete integration of:
- Data loading (utils.data)
- Architecture (from architectures/)
- Logging (utils.logging)
- Metrics (utils.metrics)
- Training loop

This shows how ALL components work together.
"""

import sys
sys.path.insert(0, '..')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
except ImportError:
    print("PyTorch required: pip install torch")
    sys.exit(1)


def example_vision_classification():
    """
    Complete example: Image classification with ResNet on CIFAR-10
    """
    print("\n" + "=" * 70)
    print("End-to-End Training - Vision Classification (ResNet + CIFAR-10)")
    print("=" * 70)

    # 1. Load Data (utils.data)
    print("\n[1/6] Loading data...")
    from utils.data import get_cifar10_loaders, get_train_augmentation

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=64,
        augment=True,
        download=True
    )
    print(f"   ✓ Train batches: {len(train_loader)}")
    print(f"   ✓ Test batches: {len(test_loader)}")

    # 2. Initialize Architecture
    print("\n[2/6] Initializing model...")
    from architectures.computer_vision.resnet import ResNet

    model = ResNet(
        num_classes=10,
        block_type='basic',
        num_blocks=[2, 2, 2, 2],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"   ✓ Model: ResNet")
    print(f"   ✓ Device: {device}")
    print(f"   ✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Setup Logging (utils.logging)
    print("\n[3/6] Setting up logging...")
    from utils.logging import WandBLogger, TensorBoardLogger

    # WandB (optional - requires API key)
    try:
        wandb_logger = WandBLogger(
            project="brain-cifar10",
            name="resnet-example",
            config={"batch_size": 64, "lr": 0.001},
            mode="offline"  # Use "online" if you have API key
        )
        print("   ✓ WandB logger initialized")
    except:
        wandb_logger = None
        print("   - WandB not available (optional)")

    # TensorBoard
    tensorboard_logger = TensorBoardLogger(log_dir="./runs/cifar10")
    print("   ✓ TensorBoard logger initialized")

    # 4. Setup Metrics (utils.metrics)
    print("\n[4/6] Setting up metrics...")
    from utils.metrics import MetricsTracker, accuracy, compute_metrics

    tracker = MetricsTracker()
    print("   ✓ Metrics tracker initialized")

    # 5. Setup Training
    print("\n[5/6] Setting up training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("   ✓ Loss: CrossEntropyLoss")
    print("   ✓ Optimizer: Adam")

    # 6. Training Loop
    print("\n[6/6] Training...")
    num_epochs = 3

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward
            loss.backward()
            optimizer.step()

            # Metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            # Log every 50 batches
            if batch_idx % 50 == 0:
                step = epoch * len(train_loader) + batch_idx

                # TensorBoard
                tensorboard_logger.log_scalar("train/loss", loss.item(), step)

                # WandB
                if wandb_logger:
                    wandb_logger.log_metrics({"train/loss": loss.item()}, step=step)

                print(f"   Epoch {epoch+1}/{num_epochs} [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")

        # Epoch metrics
        train_acc = 100. * train_correct / train_total
        train_loss_avg = train_loss / len(train_loader)

        print(f"\n   Epoch {epoch+1} Summary:")
        print(f"      Train Loss: {train_loss_avg:.4f}")
        print(f"      Train Acc:  {train_acc:.2f}%")

        # Validation
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_acc = 100. * test_correct / test_total
        test_loss_avg = test_loss / len(test_loader)

        print(f"      Test Loss:  {test_loss_avg:.4f}")
        print(f"      Test Acc:   {test_acc:.2f}%\n")

        # Log epoch metrics
        epoch_metrics = {
            "train/loss": train_loss_avg,
            "train/accuracy": train_acc,
            "test/loss": test_loss_avg,
            "test/accuracy": test_acc,
        }

        tracker.update(epoch_metrics)

        tensorboard_logger.log_scalars("metrics", epoch_metrics, epoch)

        if wandb_logger:
            wandb_logger.log_metrics(epoch_metrics, step=epoch)

    # 7. Final Evaluation
    print("\n" + "=" * 70)
    print("Final Results")
    print("=" * 70)

    final_metrics = tracker.get_all_averages()
    print(f"\n{tracker.summary()}")

    # Close loggers
    tensorboard_logger.close()
    if wandb_logger:
        wandb_logger.finish()

    print("\n✓ Training completed!")
    print(f"   TensorBoard logs: ./runs/cifar10")
    print(f"   Run: tensorboard --logdir=./runs/cifar10")


def example_text_classification():
    """
    Complete example: Text classification with Transformer
    """
    print("\n" + "=" * 70)
    print("End-to-End Training - Text Classification (Transformer)")
    print("=" * 70)

    print("\nThis example would:")
    print("   1. Load text dataset (GLUE SST-2)")
    print("   2. Initialize Transformer model")
    print("   3. Setup WandB/MLflow logging")
    print("   4. Train with metrics tracking")
    print("   5. Evaluate and save model")
    print("\nSee vision example above for complete implementation pattern.")


def main():
    """Run all examples"""
    print("=" * 70)
    print("Brain Framework - End-to-End Training Examples")
    print("=" * 70)
    print("\nDemonstrating complete integration of all components")

    try:
        # Run vision example (works without additional dependencies)
        example_vision_classification()

        # Text example (requires transformers)
        example_text_classification()

        print("\n" + "=" * 70)
        print("✓ All end-to-end examples completed!")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
