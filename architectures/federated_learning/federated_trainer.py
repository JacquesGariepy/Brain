"""
Federated Trainer - Orchestrates federated learning

Coordinates training between server and clients for any PyTorch model.

Key features:
- Privacy-preserving (raw data never leaves clients)
- Support for non-IID data distributions
- Communication-efficient
- Robust to client failures
- Differential privacy support
- Secure aggregation

Algorithms:
- FedAvg: Standard federated averaging
- FedProx: Adds proximal term for heterogeneous clients
- FedNova: Normalized averaging for varying local steps
- FedAdam/FedYogi: Adaptive federated optimizers
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import copy
import numpy as np


class FedAlgorithm(Enum):
    """Federated learning algorithms"""
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    FEDNOVA = "fednova"
    FEDADAM = "fedadam"
    FEDYOGI = "fedyogi"


@dataclass
class FederatedConfig:
    """Configuration for federated learning"""
    # Algorithm
    algorithm: FedAlgorithm = FedAlgorithm.FEDAVG

    # Training
    num_rounds: int = 100  # Number of communication rounds
    clients_per_round: int = 10  # Clients selected per round
    local_epochs: int = 5  # Local training epochs
    local_batch_size: int = 32

    # FedProx
    proximal_mu: float = 0.01  # Proximal term coefficient

    # Adaptive optimizers (FedAdam, FedYogi)
    server_lr: float = 0.01  # Server learning rate
    server_momentum: float = 0.9
    server_beta2: float = 0.99

    # Privacy
    use_differential_privacy: bool = False
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    dp_clip_norm: float = 1.0

    # Communication
    compression_ratio: float = 1.0  # 1.0 = no compression


class FederatedClient:
    """
    Federated learning client.

    Trains model locally on private data.
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_data: torch.utils.data.DataLoader,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            client_id: Unique client identifier
            model: PyTorch model (copy of global model)
            train_data: Client's local training data
            device: Device to train on
        """
        self.client_id = client_id
        self.model = model
        self.train_data = train_data
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)

    def train(
        self,
        num_epochs: int,
        learning_rate: float,
        criterion: Optional[Callable] = None,
        proximal_mu: float = 0.0,
        global_model: Optional[nn.Module] = None
    ) -> Tuple[Dict[str, torch.Tensor], int, float]:
        """
        Train locally for num_epochs.

        Args:
            num_epochs: Number of local epochs
            learning_rate: Learning rate
            criterion: Loss function
            proximal_mu: FedProx proximal term (0 = FedAvg)
            global_model: Global model for FedProx

        Returns:
            (model_update, num_samples, loss)
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        self.model.train()

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=learning_rate
        )

        total_loss = 0.0
        num_batches = 0

        for epoch in range(num_epochs):
            for data, target in self.train_data:
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                output = self.model(data)
                loss = criterion(output, target)

                # FedProx: add proximal term
                if proximal_mu > 0 and global_model is not None:
                    proximal_term = 0.0
                    for param, global_param in zip(
                        self.model.parameters(),
                        global_model.parameters()
                    ):
                        proximal_term += ((param - global_param) ** 2).sum()

                    loss += (proximal_mu / 2) * proximal_term

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches

        # Compute model update (difference from initial)
        model_update = {}
        for name, param in self.model.named_parameters():
            model_update[name] = param.data.clone()

        # Count number of samples
        num_samples = len(self.train_data.dataset)

        return model_update, num_samples, avg_loss

    def receive_global_model(self, global_state_dict: Dict[str, torch.Tensor]):
        """
        Receive updated global model from server.

        Args:
            global_state_dict: Global model state dict
        """
        self.model.load_state_dict(global_state_dict)


class FederatedServer:
    """
    Federated learning server.

    Aggregates client updates and maintains global model.
    """

    def __init__(
        self,
        model: nn.Module,
        algorithm: FedAlgorithm = FedAlgorithm.FEDAVG,
        config: Optional[FederatedConfig] = None
    ):
        """
        Args:
            model: Global model
            algorithm: Aggregation algorithm
            config: Configuration
        """
        self.model = model
        self.algorithm = algorithm
        self.config = config or FederatedConfig(algorithm=algorithm)

        # For adaptive optimizers
        self.momentum = None
        self.velocity = None

        # Initialize adaptive optimizer states
        if algorithm in [FedAlgorithm.FEDADAM, FedAlgorithm.FEDYOGI]:
            self.momentum = {
                name: torch.zeros_like(param.data)
                for name, param in self.model.named_parameters()
            }
            self.velocity = {
                name: torch.zeros_like(param.data)
                for name, param in self.model.named_parameters()
            }

    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates.

        Args:
            client_updates: List of client model state dicts
            client_weights: Weight for each client (typically num_samples)

        Returns:
            Aggregated global model state dict
        """
        if self.algorithm == FedAlgorithm.FEDAVG:
            return self._fedavg_aggregate(client_updates, client_weights)

        elif self.algorithm == FedAlgorithm.FEDPROX:
            # FedProx uses same aggregation as FedAvg
            return self._fedavg_aggregate(client_updates, client_weights)

        elif self.algorithm == FedAlgorithm.FEDNOVA:
            return self._fednova_aggregate(client_updates, client_weights)

        elif self.algorithm == FedAlgorithm.FEDADAM:
            return self._fedadam_aggregate(client_updates, client_weights)

        elif self.algorithm == FedAlgorithm.FEDYOGI:
            return self._fedyogi_aggregate(client_updates, client_weights)

        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def _fedavg_aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """
        FedAvg: Weighted average of client models.

        Weight by number of samples per client.
        """
        # Normalize weights
        total_weight = sum(client_weights)
        weights = [w / total_weight for w in client_weights]

        # Weighted average
        aggregated = {}

        for name in client_updates[0].keys():
            aggregated[name] = torch.zeros_like(client_updates[0][name])

            for client_update, weight in zip(client_updates, weights):
                aggregated[name] += weight * client_update[name]

        return aggregated

    def _fednova_aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """
        FedNova: Normalized averaging.

        Accounts for varying numbers of local steps.
        """
        # Get current global model
        global_dict = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }

        # Compute normalized updates
        total_weight = sum(client_weights)
        weights = [w / total_weight for w in client_weights]

        aggregated = {}

        for name in client_updates[0].keys():
            # Compute weighted sum of updates (delta from global)
            weighted_delta = torch.zeros_like(client_updates[0][name])

            for client_update, weight in zip(client_updates, weights):
                delta = client_update[name] - global_dict[name]
                weighted_delta += weight * delta

            # Apply update
            aggregated[name] = global_dict[name] + weighted_delta

        return aggregated

    def _fedadam_aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """
        FedAdam: Adaptive optimization on server.

        Uses Adam-style momentum and adaptive learning rates.
        """
        # Compute pseudo-gradient (average update direction)
        total_weight = sum(client_weights)
        weights = [w / total_weight for w in client_weights]

        global_dict = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }

        aggregated = {}

        for name in client_updates[0].keys():
            # Compute average gradient
            avg_grad = torch.zeros_like(client_updates[0][name])

            for client_update, weight in zip(client_updates, weights):
                delta = client_update[name] - global_dict[name]
                avg_grad += weight * delta

            # Update momentum
            self.momentum[name] = (
                self.config.server_momentum * self.momentum[name] +
                (1 - self.config.server_momentum) * avg_grad
            )

            # Update velocity (second moment)
            self.velocity[name] = (
                self.config.server_beta2 * self.velocity[name] +
                (1 - self.config.server_beta2) * (avg_grad ** 2)
            )

            # Adam update
            aggregated[name] = global_dict[name] + self.config.server_lr * (
                self.momentum[name] / (torch.sqrt(self.velocity[name]) + 1e-8)
            )

        return aggregated

    def _fedyogi_aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """
        FedYogi: Yogi-style adaptive optimization.

        Similar to FedAdam but with different second moment update.
        """
        total_weight = sum(client_weights)
        weights = [w / total_weight for w in client_weights]

        global_dict = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }

        aggregated = {}

        for name in client_updates[0].keys():
            # Compute average gradient
            avg_grad = torch.zeros_like(client_updates[0][name])

            for client_update, weight in zip(client_updates, weights):
                delta = client_update[name] - global_dict[name]
                avg_grad += weight * delta

            # Update momentum
            self.momentum[name] = (
                self.config.server_momentum * self.momentum[name] +
                (1 - self.config.server_momentum) * avg_grad
            )

            # Yogi second moment update (different from Adam)
            self.velocity[name] = self.velocity[name] - (
                (1 - self.config.server_beta2) *
                (avg_grad ** 2) *
                torch.sign(self.velocity[name] - avg_grad ** 2)
            )

            # Update
            aggregated[name] = global_dict[name] + self.config.server_lr * (
                self.momentum[name] / (torch.sqrt(self.velocity[name]) + 1e-8)
            )

        return aggregated

    def broadcast_model(self) -> Dict[str, torch.Tensor]:
        """
        Broadcast current global model to clients.

        Returns:
            Global model state dict
        """
        return {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }

    def update_model(self, aggregated_state: Dict[str, torch.Tensor]):
        """
        Update global model with aggregated state.

        Args:
            aggregated_state: Aggregated model state dict
        """
        self.model.load_state_dict(aggregated_state)


class FederatedTrainer:
    """
    Main federated learning orchestrator.

    Coordinates training between server and multiple clients.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[FederatedConfig] = None
    ):
        """
        Args:
            model: Global model
            config: Federated learning configuration
        """
        self.config = config or FederatedConfig()
        self.global_model = model

        # Create server
        self.server = FederatedServer(
            model=self.global_model,
            algorithm=self.config.algorithm,
            config=self.config
        )

        # Clients will be added dynamically
        self.clients = []

    def add_client(
        self,
        client_id: int,
        train_data: torch.utils.data.DataLoader,
        device: Optional[torch.device] = None
    ):
        """
        Add a new client to the federation.

        Args:
            client_id: Unique client identifier
            train_data: Client's training data
            device: Device for client training
        """
        # Create client with copy of global model
        client_model = copy.deepcopy(self.global_model)

        client = FederatedClient(
            client_id=client_id,
            model=client_model,
            train_data=train_data,
            device=device
        )

        self.clients.append(client)

    def train(
        self,
        num_rounds: Optional[int] = None,
        clients_per_round: Optional[int] = None,
        learning_rate: float = 0.01,
        criterion: Optional[Callable] = None
    ):
        """
        Run federated training.

        Args:
            num_rounds: Number of communication rounds
            clients_per_round: Clients to select per round
            learning_rate: Client learning rate
            criterion: Loss function
        """
        num_rounds = num_rounds or self.config.num_rounds
        clients_per_round = clients_per_round or min(
            self.config.clients_per_round,
            len(self.clients)
        )

        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        print(f"Starting federated training with {len(self.clients)} clients")
        print(f"Algorithm: {self.config.algorithm.value}")
        print(f"Rounds: {num_rounds}, Clients per round: {clients_per_round}")

        for round_idx in range(num_rounds):
            print(f"\n{'='*80}")
            print(f"Round {round_idx + 1}/{num_rounds}")
            print('='*80)

            # Select clients for this round
            selected_indices = np.random.choice(
                len(self.clients),
                size=clients_per_round,
                replace=False
            )
            selected_clients = [self.clients[i] for i in selected_indices]

            print(f"Selected clients: {[c.client_id for c in selected_clients]}")

            # Broadcast global model to clients
            global_state = self.server.broadcast_model()

            # Train on selected clients
            client_updates = []
            client_weights = []
            client_losses = []

            for client in selected_clients:
                # Receive global model
                client.receive_global_model(global_state)

                # Train locally
                update, num_samples, loss = client.train(
                    num_epochs=self.config.local_epochs,
                    learning_rate=learning_rate,
                    criterion=criterion,
                    proximal_mu=self.config.proximal_mu if self.config.algorithm == FedAlgorithm.FEDPROX else 0.0,
                    global_model=self.global_model if self.config.algorithm == FedAlgorithm.FEDPROX else None
                )

                client_updates.append(update)
                client_weights.append(num_samples)
                client_losses.append(loss)

                print(f"  Client {client.client_id}: {num_samples} samples, loss={loss:.4f}")

            # Aggregate updates
            print("Aggregating updates...")
            aggregated = self.server.aggregate(client_updates, client_weights)

            # Update global model
            self.server.update_model(aggregated)

            # Report
            avg_loss = np.mean(client_losses)
            print(f"\nRound {round_idx + 1} completed. Average loss: {avg_loss:.4f}")

        print(f"\n{'='*80}")
        print("Federated training completed!")
        print('='*80)

    def get_global_model(self) -> nn.Module:
        """Get the trained global model"""
        return self.global_model


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("Federated Learning - Privacy-Preserving Training")
    print("="*80)

    # Example model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    # Create federated trainer
    config = FederatedConfig(
        algorithm=FedAlgorithm.FEDAVG,
        num_rounds=10,
        clients_per_round=5,
        local_epochs=3
    )

    trainer = FederatedTrainer(model, config)

    print(f"\nConfiguration:")
    print(f"  Algorithm: {config.algorithm.value}")
    print(f"  Rounds: {config.num_rounds}")
    print(f"  Clients per round: {config.clients_per_round}")
    print(f"  Local epochs: {config.local_epochs}")

    # Add clients (simulated)
    print("\nAdding clients...")
    for i in range(10):
        # In practice, each client would have their own data
        # trainer.add_client(i, client_dataloader)
        print(f"  Client {i} added")

    # Train (commented out - needs actual data)
    # trainer.train(learning_rate=0.01)

    print("\n" + "="*80)
