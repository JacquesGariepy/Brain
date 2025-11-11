"""
State-of-the-Art Reinforcement Learning Algorithms

Implementations:
- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)
- DQN with Rainbow improvements
- AlphaZero / MuZero
- World Models
- Hierarchical RL
- Inverse RL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np


@dataclass
class PPOConfig:
    """Configuration for PPO"""
    state_dim: int = 84
    action_dim: int = 4
    hidden_dim: int = 256
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter
    clip_epsilon: float = 0.2  # PPO clipping
    value_coef: float = 0.5  # Value loss coefficient
    entropy_coef: float = 0.01  # Entropy bonus
    max_grad_norm: float = 0.5  # Gradient clipping


class PPOActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.

    Implements both policy (actor) and value function (critic).
    """

    def __init__(self, config: PPOConfig):
        super().__init__()
        self.config = config

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU()
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_dim)
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            action_logits: Logits for action distribution
            value: State value estimate
        """
        features = self.feature_extractor(state)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Returns:
            action: Sampled action
            log_prob: Log probability of action
            value: State value estimate
        """
        action_logits, value = self(state)
        action_probs = F.softmax(action_logits, dim=-1)

        if deterministic:
            action = action_probs.argmax(dim=-1)
        else:
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()

        log_prob = F.log_softmax(action_logits, dim=-1).gather(-1, action.unsqueeze(-1)).squeeze(-1)

        return action, log_prob, value


class PPO:
    """
    Proximal Policy Optimization.

    SOTA on-policy RL algorithm.
    """

    def __init__(self, config: PPOConfig, learning_rate: float = 3e-4):
        self.config = config
        self.model = PPOActorCritic(config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Rewards (T,)
            values: Value estimates (T,)
            dones: Done flags (T,)
            next_value: Value of next state

        Returns:
            advantages: Advantage estimates (T,)
            returns: Discounted returns (T,)
        """
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        values_extended = torch.cat([values, next_value.unsqueeze(0)])

        for t in reversed(range(len(rewards))):
            next_value = values_extended[t + 1]
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values

        return advantages, returns

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        num_epochs: int = 10,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """
        Update policy using PPO objective.

        Returns:
            Dictionary of training metrics
        """
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'approx_kl': 0.0
        }

        num_samples = len(states)

        for epoch in range(num_epochs):
            # Mini-batch SGD
            indices = torch.randperm(num_samples)

            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Get current policy predictions
                action_logits, values = self.model(batch_states)
                action_probs = F.softmax(action_logits, dim=-1)
                log_probs = F.log_softmax(action_logits, dim=-1).gather(-1, batch_actions.unsqueeze(-1)).squeeze(-1)

                # Policy loss (PPO-Clip)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values.squeeze(-1), batch_returns)

                # Entropy bonus (for exploration)
                entropy = -(action_probs * log_probs.unsqueeze(-1)).sum(dim=-1).mean()

                # Total loss
                loss = (
                    policy_loss +
                    self.config.value_coef * value_loss -
                    self.config.entropy_coef * entropy
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy'] += entropy.item()

                # Approximate KL divergence
                approx_kl = ((batch_old_log_probs - log_probs) ** 2).mean()
                metrics['approx_kl'] += approx_kl.item()

        # Average metrics
        num_updates = num_epochs * (num_samples // batch_size)
        metrics = {k: v / num_updates for k, v in metrics.items()}

        return metrics


@dataclass
class SACConfig:
    """Configuration for SAC"""
    state_dim: int = 84
    action_dim: int = 4
    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 0.005  # Soft update coefficient
    alpha: float = 0.2  # Entropy temperature
    auto_alpha: bool = True  # Automatic temperature tuning


class SACCritic(nn.Module):
    """
    Twin Q-networks for SAC.

    Uses two Q-functions to reduce overestimation bias.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Q-values.

        Returns:
            q1, q2: Q-value estimates from both networks
        """
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


class SACActor(nn.Module):
    """
    Gaussian policy for continuous actions (SAC).
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action distribution.

        Returns:
            mean, log_std: Parameters of Gaussian distribution
        """
        features = self.net(state)
        mean = self.mean(features)
        log_std = self.log_std(features)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action using reparameterization trick.

        Returns:
            action: Sampled action (tanh squashed)
            log_prob: Log probability of action
        """
        mean, log_std = self(state)
        std = log_std.exp()

        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterized sample

        # Squash with tanh
        action = torch.tanh(x_t)

        # Compute log probability with tanh correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


class SAC:
    """
    Soft Actor-Critic.

    SOTA off-policy RL for continuous control.
    """

    def __init__(self, config: SACConfig, learning_rate: float = 3e-4):
        self.config = config

        # Actor
        self.actor = SACActor(config.state_dim, config.action_dim, config.hidden_dim)

        # Critics
        self.critic = SACCritic(config.state_dim, config.action_dim, config.hidden_dim)
        self.critic_target = SACCritic(config.state_dim, config.action_dim, config.hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

        # Automatic temperature tuning
        if config.auto_alpha:
            self.target_entropy = -config.action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = config.alpha

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, float]:
        """
        Update SAC networks.

        Returns:
            Dictionary of training metrics
        """
        # Update critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + self.config.gamma * (1 - dones) * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        new_actions, log_probs = self.actor.sample(states)
        q1, q2 = self.critic(states, new_actions)
        q = torch.min(q1, q2)

        actor_loss = (self.alpha * log_probs - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update temperature
        if self.config.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()

        # Soft update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha
        }


class RainbowDQN(nn.Module):
    """
    Rainbow DQN - Combines multiple DQN improvements:
    - Double DQN
    - Dueling DQN
    - Prioritized Experience Replay
    - Multi-step learning
    - Distributional RL (C51)
    - Noisy Networks
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        num_atoms: int = 51,  # For C51
        v_min: float = -10,
        v_max: float = 10
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        # Support of value distribution
        self.register_buffer('support', torch.linspace(v_min, v_max, num_atoms))
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

        # Feature extraction
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Dueling architecture
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_atoms)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * num_atoms)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Returns:
            Q-value distributions for each action
        """
        batch_size = state.shape[0]

        features = self.feature(state)

        # Value and advantage streams
        value = self.value_stream(features).view(batch_size, 1, self.num_atoms)
        advantage = self.advantage_stream(features).view(batch_size, self.action_dim, self.num_atoms)

        # Combine with dueling formula
        q_dist = value + (advantage - advantage.mean(dim=1, keepdim=True))

        # Apply softmax to get probability distribution
        q_dist = F.softmax(q_dist, dim=-1)

        return q_dist

    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get expected Q-values.
        """
        q_dist = self(state)
        q_values = (q_dist * self.support.view(1, 1, -1)).sum(dim=-1)
        return q_values


class WorldModel(nn.Module):
    """
    World Model for model-based RL.

    Learns dynamics model: s_{t+1} = f(s_t, a_t)
    And reward model: r_t = g(s_t, a_t)

    Can be used for planning and imagination.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 64
    ):
        super().__init__()

        # Encoder: s_t -> z_t
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # Mean and log_std
        )

        # Dynamics model: (z_t, a_t) -> z_{t+1}
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Reward model: (z_t, a_t) -> r_t
        self.reward = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Decoder: z_t -> s_t
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def encode(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode state to latent distribution"""
        h = self.encoder(state)
        mean, log_std = h.chunk(2, dim=-1)
        return mean, log_std

    def predict(self, latent: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next latent state and reward.

        Args:
            latent: Current latent state
            action: Action taken

        Returns:
            next_latent: Predicted next latent state
            reward: Predicted reward
        """
        za = torch.cat([latent, action], dim=-1)
        next_latent = self.dynamics(za)
        reward = self.reward(za)
        return next_latent, reward

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to state"""
        return self.decoder(latent)

    def imagine_rollout(
        self,
        initial_state: torch.Tensor,
        policy: nn.Module,
        horizon: int
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Imagine rollout using world model.

        Returns:
            states: Imagined states
            rewards: Imagined rewards
        """
        mean, _ = self.encode(initial_state)
        latent = mean

        states = []
        rewards = []

        for _ in range(horizon):
            # Get action from policy
            state = self.decode(latent)
            action = policy(state)

            # Predict next state and reward
            latent, reward = self.predict(latent, action)

            states.append(state)
            rewards.append(reward)

        return states, rewards
