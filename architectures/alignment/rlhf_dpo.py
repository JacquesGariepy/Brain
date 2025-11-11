"""
RLHF and DPO - Alignment Techniques for Safe and Helpful LLMs

Aligns language models with human preferences and values.

Key Techniques:
- RLHF: Reinforcement Learning from Human Feedback
- DPO: Direct Preference Optimization (no RL needed!)
- Constitutional AI: Self-critique and revision
- Safe RLHF: Separate helpfulness and harmlessness rewards

References:
- InstructGPT (RLHF): https://arxiv.org/abs/2203.02155
- DPO: https://arxiv.org/abs/2305.18290
- Constitutional AI: https://arxiv.org/abs/2212.08073
- Safe RLHF: https://arxiv.org/abs/2310.12773
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
import math


# ============================================================================
# Reward Model
# ============================================================================

@dataclass
class RewardModelConfig:
    """Configuration for reward model"""
    d_model: int = 768
    num_layers: int = 6  # Smaller than policy model
    dropout: float = 0.1


class RewardModel(nn.Module):
    """
    Reward Model for RLHF

    Takes (prompt, completion) and outputs scalar reward.
    Trained on human preference data: "Which completion is better?"

    Example:
        >>> config = RewardModelConfig(d_model=768)
        >>> reward_model = RewardModel(config, base_model)
        >>>
        >>> # Score completions
        >>> prompt = "Write a poem"
        >>> completion_a = "Roses are red..."
        >>> completion_b = "The moonlight dances..."
        >>>
        >>> reward_a = reward_model(prompt, completion_a)
        >>> reward_b = reward_model(prompt, completion_b)
        >>> # reward_b > reward_a means completion_b is preferred
    """

    def __init__(self, config: RewardModelConfig, base_model: nn.Module):
        super().__init__()
        self.config = config
        self.base_model = base_model

        # Freeze base model initially (optional)
        # for param in self.base_model.parameters():
        #     param.requires_grad = False

        # Reward head (scalar output)
        self.reward_head = nn.Linear(config.d_model, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute reward for input.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]

        Returns:
            rewards: [batch] scalar reward per sequence
        """
        # Get last hidden state from base model
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [batch, seq, d_model]

        # Get last non-padding token
        if attention_mask is not None:
            # Find last non-padding position
            seq_lengths = attention_mask.sum(dim=1) - 1
            last_hidden = last_hidden[torch.arange(last_hidden.size(0)), seq_lengths]
        else:
            last_hidden = last_hidden[:, -1]  # [batch, d_model]

        # Compute scalar reward
        reward = self.reward_head(last_hidden).squeeze(-1)  # [batch]

        return reward

    def compute_pairwise_loss(
        self,
        input_ids_chosen: torch.Tensor,
        input_ids_rejected: torch.Tensor,
        attention_mask_chosen: Optional[torch.Tensor] = None,
        attention_mask_rejected: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Bradley-Terry loss for preference pairs.

        Loss = -log(sigmoid(r_chosen - r_rejected))

        Args:
            input_ids_chosen: Preferred completion
            input_ids_rejected: Rejected completion

        Returns:
            loss: Scalar loss
        """
        # Get rewards
        r_chosen = self.forward(input_ids_chosen, attention_mask_chosen)
        r_rejected = self.forward(input_ids_rejected, attention_mask_rejected)

        # Bradley-Terry loss
        loss = -F.logsigmoid(r_chosen - r_rejected).mean()

        return loss


# ============================================================================
# PPO for RLHF
# ============================================================================

@dataclass
class PPOConfig:
    """Configuration for PPO"""
    # PPO hyperparameters
    clip_eps: float = 0.2  # Clipping epsilon
    vf_coef: float = 0.1  # Value function coefficient
    ent_coef: float = 0.01  # Entropy coefficient
    max_grad_norm: float = 1.0  # Gradient clipping

    # Training
    num_epochs: int = 4  # PPO epochs per batch
    batch_size: int = 64
    mini_batch_size: int = 16

    # KL penalty (prevent divergence from reference model)
    kl_coef: float = 0.1
    target_kl: Optional[float] = 0.01  # Early stopping if KL > target


class PPOTrainer:
    """
    PPO Trainer for RLHF

    Three models:
    1. Policy model (being trained)
    2. Reference model (frozen, for KL penalty)
    3. Reward model (frozen, provides rewards)

    Training loop:
        1. Generate completions with policy
        2. Score with reward model
        3. Compute advantages
        4. Update policy with PPO

    Example:
        >>> config = PPOConfig(clip_eps=0.2)
        >>> trainer = PPOTrainer(
        ...     policy_model=model,
        ...     ref_model=ref_model,
        ...     reward_model=reward_model,
        ...     config=config
        ... )
        >>>
        >>> for prompts in dataloader:
        ...     metrics = trainer.step(prompts)
        ...     print(f"Reward: {metrics['reward']:.2f}, KL: {metrics['kl']:.4f}")
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        reward_model: nn.Module,
        config: PPOConfig
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.config = config

        # Freeze reference and reward models
        for param in self.ref_model.parameters():
            param.requires_grad = False
        for param in self.reward_model.parameters():
            param.requires_grad = False

    def generate_completions(
        self,
        prompts: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate completions from prompts.

        Returns:
            completions: [batch, prompt_len + max_new_tokens]
            log_probs: [batch, max_new_tokens] log probabilities
        """
        batch_size = prompts.size(0)
        completions = prompts.clone()
        log_probs_list = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get logits
                outputs = self.policy_model(completions)
                logits = outputs.logits[:, -1, :] / temperature

                # Sample token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Store log prob
                log_prob = F.log_softmax(logits, dim=-1)
                log_prob = log_prob.gather(dim=-1, index=next_token)
                log_probs_list.append(log_prob)

                # Append token
                completions = torch.cat([completions, next_token], dim=1)

        log_probs = torch.cat(log_probs_list, dim=1)  # [batch, max_new_tokens]

        return completions, log_probs

    def compute_rewards(
        self,
        prompts: torch.Tensor,
        completions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute rewards for completions.

        Returns:
            rewards: [batch, seq_len] reward per token
        """
        with torch.no_grad():
            # Get scalar reward for full sequence
            reward_score = self.reward_model(completions)

        # Distribute reward to last token (sparse reward)
        # In practice, might use reward shaping
        seq_len = completions.size(1)
        rewards = torch.zeros(completions.size(), device=completions.device)
        rewards[:, -1] = reward_score

        return rewards

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE (Generalized Advantage Estimation).

        Returns:
            advantages: [batch, seq_len]
            returns: [batch, seq_len]
        """
        batch_size, seq_len = rewards.shape

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        # Compute advantages using GAE
        last_gae = 0
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = 0
            else:
                next_value = values[:, t + 1]

            delta = rewards[:, t] + gamma * next_value - values[:, t]
            advantages[:, t] = last_gae = delta + gamma * lam * last_gae

        # Returns = advantages + values
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def compute_kl(
        self,
        log_probs_policy: torch.Tensor,
        log_probs_ref: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence between policy and reference"""
        # KL(policy || ref) = E[log(policy) - log(ref)]
        kl = log_probs_policy - log_probs_ref
        return kl.mean()

    def ppo_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute PPO loss.

        Returns:
            Dictionary with policy_loss, value_loss, entropy_loss
        """
        # Policy loss (clipped surrogate objective)
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(values, returns)

        # Entropy loss (encourage exploration)
        entropy = -(log_probs * torch.exp(log_probs)).sum(dim=-1).mean()
        entropy_loss = -entropy

        # Total loss
        loss = (
            policy_loss
            + self.config.vf_coef * value_loss
            + self.config.ent_coef * entropy_loss
        )

        return {
            'loss': loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss
        }


# ============================================================================
# DPO - Direct Preference Optimization
# ============================================================================

@dataclass
class DPOConfig:
    """Configuration for DPO"""
    beta: float = 0.1  # Temperature parameter (controls how much to optimize)
    label_smoothing: float = 0.0  # Label smoothing
    loss_type: str = "sigmoid"  # "sigmoid" or "hinge"


class DPOTrainer:
    """
    DPO Trainer - Direct Preference Optimization

    Key insight: Skip reward model and RL entirely!
    Directly optimize policy from preference data.

    Much simpler than RLHF:
    - No reward model needed
    - No PPO needed
    - Just supervised learning on preferences

    Loss: -log(sigmoid(beta * (log π_θ(y_w | x) - log π_ref(y_w | x)
                                 - log π_θ(y_l | x) + log π_ref(y_l | x))))

    Where:
    - y_w: chosen (winning) completion
    - y_l: rejected (losing) completion
    - π_θ: policy being trained
    - π_ref: reference policy (frozen)

    Example:
        >>> config = DPOConfig(beta=0.1)
        >>> trainer = DPOTrainer(policy_model, ref_model, config)
        >>>
        >>> # Training is just supervised learning!
        >>> for batch in dataloader:
        ...     loss = trainer.compute_loss(
        ...         prompts=batch['prompt'],
        ...         chosen=batch['chosen'],
        ...         rejected=batch['rejected']
        ...     )
        ...     loss.backward()
        ...     optimizer.step()
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        config: DPOConfig
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.config = config

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def get_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get log probabilities for labels.

        Args:
            model: Model to use
            input_ids: [batch, seq_len]
            labels: [batch, seq_len]
            attention_mask: [batch, seq_len]

        Returns:
            log_probs: [batch] average log prob per sequence
        """
        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [batch, seq_len, vocab_size]

        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Compute log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather log probs for actual labels
        log_probs = log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # [batch, seq_len-1]

        # Mask padding tokens
        if attention_mask is not None:
            mask = attention_mask[:, 1:].contiguous()
            log_probs = log_probs * mask

            # Average over sequence
            log_probs = log_probs.sum(dim=-1) / mask.sum(dim=-1)
        else:
            log_probs = log_probs.mean(dim=-1)

        return log_probs

    def compute_loss(
        self,
        input_ids_chosen: torch.Tensor,
        input_ids_rejected: torch.Tensor,
        attention_mask_chosen: Optional[torch.Tensor] = None,
        attention_mask_rejected: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute DPO loss.

        Args:
            input_ids_chosen: Preferred completions
            input_ids_rejected: Rejected completions

        Returns:
            loss: Scalar loss
            metrics: Dictionary of metrics
        """
        # Get log probs from policy
        policy_chosen_log_probs = self.get_log_probs(
            self.policy_model,
            input_ids_chosen,
            input_ids_chosen,
            attention_mask_chosen
        )
        policy_rejected_log_probs = self.get_log_probs(
            self.policy_model,
            input_ids_rejected,
            input_ids_rejected,
            attention_mask_rejected
        )

        # Get log probs from reference (frozen)
        with torch.no_grad():
            ref_chosen_log_probs = self.get_log_probs(
                self.ref_model,
                input_ids_chosen,
                input_ids_chosen,
                attention_mask_chosen
            )
            ref_rejected_log_probs = self.get_log_probs(
                self.ref_model,
                input_ids_rejected,
                input_ids_rejected,
                attention_mask_rejected
            )

        # Compute rewards (implicit)
        policy_rewards = policy_chosen_log_probs - policy_rejected_log_probs
        ref_rewards = ref_chosen_log_probs - ref_rejected_log_probs

        # DPO loss
        logits = self.config.beta * (policy_rewards - ref_rewards)

        if self.config.loss_type == "sigmoid":
            loss = -F.logsigmoid(logits).mean()
        elif self.config.loss_type == "hinge":
            loss = F.relu(1 - logits).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")

        # Metrics
        metrics = {
            'loss': loss.item(),
            'policy_chosen_log_probs': policy_chosen_log_probs.mean().item(),
            'policy_rejected_log_probs': policy_rejected_log_probs.mean().item(),
            'ref_chosen_log_probs': ref_chosen_log_probs.mean().item(),
            'ref_rejected_log_probs': ref_rejected_log_probs.mean().item(),
            'rewards/chosen': policy_rewards.mean().item(),
            'rewards/margins': policy_rewards.mean().item(),
            'accuracy': (logits > 0).float().mean().item()
        }

        return loss, metrics


# ============================================================================
# Constitutional AI
# ============================================================================

@dataclass
class ConstitutionalAIConfig:
    """Configuration for Constitutional AI"""
    constitution: List[str] = None  # List of principles
    num_critique_iterations: int = 1  # How many self-critique rounds
    critique_prompt_template: str = "Critique the following response based on this principle: {principle}\n\nResponse: {response}\n\nCritique:"
    revision_prompt_template: str = "Revise the following response based on this critique:\n\nOriginal: {response}\n\nCritique: {critique}\n\nRevised:"

    def __post_init__(self):
        if self.constitution is None:
            # Default constitution
            self.constitution = [
                "The response should be helpful and informative.",
                "The response should be harmless and avoid offensive content.",
                "The response should be honest and not contain misinformation.",
                "The response should respect privacy and not share personal information."
            ]


class ConstitutionalAI:
    """
    Constitutional AI - Self-Critique and Revision

    Instead of human feedback, model critiques and revises its own outputs
    according to a "constitution" (set of principles).

    Process:
        1. Generate initial response
        2. For each principle:
           a. Generate critique based on principle
           b. Revise response based on critique
        3. Final revised response

    Can be combined with RLHF:
        - Use Constitutional AI for self-improvement
        - Then fine-tune with RL on constitutional pairs

    Example:
        >>> config = ConstitutionalAIConfig(constitution=[
        ...     "Be helpful", "Be harmless", "Be honest"
        ... ])
        >>> cai = ConstitutionalAI(model, config)
        >>>
        >>> response = "How to make a weapon?"
        >>> revised = cai.critique_and_revise(
        ...     prompt="User asked harmful question",
        ...     response=response
        ... )
        >>> # revised will refuse to answer harmful question
    """

    def __init__(self, model: nn.Module, config: ConstitutionalAIConfig):
        self.model = model
        self.config = config

    def generate(self, prompt: str, max_length: int = 512) -> str:
        """Generate response from model"""
        # In practice, use actual generation
        return f"Generated response to: {prompt}"

    def critique_and_revise(
        self,
        prompt: str,
        response: str
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Critique and revise response according to constitution.

        Returns:
            final_response: Revised response
            history: List of critiques and revisions
        """
        history = []
        current_response = response

        for iteration in range(self.config.num_critique_iterations):
            for principle in self.config.constitution:
                # Generate critique
                critique_prompt = self.config.critique_prompt_template.format(
                    principle=principle,
                    response=current_response
                )
                critique = self.generate(critique_prompt)

                # Generate revision
                revision_prompt = self.config.revision_prompt_template.format(
                    response=current_response,
                    critique=critique
                )
                revised_response = self.generate(revision_prompt)

                # Store history
                history.append({
                    'principle': principle,
                    'critique': critique,
                    'revision': revised_response
                })

                current_response = revised_response

        return current_response, history


# ============================================================================
# Safe RLHF
# ============================================================================

@dataclass
class SafeRLHFConfig:
    """Configuration for Safe RLHF"""
    # Two reward models
    helpfulness_coef: float = 1.0
    harmlessness_coef: float = 1.0

    # Safety threshold
    min_harmlessness_reward: float = 0.0


class SafeRLHFTrainer:
    """
    Safe RLHF - Separate Helpfulness and Harmlessness

    Key insight: Helpfulness and harmlessness can conflict.
    Solution: Train two separate reward models.

    reward_total = α * reward_helpfulness + β * reward_harmlessness

    With constraint: reward_harmlessness > threshold

    This prevents "helpful but harmful" responses.

    Example:
        >>> config = SafeRLHFConfig(
        ...     helpfulness_coef=1.0,
        ...     harmlessness_coef=2.0  # Prioritize safety
        ... )
        >>> trainer = SafeRLHFTrainer(
        ...     policy_model=model,
        ...     helpfulness_rm=helpful_rm,
        ...     harmlessness_rm=harmless_rm,
        ...     config=config
        ... )
    """

    def __init__(
        self,
        policy_model: nn.Module,
        helpfulness_rm: nn.Module,
        harmlessness_rm: nn.Module,
        config: SafeRLHFConfig
    ):
        self.policy_model = policy_model
        self.helpfulness_rm = helpfulness_rm
        self.harmlessness_rm = harmlessness_rm
        self.config = config

    def compute_rewards(
        self,
        completions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute separate rewards.

        Returns:
            total_reward: Combined reward
            helpfulness_reward: Helpfulness component
            harmlessness_reward: Harmlessness component
        """
        with torch.no_grad():
            helpfulness = self.helpfulness_rm(completions)
            harmlessness = self.harmlessness_rm(completions)

        # Combine rewards
        total = (
            self.config.helpfulness_coef * helpfulness
            + self.config.harmlessness_coef * harmlessness
        )

        # Apply safety constraint
        safe_mask = harmlessness > self.config.min_harmlessness_reward
        total = total * safe_mask.float()  # Zero reward if unsafe

        return total, helpfulness, harmlessness


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("RLHF and DPO - Alignment Techniques")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("Comparison")
    print("=" * 80)
    print("""
RLHF vs DPO:

RLHF (InstructGPT):
------------------
1. Train reward model on preferences
2. Use reward model to score completions
3. Train policy with PPO to maximize reward
4. Add KL penalty to prevent divergence

Pros:
- Well-established, used in ChatGPT
- Flexible reward functions
- Can incorporate multiple objectives

Cons:
- Complex: 3 models (policy, ref, reward)
- Unstable: RL training is tricky
- Expensive: Multiple forward passes

DPO (Direct):
------------
1. Directly optimize policy on preferences
2. No reward model needed
3. No RL needed

Pros:
- Simple: just supervised learning!
- Stable: no RL instability
- Efficient: fewer forward passes
- Same final performance as RLHF

Cons:
- Less flexible than RLHF
- Harder to incorporate constraints

Constitutional AI:
-----------------
- Self-critique and revision
- No human feedback needed (after initial)
- Can be combined with RLHF/DPO

Safe RLHF:
----------
- Separate helpfulness and harmlessness
- Prevents helpful but harmful responses
- Important for deployment

Recommendation:
--------------
- Start: DPO (simplest, works well)
- Need flexibility: RLHF
- Need safety: Safe RLHF or Constitutional AI
- Production: Combine all techniques

Performance:
-----------
- DPO matches RLHF quality
- Constitutional AI adds safety
- Safe RLHF prevents harmful outputs

All techniques are critical for:
- Alignment with human values
- Safety and harmlessness
- Helpfulness and instruction-following
- Deployment readiness
""")

    print("\n" + "=" * 80)
    print("Practical Usage")
    print("=" * 80)
    print("""
Step-by-step RLHF:
1. Supervised fine-tuning (SFT) on instructions
2. Collect human preferences (A vs B comparisons)
3. Train reward model on preferences
4. RL training (PPO) with reward model
5. Final model: helpful, harmless, honest

Step-by-step DPO:
1. Supervised fine-tuning (SFT) on instructions
2. Collect human preferences (A vs B comparisons)
3. Direct optimization on preferences (no RM, no RL!)
4. Final model: same quality as RLHF, simpler

With Constitutional AI:
1. Define constitution (principles)
2. Generate responses
3. Self-critique against principles
4. Self-revise based on critiques
5. Create synthetic preference dataset
6. Train with DPO on synthetic data

This is the path to safe, helpful AI!
""")

    print("=" * 80)
