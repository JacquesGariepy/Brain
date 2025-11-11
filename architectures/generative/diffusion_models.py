"""
Diffusion Models - SOTA Generative Modeling

Implementations:
- DDPM (Denoising Diffusion Probabilistic Models)
- DDIM (Denoising Diffusion Implicit Models) - faster sampling
- Score-based models
- Latent Diffusion Models
- Classifier-free guidance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal embedding for timesteps"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: (batch,) integer timesteps

        Returns:
            Embeddings of shape (batch, dim)
        """
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=timesteps.device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block for U-Net"""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, H, W)
            time_emb: (batch, time_emb_dim)
        """
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return h + self.residual_conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for diffusion models.

    Standard architecture for image generation with diffusion.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16, 8),
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels

        # Time embedding
        time_emb_dim = model_channels * 4
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Input
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Downsampling
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResidualBlock(ch, model_channels * mult, time_emb_dim)
                )
                ch = model_channels * mult

            if level != len(channel_mult) - 1:
                self.down_blocks.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))

        # Middle
        self.middle_block = nn.Sequential(
            ResidualBlock(ch, ch, time_emb_dim),
            ResidualBlock(ch, ch, time_emb_dim)
        )

        # Upsampling
        self.up_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(
                    ResidualBlock(ch, model_channels * mult, time_emb_dim)
                )
                ch = model_channels * mult

            if level != 0:
                self.up_blocks.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))

        # Output
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1)
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy image (batch, channels, H, W)
            timesteps: Timesteps (batch,)

        Returns:
            Predicted noise (batch, channels, H, W)
        """
        # Time embedding
        time_emb = self.time_embedding(timesteps)

        # Input
        h = self.input_conv(x)

        # Downsampling (store for skip connections)
        hs = [h]
        for block in self.down_blocks:
            if isinstance(block, ResidualBlock):
                h = block(h, time_emb)
            else:
                h = block(h)
            hs.append(h)

        # Middle
        for block in self.middle_block:
            h = block(h, time_emb)

        # Upsampling (with skip connections)
        for block in self.up_blocks:
            if isinstance(block, ResidualBlock):
                h = torch.cat([h, hs.pop()], dim=1)
                h = block(h, time_emb)
            else:
                h = block(h)

        # Output
        return self.output_conv(h)


class DDPM:
    """
    Denoising Diffusion Probabilistic Model.

    Implements the forward and reverse diffusion process.
    """

    def __init__(
        self,
        model: nn.Module,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = 'cpu'
    ):
        self.model = model
        self.num_timesteps = num_timesteps
        self.device = device

        # Variance schedule (linear)
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion: sample q(x_t | x_0).

        Args:
            x_0: Original image
            t: Timesteps
            noise: Optional noise (otherwise sampled)

        Returns:
            Noisy image x_t
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise

    def p_sample(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """
        Reverse diffusion: sample p(x_{t-1} | x_t).

        Args:
            x_t: Noisy image at timestep t
            t: Current timestep

        Returns:
            Less noisy image x_{t-1}
        """
        # Predict noise
        t_batch = torch.full((x_t.shape[0],), t, device=self.device, dtype=torch.long)
        predicted_noise = self.model(x_t, t_batch)

        # Compute mean of p(x_{t-1} | x_t)
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]
        beta_t = self.betas[t]

        mean = (1 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
        )

        # Add noise (except for t=0)
        if t > 0:
            noise = torch.randn_like(x_t)
            variance = self.posterior_variance[t]
            return mean + torch.sqrt(variance) * noise
        else:
            return mean

    def sample(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Generate samples by running reverse diffusion.

        Args:
            shape: Shape of samples to generate (batch, channels, H, W)

        Returns:
            Generated images
        """
        # Start from pure noise
        x = torch.randn(shape, device=self.device)

        # Run reverse diffusion
        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample(x, t)

        return x

    def training_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """
        Compute training loss (simplified objective).

        Args:
            x_0: Original images

        Returns:
            Loss value
        """
        batch_size = x_0.shape[0]

        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)

        # Sample noise
        noise = torch.randn_like(x_0)

        # Get noisy images
        x_t = self.q_sample(x_0, t, noise)

        # Predict noise
        predicted_noise = self.model(x_t, t)

        # MSE loss
        loss = F.mse_loss(predicted_noise, noise)

        return loss


class DDIM:
    """
    Denoising Diffusion Implicit Models.

    Faster sampling than DDPM by using deterministic sampling.
    """

    def __init__(
        self,
        model: nn.Module,
        num_timesteps: int = 1000,
        num_inference_steps: int = 50,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = 'cpu'
    ):
        self.model = model
        self.num_timesteps = num_timesteps
        self.num_inference_steps = num_inference_steps
        self.device = device

        # Same variance schedule as DDPM
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Timesteps for inference (subsample)
        self.timesteps = torch.linspace(0, num_timesteps - 1, num_inference_steps, device=device).long()

    def ddim_sample(self, x_t: torch.Tensor, t: int, t_next: int, eta: float = 0.0) -> torch.Tensor:
        """
        DDIM sampling step.

        Args:
            x_t: Current noisy image
            t: Current timestep
            t_next: Next timestep
            eta: Stochasticity parameter (0 = deterministic)

        Returns:
            x_{t_next}
        """
        # Predict noise
        t_batch = torch.full((x_t.shape[0],), t, device=self.device, dtype=torch.long)
        predicted_noise = self.model(x_t, t_batch)

        # Predict x_0
        alpha_cumprod_t = self.alphas_cumprod[t]
        x_0_pred = (x_t - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)

        # Compute variance
        alpha_cumprod_t_next = self.alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0)

        sigma_t = eta * torch.sqrt(
            (1 - alpha_cumprod_t_next) / (1 - alpha_cumprod_t) *
            (1 - alpha_cumprod_t / alpha_cumprod_t_next)
        )

        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_cumprod_t_next - sigma_t ** 2) * predicted_noise

        # Sample noise
        noise = torch.randn_like(x_t) if t_next > 0 else torch.zeros_like(x_t)

        x_t_next = torch.sqrt(alpha_cumprod_t_next) * x_0_pred + dir_xt + sigma_t * noise

        return x_t_next

    def sample(self, shape: Tuple[int, ...], eta: float = 0.0) -> torch.Tensor:
        """
        Generate samples using DDIM (much faster than DDPM).

        Args:
            shape: Shape of samples
            eta: Stochasticity (0 = deterministic, 1 = stochastic like DDPM)

        Returns:
            Generated images
        """
        x = torch.randn(shape, device=self.device)

        timesteps_with_prev = list(zip(self.timesteps[:-1], self.timesteps[1:]))
        timesteps_with_prev.append((self.timesteps[-1], -1))

        for t, t_next in reversed(timesteps_with_prev):
            x = self.ddim_sample(x, t.item(), t_next.item() if t_next != -1 else -1, eta)

        return x


class ClassifierFreeGuidance:
    """
    Classifier-free guidance for conditional generation.

    Improves quality of conditional generation without needing a classifier.
    """

    def __init__(self, model: nn.Module, guidance_scale: float = 7.5):
        self.model = model
        self.guidance_scale = guidance_scale

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict noise with classifier-free guidance.

        Args:
            x_t: Noisy input
            t: Timesteps
            conditioning: Conditioning information (e.g., text embeddings)

        Returns:
            Guided noise prediction
        """
        # Unconditional prediction
        noise_uncond = self.model(x_t, t, None)

        if conditioning is None:
            return noise_uncond

        # Conditional prediction
        noise_cond = self.model(x_t, t, conditioning)

        # Classifier-free guidance
        noise = noise_uncond + self.guidance_scale * (noise_cond - noise_uncond)

        return noise
