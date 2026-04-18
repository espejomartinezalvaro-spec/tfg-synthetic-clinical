"""
TabDDPM — arquitectura y proceso de difusión gaussiana para datos tabulares.

Referencia: Kotelnikov et al., "TabDDPM: Modelling Tabular Data with Diffusion Models" (2022).
            https://arxiv.org/abs/2209.15421

Implementación:
- Difusión gaussiana sobre el espacio de features preprocesado (continuo)
- Class conditioning por mortalidad via label embedding sumado al time embedding
- Cosine noise schedule (Nichol & Dhariwal, Improved DDPM, 2021)
- MLP denoiser con ResidualBlocks y SiLU activations
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Positional encoding para timesteps de difusión (mismo mecanismo que Transformers)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half  = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        emb = t[:, None].float() * freqs[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResidualBlock(nn.Module):
    """Bloque MLP con residual connection, LayerNorm y time conditioning."""

    def __init__(self, dim: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1     = nn.LayerNorm(dim)
        self.linear1   = nn.Linear(dim, dim)
        self.time_proj = nn.Linear(time_dim, dim)
        self.norm2     = nn.LayerNorm(dim)
        self.linear2   = nn.Linear(dim, dim)
        self.dropout   = nn.Dropout(dropout)
        self.act       = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.linear1(self.norm1(x))) + self.time_proj(t_emb)
        h = self.dropout(self.act(self.linear2(self.norm2(h))))
        return x + h


class TabDDPMDenoiser(nn.Module):
    """
    Red denoising: dado x_t y el timestep t, predice el ruido epsilon añadido.

    Condicionado en:
    - t  → SinusoidalTimeEmbedding → MLP
    - y  → Embedding (mortalidad 0/1), sumado al time embedding

    El índice num_classes se reserva como "null class" para dropout de clase
    durante el entrenamiento (classifier-free guidance).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple = (512, 512, 512, 512),
        time_emb_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.time_emb  = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp  = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        self.class_emb  = nn.Embedding(num_classes + 1, time_emb_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        self.blocks     = nn.ModuleList([
            ResidualBlock(hidden_dims[i], time_emb_dim, dropout)
            for i in range(len(hidden_dims))
        ])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1]),
            nn.Linear(hidden_dims[-1], input_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t_emb = self.time_mlp(self.time_emb(t))
        if y is not None:
            t_emb = t_emb + self.class_emb(y)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.output_proj(h)


class CosineScheduler:
    """
    Cosine noise schedule (Nichol & Dhariwal, 2021).

    Ventaja sobre el schedule lineal: la transición de señal a ruido es más
    gradual al final de la cadena, lo que mejora la calidad en distribuciones
    tabulares con colas largas (creatinina, lactato, LOS).
    """

    def __init__(self, T: int = 1000, s: float = 0.008):
        self.T   = T
        steps    = torch.arange(T + 1, dtype=torch.float64)
        f        = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
        abar     = (f / f[0]).float()
        betas    = torch.clamp(1 - abar[1:] / abar[:-1], 1e-5, 0.999)
        alphas   = 1.0 - betas
        abar     = torch.cumprod(alphas, dim=0)
        abar_prev = F.pad(abar[:-1], (1, 0), value=1.0)

        self.betas                         = betas
        self.alphas_cumprod                = abar
        self.sqrt_alphas_cumprod           = abar.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - abar).sqrt()
        self.posterior_variance            = betas * (1 - abar_prev) / (1 - abar)

    def to(self, device: torch.device) -> "CosineScheduler":
        for attr in [
            "betas", "alphas_cumprod", "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod", "posterior_variance",
        ]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward process: x_t = sqrt(abar_t)*x0 + sqrt(1-abar_t)*epsilon."""
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_alphas_cumprod[t][:, None]
        b = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        return a * x0 + b * noise, noise

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t_scalar: int,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Un paso del reverse process (Algorithm 2, Ho et al. 2020).

        x_{t-1} = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-abar_t) * eps_pred)
                  + sqrt(beta_t) * z
        """
        t_tensor         = torch.full((x_t.shape[0],), t_scalar, device=x_t.device, dtype=torch.long)
        eps_pred         = model(x_t, t_tensor, y)
        beta_t           = self.betas[t_scalar]
        sqrt_recip_alpha = (1.0 - beta_t).rsqrt()
        sqrt_1m_abar     = self.sqrt_one_minus_alphas_cumprod[t_scalar]
        mu               = sqrt_recip_alpha * (x_t - beta_t / sqrt_1m_abar * eps_pred)
        if t_scalar == 0:
            return mu
        return mu + beta_t.sqrt() * torch.randn_like(x_t)

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        n: int,
        input_dim: int,
        device: torch.device,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Genera n muestras ejecutando el reverse process completo (T → 0)."""
        model.eval()
        x = torch.randn(n, input_dim, device=device)
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t, y)
        return x

    def training_loss(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """MSE sobre predicción de ruido epsilon (simple loss, Ho et al. 2020)."""
        t          = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        x_t, noise = self.q_sample(x0, t)
        eps_pred   = model(x_t, t, y)
        return F.mse_loss(eps_pred, noise)
