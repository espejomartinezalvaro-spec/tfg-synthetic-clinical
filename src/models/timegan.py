"""
TimeGAN — Generación de series temporales clínicas mediante GAN temporal.

Referencia: Yoon et al., "Time-series Generative Adversarial Networks" (NeurIPS 2019).
            https://proceedings.neurips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html

Arquitectura (5 redes GRU):
    Embedder  (E): X → H              mapea series reales al espacio latente
    Recovery  (R): H → X̂             reconstruye la serie desde el espacio latente
    Generator (G): Z → Ĥ             genera representaciones latentes desde ruido
    Supervisor (S): H_t → H_{t+1}    modela la dinámica temporal paso a paso
    Discriminator (D): H → [0,1]     distingue representaciones reales vs sintéticas

Entrenamiento en 3 fases:
    1. Pre-entrenamiento del autoencoder (E + R)
    2. Pre-entrenamiento del supervisor (S) con el embedder congelado
    3. Entrenamiento conjunto adversarial (E + R + G + S + D)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GRUBlock(nn.Module):
    """Backbone GRU compartido por todos los componentes de TimeGAN."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        output_activation: nn.Module | None = None,
    ):
        super().__init__()
        self.rnn = nn.GRU(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_dim, output_dim)
        self.act  = output_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(x)
        out  = self.proj(h)
        if self.act is not None:
            out = self.act(out)
        return out


class Embedder(_GRUBlock):
    """X → H: codifica la serie real en el espacio latente continuo."""

    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int):
        super().__init__(feature_dim, hidden_dim, hidden_dim, num_layers, nn.Sigmoid())


class Recovery(_GRUBlock):
    """H → X̂: reconstruye la serie desde el espacio latente."""

    def __init__(self, hidden_dim: int, feature_dim: int, num_layers: int):
        super().__init__(hidden_dim, hidden_dim, feature_dim, num_layers, nn.Sigmoid())


class Generator(_GRUBlock):
    """Z → Ĥ: genera representaciones latentes a partir de ruido uniforme."""

    def __init__(self, noise_dim: int, hidden_dim: int, num_layers: int):
        super().__init__(noise_dim, hidden_dim, hidden_dim, num_layers, nn.Sigmoid())


class Supervisor(_GRUBlock):
    """H_t → H_{t+1}: captura la dinámica temporal paso a paso en el espacio latente."""

    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__(hidden_dim, hidden_dim, hidden_dim, num_layers, nn.Sigmoid())


class Discriminator(nn.Module):
    """
    H → escalar: clasifica si una secuencia latente es real o sintética.

    Usa el último hidden state de la GRU para obtener un escalar por secuencia
    (en lugar de por timestep), coherente con el esquema de clasificación del paper.
    """

    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        self.rnn  = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        _, h_n = self.rnn(h)          # h_n: (num_layers, batch, hidden_dim)
        return self.proj(h_n[-1])     # (batch, 1)


# ---------------------------------------------------------------------------
# Funciones de pérdida
# ---------------------------------------------------------------------------

def reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """MSE entre la serie real y la reconstrucción del autoencoder."""
    return F.mse_loss(x_hat, x)


def supervised_step_loss(h: torch.Tensor, h_sup: torch.Tensor) -> torch.Tensor:
    """
    MSE entre S(H_t) y H_{t+1}: penaliza errores en la predicción del siguiente paso.
    Opera sobre todos los timesteps excepto el último.
    """
    return F.mse_loss(h_sup[:, :-1, :], h[:, 1:, :])


def generator_adversarial_loss(d_fake: torch.Tensor) -> torch.Tensor:
    """El generador intenta que el discriminador clasifique sus muestras como reales."""
    return F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))


def discriminator_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """Loss estándar del discriminador GAN (real → 1, fake → 0)."""
    return (
        F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
        + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
    )


def moments_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """
    Penaliza la discrepancia en media y varianza entre series reales y sintéticas.
    Actúa como regularizador estadístico para preservar la distribución marginal.
    """
    mean_loss = F.mse_loss(x_hat.mean(dim=0), x.mean(dim=0))
    var_loss  = F.mse_loss(x_hat.var(dim=0),  x.var(dim=0))
    return mean_loss + var_loss
