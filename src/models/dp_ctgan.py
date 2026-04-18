"""
DP-CTGAN — GAN tabular con Differential Privacy via DP-SGD (Opacus).

La privacidad diferencial se aplica exclusivamente al Discriminador:
es el único componente que procesa datos reales, y por tanto el único
que puede filtrar información privada a través de sus gradientes.

El Generador no ve datos reales directamente, por lo que no requiere DP
y puede usar BatchNorm1d sin restricciones.

Restricciones del Discriminador para compatibilidad con Opacus:
  - Sin BatchNorm (calcula estadísticos cross-batch, incompatible con per-sample gradients)
  - LayerNorm como alternativa: normaliza por features dentro de cada muestra
  - Sin módulos con estado compartido entre muestras del batch
"""

import torch
import torch.nn as nn


class _ResBlock(nn.Module):
    """Bloque residual con LayerNorm (Opacus-compatible)."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(0.2),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class TabGANGenerator(nn.Module):
    """
    MLP generator: z ∈ R^{noise_dim} → x̂ ∈ R^{input_dim}.

    Puede usar BatchNorm1d porque no procesa datos reales
    y no está sujeto al mecanismo de privacidad diferencial.
    """

    def __init__(
        self,
        noise_dim: int,
        output_dim: int,
        hidden_dims: tuple = (256, 256),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = noise_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class TabGANDiscriminator(nn.Module):
    """
    MLP discriminator: x ∈ R^{input_dim} → logit ∈ R.

    Diseñado para compatibilidad con Opacus DP-SGD:
      - LayerNorm en lugar de BatchNorm
      - Bloques residuales para compensar la menor capacidad de LayerNorm
      - Sin operaciones cross-batch
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple = (256, 256),
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.LeakyReLU(0.2),
        )
        self.blocks = nn.ModuleList([
            _ResBlock(hidden_dims[i])
            for i in range(len(hidden_dims))
        ])
        self.output_proj = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)
