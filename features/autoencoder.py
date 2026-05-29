from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset


class VariationalAutoencoder(nn.Module):
    """
    Beta-VAE for non-linear latent factor extraction.

    Captures non-Gaussian market dynamics that PCA misses.
    The latent space provides the initial state for the PINN-PDE system.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        dropout: float = 0.1,
        beta: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder
        enc_layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU(), nn.Dropout(dropout)]
            prev = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        # Decoder
        dec_layers: List[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU(), nn.Dropout(dropout)]
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def loss(
        self, x: torch.Tensor, x_recon: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = recon_loss + self.beta * kl_loss
        return total, recon_loss, kl_loss

    @torch.no_grad()
    def encode_numpy(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        t = torch.FloatTensor(x)
        mu, _ = self.encode(t)
        return mu.numpy()


class AutoencoderTrainer:
    """Handles training, validation, and feature extraction for the VAE."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        beta: float = 1.0,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        device: Optional[str] = None,
    ):
        self.device = torch.device(
            device or (
                "cuda" if torch.cuda.is_available() else
                "mps"  if torch.backends.mps.is_available() else
                "cpu"
            )
        )
        self.model = VariationalAutoencoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout,
            beta=beta,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=50, factor=0.5, min_lr=1e-5
        )
        self._input_mean: Optional[np.ndarray] = None
        self._input_std: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        epochs: int = 500,
        batch_size: int = 128,
        val_split: float = 0.1,
        verbose: bool = True,
    ) -> List[float]:
        """Train the VAE on a 2D feature matrix. Returns loss history."""
        # Normalize
        self._input_mean = X.mean(axis=0)
        self._input_std = X.std(axis=0) + 1e-8
        X_norm = (X - self._input_mean) / self._input_std

        n_val = max(1, int(len(X_norm) * val_split))
        X_train, X_val = X_norm[:-n_val], X_norm[-n_val:]

        train_ds = TensorDataset(torch.FloatTensor(X_train))
        val_ds   = TensorDataset(torch.FloatTensor(X_val))
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=batch_size)

        loss_history = []
        best_val_loss = float("inf")
        patience_counter = 0
        patience = 100

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            for (batch,) in train_dl:
                batch = batch.to(self.device)
                x_recon, mu, logvar = self.model(batch)
                loss, _, _ = self.model.loss(batch, x_recon, mu, logvar)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item()

            # Validation — track recon and KL separately to detect posterior collapse
            self.model.eval()
            val_loss = val_recon = val_kl = 0.0
            with torch.no_grad():
                for (batch,) in val_dl:
                    batch = batch.to(self.device)
                    x_recon, mu, logvar = self.model(batch)
                    loss, recon, kl = self.model.loss(batch, x_recon, mu, logvar)
                    val_loss  += loss.item()
                    val_recon += recon.item()
                    val_kl    += kl.item()

            n_val_batches = len(val_dl)
            avg_val   = val_loss  / n_val_batches
            avg_recon = val_recon / n_val_batches
            avg_kl    = val_kl   / n_val_batches
            self.scheduler.step(avg_val)
            loss_history.append(avg_val)

            if avg_val < best_val_loss:
                logger.info(f"VAE new best val={avg_val:.4f} at epoch {epoch}")
                best_val_loss = avg_val
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and epoch % 100 == 0:
                logger.info(
                    f"VAE Epoch {epoch}/{epochs} | val={avg_val:.4f} | "
                    f"recon={avg_recon:.4f} | kl={avg_kl:.4f}"
                )
                if avg_kl < 0.01:
                    logger.warning(
                        f"VAE posterior collapse detected at epoch {epoch}: "
                        f"KL={avg_kl:.5f} < 0.01. Consider increasing beta or learning rate."
                    )

            if patience_counter >= patience:
                logger.info(f"VAE early stopping at epoch {epoch}")
                break

        return loss_history

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return latent means for a feature matrix."""
        X_norm = (X - self._input_mean) / self._input_std
        self.model.eval()
        results = []
        with torch.no_grad():
            for i in range(0, len(X_norm), 256):
                batch = torch.FloatTensor(X_norm[i : i + 256]).to(self.device)
                mu, _ = self.model.encode(batch)
                results.append(mu.cpu().numpy())
        return np.concatenate(results, axis=0)

    def transform_with_uncertainty(self, X: np.ndarray, n_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (mean_latent, std_latent) using MC dropout sampling.
        std_latent is the Bayesian uncertainty in the latent encoding.
        """
        X_norm = (X - self._input_mean) / self._input_std
        self.model.train()  # enable dropout for MC sampling
        all_mus = []
        with torch.no_grad():
            for _ in range(n_samples):
                mus = []
                for i in range(0, len(X_norm), 256):
                    batch = torch.FloatTensor(X_norm[i : i + 256]).to(self.device)
                    mu, _ = self.model.encode(batch)
                    mus.append(mu.cpu().numpy())
                all_mus.append(np.concatenate(mus, axis=0))
        self.model.eval()
        stacked = np.stack(all_mus, axis=0)  # (n_samples, N, latent_dim)
        return stacked.mean(axis=0), stacked.std(axis=0)

    def fit_transform(
        self,
        X: np.ndarray,
        index: Optional[pd.Index] = None,
        **fit_kwargs,
    ) -> pd.DataFrame:
        self.fit(X, **fit_kwargs)
        latent = self.transform(X)
        cols = [f"z_{i+1}" for i in range(latent.shape[1])]
        return pd.DataFrame(latent, index=index, columns=cols)

    def save(self, path: str):
        torch.save({
            "model_state":  self.model.state_dict(),
            "input_mean":   self._input_mean,
            "input_std":    self._input_std,
            "input_dim":    self.model.input_dim,
            "latent_dim":   self.model.latent_dim,
            "beta":         self.model.beta,
        }, path)
        logger.info(f"VAE saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self._input_mean = ckpt["input_mean"]
        self._input_std  = ckpt["input_std"]
        self.model.eval()
        logger.info(f"VAE loaded from {path}")
