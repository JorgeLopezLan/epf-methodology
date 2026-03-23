"""
LSTM Price Encoder for EPF v10.0
=================================
Pre-trained LSTM that converts raw price sequences into fixed-dim embeddings.
These embeddings augment XGBoost's tabular features with temporal patterns
that flattened lag columns cannot capture.

Usage:
    # Pre-train (run once):
    python scripts/train_lstm_encoder.py --hidden 64 --layers 2 --epochs 100

    # At training time (env-gated):
    EPF_LSTM_EMBEDDINGS=true python run_training.py --direct ...

    # Embeddings are 64 extra float features: lstm_emb_0 ... lstm_emb_63
"""
import logging
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PriceEncoder(nn.Module):
    """LSTM encoder: raw price/exogenous sequence → fixed-dim embedding.

    Supports:
    - input_size=1: price-only (v10.0 default)
    - input_size=3: price + demand + temperature (v10.1 exogenous)
    - output_size=1: predict next hour (v10.0 generic)
    - output_size=24: predict next 24h (v10.1 task-aligned)
    """

    def __init__(self, hidden_dim=64, n_layers=2, dropout=0.1,
                 input_size=1, output_size=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.input_size = input_size
        self.output_size = output_size
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_dim,
            num_layers=n_layers, dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, output_size)

    def encode(self, x):
        """Extract embedding from sequence. x: (batch, seq_len, input_size)"""
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]  # last layer hidden: (batch, hidden_dim)

    def forward(self, x):
        """Pre-training forward: predict target from sequence."""
        embedding = self.encode(x)
        out = self.head(embedding)  # (batch, output_size)
        if self.output_size == 1:
            return out.squeeze(-1)  # (batch,)
        return out  # (batch, 24) for task-aligned


class LSTMEmbedder:
    """Feature extraction wrapper for the pre-trained LSTM encoder.

    Loads a saved PriceEncoder and extracts embeddings from price windows.
    Handles normalization, padding, GPU acceleration, and batching.
    """

    def __init__(self, model_path="data/models/lstm_encoder.pt",
                 window=168, device="auto"):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        self.window = checkpoint.get("window", window)
        self.hidden_dim = checkpoint["hidden_dim"]
        self.n_layers = checkpoint["n_layers"]
        self.price_mean = checkpoint["price_mean"]
        self.price_std = checkpoint["price_std"]

        # Resolve device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # v10.1: Support variable input_size and output_size
        self.input_size = checkpoint.get("input_size", 1)
        self.output_size = checkpoint.get("output_size", 1)

        # Build and load encoder
        self.encoder = PriceEncoder(
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            input_size=self.input_size,
            output_size=self.output_size,
        )
        self.encoder.load_state_dict(checkpoint["state_dict"])
        self.encoder.to(self.device)
        self.encoder.eval()

        logger.info("LSTMEmbedder loaded: hidden=%d, layers=%d, window=%d, input=%d, output=%d, device=%s",
                     self.hidden_dim, self.n_layers, self.window, self.input_size, self.output_size, self.device)

    def _normalize(self, prices):
        """Z-score normalize using training statistics."""
        return (prices - self.price_mean) / max(self.price_std, 1e-6)

    def _prepare_window(self, price_series, origin_idx):
        """Extract and pad a price window ending at origin_idx."""
        start = max(0, origin_idx - self.window)
        window = price_series.iloc[start:origin_idx].values.astype(np.float32)

        # Replace NaN with forward fill then 0
        mask = np.isnan(window)
        if mask.any():
            for i in range(1, len(window)):
                if mask[i] and not mask[i-1]:
                    window[i] = window[i-1]
            window = np.nan_to_num(window, nan=0.0)

        # Normalize
        window = self._normalize(window)

        # Pad to fixed length if shorter
        if len(window) < self.window:
            pad = np.zeros(self.window - len(window), dtype=np.float32)
            window = np.concatenate([pad, window])

        return window

    @torch.no_grad()
    def compute_embedding(self, price_series, origin_idx):
        """Extract embedding for a single origin. Returns dict of lstm_emb_* features."""
        window = self._prepare_window(price_series, origin_idx)
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # (1, seq, 1)
        x = x.to(self.device)
        emb = self.encoder.encode(x).cpu().numpy()[0]  # (hidden_dim,)
        return {f"lstm_emb_{i}": float(v) for i, v in enumerate(emb)}

    @torch.no_grad()
    def compute_embeddings_batch(self, price_series, origin_dts, ree_df=None):
        """Batch compute embeddings for all training samples.

        Args:
            price_series: pd.Series of day_ahead_price (full history)
            origin_dts: pd.Series of origin datetime values from the data DataFrame
            ree_df: Optional full DataFrame for index lookup

        Returns:
            dict mapping column names to numpy arrays (one per embedding dim)
        """
        import pandas as pd

        # Get unique origins to avoid redundant computation
        unique_dts = origin_dts.unique()
        dt_to_idx = {}

        # Build index mapping
        if ree_df is not None:
            price_index = ree_df.index
        else:
            price_index = price_series.index

        for dt in unique_dts:
            try:
                idx = price_index.get_loc(dt)
                if isinstance(idx, slice):
                    idx = idx.start
                dt_to_idx[dt] = idx
            except KeyError:
                # Find nearest
                diffs = np.abs((price_index - dt).total_seconds())
                dt_to_idx[dt] = int(np.argmin(diffs))

        # Prepare all windows
        windows = []
        valid_dts = []
        for dt in unique_dts:
            idx = dt_to_idx[dt]
            if idx >= 24:  # Need at least 24h of history
                w = self._prepare_window(price_series, idx)
                windows.append(w)
                valid_dts.append(dt)

        if not windows:
            logger.warning("No valid windows for LSTM embeddings")
            return {f"lstm_emb_{i}": np.full(len(origin_dts), np.nan)
                    for i in range(self.hidden_dim)}

        # Batch GPU forward pass
        batch = torch.tensor(np.stack(windows), dtype=torch.float32).unsqueeze(-1)  # (N, seq, 1)

        # Process in chunks to avoid GPU OOM
        chunk_size = 2048
        all_embs = []
        for i in range(0, len(batch), chunk_size):
            chunk = batch[i:i+chunk_size].to(self.device)
            emb = self.encoder.encode(chunk).cpu().numpy()
            all_embs.append(emb)
        embeddings = np.concatenate(all_embs, axis=0)  # (N_unique, hidden_dim)

        # Map back to per-sample embeddings
        dt_to_emb = {dt: emb for dt, emb in zip(valid_dts, embeddings)}
        result = np.zeros((len(origin_dts), self.hidden_dim), dtype=np.float32)
        for i, dt in enumerate(origin_dts):
            if dt in dt_to_emb:
                result[i] = dt_to_emb[dt]
            # else: zeros (for origins without enough history)

        logger.info("LSTM embeddings: %d unique origins, %d total samples, %d-dim",
                     len(valid_dts), len(origin_dts), self.hidden_dim)

        return {f"lstm_emb_{i}": result[:, i] for i in range(self.hidden_dim)}
