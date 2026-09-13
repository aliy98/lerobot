from __future__ import annotations

import torch
from lerobot.utils.sample_weighting import SampleWeighter


class HILInterventionWeighter(SampleWeighter):
    """Weight only HIL frames; leave base frames at 1.0.

    Requires in each batch:
      - dataset_origin: "base" | "hil"  (str or bytes, or 0/1 encoding)
      - task_is_policy: 1 = autonomous, 0 = human correction
    """

    def __init__(
        self,
        device: torch.device,
        autonomous_weight: float = 0.3,   # HIL policy segments
        correction_weight: float = 2.0,   # HIL human fixes
        base_weight: float = 1.0,         # base demos (default)
        hil_origin_value: str = "hil",
        base_origin_value: str = "base",
    ):
        self.device = device
        self.autonomous_weight = float(autonomous_weight)
        self.correction_weight = float(correction_weight)
        self.base_weight = float(base_weight)
        self.hil_origin_value = hil_origin_value
        self.base_origin_value = base_origin_value

    def _is_hil_mask(self, batch: dict, batch_size: int) -> torch.Tensor:
        """True where sample comes from HIL (not base)."""
        if "dataset_origin" not in batch:
            raise KeyError(
                "Batch missing 'dataset_origin'. "
                "Merge script must set origin='base' or 'hil' per frame."
            )

        origin = batch["dataset_origin"]

        # Tensor of 0/1 (if you encoded base=0, hil=1)
        if isinstance(origin, torch.Tensor):
            o = origin
            while o.ndim > 1:
                o = o.squeeze(-1)
            return (o.float() > 0.5).to(self.device)

        # List / array of strings
        if isinstance(origin, (list, tuple)):
            vals = [
                (x.decode() if isinstance(x, (bytes, bytearray)) else str(x)).lower()
                for x in origin
            ]
            return torch.tensor(
                [v == self.hil_origin_value.lower() for v in vals],
                device=self.device,
                dtype=torch.bool,
            )

        # Single string broadcast (unlikely)
        if isinstance(origin, (str, bytes)):
            s = origin.decode() if isinstance(origin, (bytes, bytearray)) else origin
            is_hil = s.lower() == self.hil_origin_value.lower()
            return torch.full((batch_size,), is_hil, device=self.device, dtype=torch.bool)

        raise TypeError(f"Unsupported dataset_origin type: {type(origin)}")

    def compute_batch_weights(self, batch: dict) -> tuple[torch.Tensor, dict]:
        print("batch keys:", sorted(batch.keys()))
        if "task_is_policy" not in batch or "dataset_origin" not in batch:
            # Find batch size
            for key in ("action", "index", "observation.state"):
                if key in batch and hasattr(batch[key], "shape"):
                    bs = batch[key].shape[0]
                    break
            else:
                bs = 1
            w = torch.ones(bs, device=self.device)
            return w, {"type": "hil_intervention", "fallback": "missing_keys", "mean_weight": 1.0}

        is_policy = batch["task_is_policy"]
        while isinstance(is_policy, torch.Tensor) and is_policy.ndim > 1:
            is_policy = is_policy.squeeze(-1)
        is_policy = is_policy.float().to(self.device)

        batch_size = is_policy.shape[0]
        is_hil = self._is_hil_mask(batch, batch_size)

        # Start: everyone default 1.0 (base)
        w = torch.full((batch_size,), self.base_weight, device=self.device)

        # HIL autonomous
        hil_auto = is_hil & (is_policy > 0.5)
        # HIL correction
        hil_corr = is_hil & (is_policy <= 0.5)

        w = torch.where(
            hil_auto,
            torch.full_like(w, self.autonomous_weight),
            w,
        )
        w = torch.where(
            hil_corr,
            torch.full_like(w, self.correction_weight),
            w,
        )

        # Optional: keep mean weight ~1 so LR scale stays stable
        # (comment out if you want raw 1.0 / 0.3 / 2.0)
        # w = w * (w.numel() / (w.sum() + 1e-6))

        stats = {
            "type": "hil_intervention",
            "mean_weight": float(w.mean().item()),
            "frac_base": float((~is_hil).float().mean().item()),
            "frac_hil": float(is_hil.float().mean().item()),
            "frac_hil_correction": float(hil_corr.float().mean().item()),
            "frac_hil_autonomous": float(hil_auto.float().mean().item()),
            "base_weight": self.base_weight,
            "autonomous_weight": self.autonomous_weight,
            "correction_weight": self.correction_weight,
        }
        return w, stats

    def get_stats(self) -> dict:
        return {
            "type": "hil_intervention",
            "base_weight": self.base_weight,
            "autonomous_weight": self.autonomous_weight,
            "correction_weight": self.correction_weight,
        }