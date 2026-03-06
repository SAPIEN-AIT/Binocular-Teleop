# src/utils/data_logger.py
"""
DataLogger — stores robot states and actions in memory, then flushes to HDF5.

Optionally logs each demo to Weights & Biases as a versioned Artifact plus
summary scalars. Pass a live ``wandb.run`` (or initialise via
``DataLogger.init_wandb()``) before starting to record.

Usage
-----
    logger = DataLogger()
    logger.init_wandb(project="binocular-teleop")  # optional
    logger.start()
    logger.record(data.qpos, data.qvel, data.ctrl)
    logger.stop()   # saves HDF5 and, if W&B is active, logs the artifact

HDF5 layout
------------
    observations/qpos   (N, n_qpos)
    observations/qvel   (N, n_qvel)
    actions/ctrl        (N, n_ctrl)
    metadata/
        created_at      ISO-8601 timestamp
        n_steps         total frames recorded
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np


class DataLogger:
    """Accumulates (qpos, qvel, ctrl) tuples and saves them as HDF5."""

    def __init__(self, save_dir: str | Path = "data",
                 wandb_run=None) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._qpos_buf: List[np.ndarray] = []
        self._qvel_buf: List[np.ndarray] = []
        self._ctrl_buf: List[np.ndarray] = []
        self._recording = False
        self._wandb_run = wandb_run   # optional wandb.run object

    # ── W&B helpers ─────────────────────────────────────────────────────────

    def init_wandb(self, project: str = "binocular-teleop",
                   entity: Optional[str] = None,
                   tags: Optional[list] = None,
                   config: Optional[dict] = None) -> None:
        """
        Initialise (or reuse) a W&B run attached to this logger.
        Safe to call even if wandb is not installed — logs a warning instead.
        """
        try:
            import wandb
        except ImportError:
            print("[W&B] wandb not installed — skipping (pip install wandb).")
            return

        if wandb.run is not None:
            # Reuse an already-active run (e.g. started by the caller)
            self._wandb_run = wandb.run
        else:
            self._wandb_run = wandb.init(
                project=project,
                entity=entity,
                tags=tags or [],
                config=config or {},
                job_type="data-collection",
                reinit=True,
            )
        print(f"[W&B] Logging to project '{self._wandb_run.project}' "
              f"— run '{self._wandb_run.name}'")

    # ── Public API ───────────────────────────────────────────────────────────

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def n_steps(self) -> int:
        return len(self._qpos_buf)

    def start(self) -> None:
        """Begin a new recording session (clears previous buffers)."""
        self._qpos_buf.clear()
        self._qvel_buf.clear()
        self._ctrl_buf.clear()
        self._recording = True
        print("[DATA] Recording started.")

    def record(self, qpos: np.ndarray, qvel: np.ndarray,
               ctrl: np.ndarray) -> None:
        """Append one timestep of data (only if recording is active)."""
        if not self._recording:
            return
        self._qpos_buf.append(qpos.copy())
        self._qvel_buf.append(qvel.copy())
        self._ctrl_buf.append(ctrl.copy())

    def stop(self) -> Path | None:
        """Stop recording and auto-save with a timestamped filename."""
        if not self._recording:
            return None
        self._recording = False
        if self.n_steps == 0:
            print("[DATA] Nothing recorded — skipping save.")
            return None
        filename = self._auto_filename()
        self.save(filename)
        return filename

    def save(self, path: str | Path) -> None:
        """Flush buffers to an HDF5 file at *path*, then log to W&B if active."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        qpos = np.stack(self._qpos_buf)
        qvel = np.stack(self._qvel_buf)
        ctrl = np.stack(self._ctrl_buf)

        with h5py.File(str(path), "w") as f:
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=qpos, compression="gzip")
            obs.create_dataset("qvel", data=qvel, compression="gzip")

            act = f.create_group("actions")
            act.create_dataset("ctrl", data=ctrl, compression="gzip")

            meta = f.create_group("metadata")
            meta.attrs["created_at"] = datetime.datetime.now().isoformat()
            meta.attrs["n_steps"] = len(self._qpos_buf)

        print(f"[DATA] Saved {len(self._qpos_buf)} steps → {path}")

        # ── W&B logging ────────────────────────────────────────────────────
        if self._wandb_run is not None:
            self._log_to_wandb(path, qpos, qvel, ctrl)

    def _log_to_wandb(self, path: Path, qpos: np.ndarray,
                      qvel: np.ndarray, ctrl: np.ndarray) -> None:
        """Upload the HDF5 as an Artifact and log summary scalars."""
        try:
            import wandb
        except ImportError:
            return

        run = self._wandb_run

        # --- Artifact (versioned dataset) ---
        artifact = wandb.Artifact(
            name=path.stem,             # e.g. "demo_20260305_180146"
            type="demo",
            description=f"{len(self._qpos_buf)}-step LEAP Hand demo",
            metadata={
                "n_steps": len(self._qpos_buf),
                "qpos_shape": list(qpos.shape),
                "ctrl_shape": list(ctrl.shape),
                "created_at": datetime.datetime.now().isoformat(),
            },
        )
        artifact.add_file(str(path))
        run.log_artifact(artifact)

        # --- Scalar summary ---
        run.log({
            "demo/n_steps":          len(self._qpos_buf),
            "demo/ctrl_mean":        float(np.mean(np.abs(ctrl))),
            "demo/ctrl_std":         float(np.std(ctrl)),
            "demo/qpos_range_mean":  float(np.mean(qpos.max(0) - qpos.min(0))),
            "demo/qvel_rms":         float(np.sqrt(np.mean(qvel ** 2))),
        })

        print(f"[W&B] Artifact '{path.stem}' uploaded — "
              f"{run.url}")

    # ── Internals ────────────────────────────────────────────────────────────

    def _auto_filename(self) -> Path:
        """Generate a unique filename based on current timestamp."""
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.save_dir / f"demo_{stamp}.h5"
