# src/utils/data_logger.py
"""
DataLogger — stores robot states and actions in memory, then flushes to HDF5.

Usage
-----
    logger = DataLogger()
    logger.start()                             # begin recording
    logger.record(data.qpos, data.qvel, data.ctrl)   # each sim step
    logger.stop()                              # stop & auto-save to data/
    # or manually:
    logger.save(Path("data/my_demo.h5"))

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
from typing import List

import h5py
import numpy as np


class DataLogger:
    """Accumulates (qpos, qvel, ctrl) tuples and saves them as HDF5."""

    def __init__(self, save_dir: str | Path = "data") -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._qpos_buf: List[np.ndarray] = []
        self._qvel_buf: List[np.ndarray] = []
        self._ctrl_buf: List[np.ndarray] = []
        self._recording = False

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
        """Flush buffers to an HDF5 file at *path*."""
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

    # ── Internals ────────────────────────────────────────────────────────────

    def _auto_filename(self) -> Path:
        """Generate a unique filename based on current timestamp."""
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.save_dir / f"demo_{stamp}.h5"
