from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def save_airfoil_plot(
    coords: np.ndarray,
    out_path: str | Path = "outputs/generated_airfoil.png",
    color: str = "#1f77b4",
) -> Path:
    """Save airfoil plot in a single color and return output path."""
    c = np.asarray(coords, dtype=np.float32)
    n = (c.shape[0] + 1) // 2
    upper = c[:n]
    lower = np.vstack([c[n - 1], c[n:]])

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 3))
    plt.plot(upper[:, 0], upper[:, 1], lw=2, color=color)
    plt.plot(lower[:, 0], lower[:, 1], lw=2, color=color)
    plt.axhline(0.0, color="gray", lw=0.8)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(alpha=0.25)
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()
    return out


def save_reward_plot(
    episode_returns: list[float],
    out_path: str | Path = "outputs/training_rewards.png",
    window: int = 20,
    color: str = "#1f77b4",
) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    values = np.asarray(episode_returns, dtype=np.float32)
    plt.figure(figsize=(9, 3))

    if values.size == 0:
        plt.text(0.5, 0.5, "No episode rewards collected", ha="center", va="center", transform=plt.gca().transAxes)
    else:
        x = np.arange(values.size, dtype=np.float32)
        plt.plot(x, values, lw=1.2, alpha=0.35, color=color)
        if values.size >= 2:
            k = max(2, min(int(window), int(values.size)))
            kernel = np.ones(k, dtype=np.float32) / float(k)
            smooth = np.convolve(values, kernel, mode="valid")
            xs = np.arange(k - 1, k - 1 + smooth.size, dtype=np.float32)
            plt.plot(xs, smooth, lw=2.2, color=color)

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()
    return out
