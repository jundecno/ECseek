import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_binary_mask(mask, save_path="mask.png"):
    """
    mask:
        List[List[float]] or np.ndarray
    """

    mask = np.asarray(mask)

    # >0.5 红色，<=0.5 蓝色
    binary_mask = (mask > 0.5).astype(int)

    # 深蓝 + 深红（避免亮色）
    cmap = ListedColormap(
        [
            "#1f3b73",  # dark blue
            "#8b1e1e",  # dark red
        ]
    )

    plt.figure(figsize=(8, 8))

    plt.imshow(binary_mask, cmap=cmap, interpolation="nearest", aspect="auto")

    plt.xlabel("Reaction")
    plt.ylabel("Pocket")

    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved to: {save_path}")


