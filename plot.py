import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Dados por seed
data = {
    "Modelo": [
        "CLIPDenoising (RN50)",
        "CLIPDenoising (RN50)",
        "CLIPDenoising (RN50)",
        "CLIPDenoising (RN50)",
        "CLIPDenoising (RN50)",
        "CLIPDenoising-Simple (ViT)",
        "CLIPDenoising-Simple (ViT)",
        "CLIPDenoising-Simple (ViT)",
        "CLIPDenoising-Simple (ViT)",
        "CLIPDenoising-Simple (ViT)",
    ],
    "Seed": [100, 200, 300, 400, 500, 100, 200, 300, 400, 500],
    "PSNR": [45.85, 42.03, 45.71, 44.33, 44.33, 45.89, 45.23, 44.37, 45.75, 45.75],
    "SSIM": [0.976, 0.952, 0.976, 0.973, 0.968, 0.977, 0.975, 0.975, 0.976, 0.976],
}

df = pd.DataFrame(data)

# Configurações globais
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# Paleta mais sóbria para publicação
palette = sns.color_palette("muted")

# PSNR
sns.boxplot(
    x="Modelo",
    y="PSNR",
    data=df,
    palette=palette,
    ax=axes[0],
    width=0.5,
    linewidth=1.2,
    whis=1.5,
    fliersize=0,  # outliers mostrados pelo stripplot
)
sns.stripplot(
    x="Modelo",
    y="PSNR",
    data=df,
    color="0.3",
    size=5,
    alpha=0.7,
    jitter=0.15,
    ax=axes[0],
)
axes[0].set_xlabel("")
axes[0].set_ylabel("PSNR (dB)")
axes[0].tick_params(axis="x", rotation=25)
axes[0].spines[["top", "right"]].set_visible(False)
axes[0].grid(axis="y", linestyle="--", alpha=0.4)

# SSIM
sns.boxplot(
    x="Modelo",
    y="SSIM",
    data=df,
    palette=palette,
    ax=axes[1],
    width=0.5,
    linewidth=1.2,
    whis=1.5,
    fliersize=0,  # outliers mostrados pelo stripplot
)
sns.stripplot(
    x="Modelo",
    y="SSIM",
    data=df,
    color="0.3",
    size=5,
    alpha=0.7,
    jitter=0.15,
    ax=axes[1],
)
axes[1].set_xlabel("")
axes[1].set_ylabel("SSIM")
axes[1].tick_params(axis="x", rotation=25)
axes[1].spines[["top", "right"]].set_visible(False)
axes[1].grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("boxplot_psnr_ssim.eps", format="eps", bbox_inches="tight")
plt.savefig("boxplot_psnr_ssim.pdf", format="pdf", bbox_inches="tight")
plt.show()
