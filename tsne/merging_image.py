import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from sklearn.manifold import TSNE

# Load embeddings and labels
test_embeddings = torch.load(
    "/home/jackson-devworks/Desktop/CloudSense/tsne/z.pt"
)
z_q_test_embedding = torch.load(
    "/home/jackson-devworks/Desktop/CloudSense/tsne/z_quantized.pt"
)

action_targets = np.load(
    "/home/jackson-devworks/Desktop/CloudSense/tsne/action.npy"
)
subject_targets = np.load(
    "/home/jackson-devworks/Desktop/CloudSense/tsne/subject.npy"
)

# Action and subject mappings
action_map = {
    "A02": 0,
    "A03": 1,
    "A04": 2,
    "A05": 3,
    "A13": 4,
    "A14": 5,
    "A17": 6,
    "A18": 7,
    "A19": 8,
    "A20": 9,
    "A21": 10,
    "A22": 11,
    "A23": 12,
    "A27": 13,
}
subject_map = {
    "S01": 0,
    "S02": 1,
    "S03": 2,
    "S04": 3,
    "S05": 4,
    "S06": 5,
    "S07": 6,
    "S08": 7,
    "S09": 8,
    "S10": 9,
    "S11": 10,
    "S12": 11,
    "S13": 12,
    "S14": 13,
    "S15": 14,
    "S16": 15,
    "S17": 16,
    "S18": 17,
    "S19": 18,
    "S20": 19,
    "S21": 20,
    "S22": 21,
    "S23": 22,
    "S24": 23,
    "S25": 24,
    "S26": 25,
    "S27": 26,
    "S28": 27,
    "S29": 28,
    "S30": 29,
    "S31": 30,
    "S32": 31,
    "S33": 32,
    "S34": 33,
    "S35": 34,
    "S36": 35,
    "S37": 36,
    "S38": 37,
    "S39": 38,
    "S40": 39,
}
# Run t-SNE for both embeddings
tsne = TSNE(
    n_components=2,
    init="random",
    perplexity=26,
    metric="cosine",
    early_exaggeration=12,
    n_iter=1000,
)

# t-SNE projections
tsne_proj_test = tsne.fit_transform(test_embeddings)
tsne_proj_quantized = tsne.fit_transform(z_q_test_embedding)

# Plot settings
cmap = cm.get_cmap("tab20")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Plot for test embeddings
scatter1 = ax1.scatter(
    tsne_proj_test[:, 0],
    tsne_proj_test[:, 1],
    c=subject_targets,
    cmap=cmap,
    alpha=0.75,
)
ax1.set_xticks([])
ax1.set_yticks([])
# ax1.set_title("t-SNE for Test Embeddings")
ax1.grid(linestyle="--")

# Plot for quantized embeddings
scatter2 = ax2.scatter(
    tsne_proj_quantized[:, 0],
    tsne_proj_quantized[:, 1],
    c=subject_targets,
    cmap=cmap,
    alpha=0.75,
)
ax2.set_xticks([])
ax2.set_yticks([])
# ax2.set_title("t-SNE for Quantized Embeddings")
ax2.grid(linestyle="--")

# Save and show the plot
plt.savefig(
    "/home/jackson-devworks/Desktop/CloudSense/tsne/tsne_comparison_subject.pdf",
    dpi=500,
)
# plt.show()
