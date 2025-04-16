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

# Mapping dictionaries
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

# Map action labels to integers using action_map
# action_targets = np.array(
#     [action_map.get(action, -1) for action in action_label]
# )

# Initialize and run TSNE
tsne = TSNE(
    n_components=2,
    init="random",
    perplexity=26,
    metric="cosine",
    early_exaggeration=12,
    n_iter=1000,
)
tsne_proj = tsne.fit_transform(test_embeddings)

# Plot t-SNE projection
cmap = cm.get_cmap("tab20")
fig, ax = plt.subplots(figsize=(15, 15))
ax.set_xticks([])
ax.set_yticks([])

# Scatter plot with color based on action targets
scatter = ax.scatter(
    tsne_proj[:, 0],
    tsne_proj[:, 1],
    c=action_targets,
    cmap=cmap,
    alpha=0.75,
)

# Add color legend
handles, labels = scatter.legend_elements()
ax.legend(handles, list(action_map.keys()), title="Actions", fontsize="large")

# Optional: Add grid and title
ax.grid(linestyle="--")

# Save the plot
plt.savefig(
    "/home/jackson-devworks/Desktop/CloudSense/tsne/z_action_tsne.jpg",
    dpi=300,
)
plt.show()
