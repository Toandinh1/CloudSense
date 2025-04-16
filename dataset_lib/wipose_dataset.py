import os

import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class WiPoseDataset(Dataset):
    def __init__(self, root_dir, split="Train"):
        self.root_dir = root_dir
        self.split = split
        self.file_list = os.listdir(os.path.join(root_dir, split))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.split, self.file_list[idx])
        with h5py.File(file_path, "r") as file:
            CSI_data = torch.Tensor(
                file["CSI"][:]
            )  # Assuming 'CSI' is a dataset in your file
            CSI_data = CSI_data.view(9, 30, 5)
            keypoints = torch.Tensor(
                file["SkeletonPoints"][:]
            )  # Assuming 'SkeletonPoints' is a dataset in your file
            keypoints = keypoints.unsqueeze(0)
            keypoints = keypoints.view(1, -1, 3)
            keypoints = keypoints.squeeze(0)
        return {"input_wifi-csi": CSI_data, "output": keypoints}


if __name__ == "__main__":
    data = WiPoseDataset("/home/jackson-devworks/Desktop/HPE/Wi-Pose")
    print(data[0]["output"].shape)
