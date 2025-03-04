import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import get_model
from tqdm import tqdm
from glob import glob
from sklearn.manifold import TSNE
from dataset import DeepLesionDataset

if __name__ == "__main__":

    model_name = "ResNet50"
    sequence = "deeplesion"
    data_path = "/home/moonsurfer/Code/foundation-cancer-image-biomarker/data/preprocessing/deeplesion/annotations/task1_test.csv"

    dataset = DeepLesionDataset(args=None, path=data_path, transform="inference", radius=25)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1) 

    for checkpoint_path in tqdm(sorted(glob(f"/home/moonsurfer/Code/3DMedicalDINO/results/{model_name}*checkpoint*"))):

        if os.path.exists(f"results/{checkpoint_path.split('/')[-1].split('.')[0]}.png"):
            continue

        print(f"Loading checkpoint: {checkpoint_path}") 

        _, model, _ = get_model(model_name=model_name, sequence=sequence)
        model.load_state_dict(torch.load(checkpoint_path, weights_only=False)["teacher"], strict=False)
        model.cuda()
        model.eval()               

        features = []
        labels = []
        dims = []
        df = pd.read_csv(data_path)

        for volume, filepath in tqdm(dataloader):                

            file_id = filepath[0].split("/")[-1]
            label = df[df["Volume_fn"] == file_id]["Coarse_lesion_type"].iloc[0].item()
            volume = volume.to(torch.float32).cuda()
            out = model(volume) 
            labels.append(label)
            features.append(out.detach().cpu())                

        features = torch.concatenate(features)        
        labels = torch.from_numpy(np.array(labels))            

        data_np = features.cpu().numpy()
        labels_np = labels.cpu().numpy()

        # Perform t-SNE to reduce dimensionality to 2D
        print("Running t-SNE...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        data_2d = tsne.fit_transform(data_np)

        # Visualization
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels_np, cmap="tab10", alpha=0.7)
        plt.colorbar(scatter, ticks=range(8), label="Labels")
        plt.title(f"{checkpoint_path}")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.grid(True)
        plt.savefig(f"results/{checkpoint_path.split('/')[-1].split('.')[0]}.png")
        # plt.show()
        plt.close()    