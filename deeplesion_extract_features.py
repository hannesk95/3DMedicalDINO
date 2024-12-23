import torch
import monai
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import get_model
from tqdm import tqdm
from glob import glob
from sklearn.manifold import TSNE
from dataset import DeepLesionDataset
from torchvision.transforms import Compose
from monai.transforms import ToTensor, EnsureChannelFirst, NormalizeIntensity, SpatialPad

if __name__ == "__main__":

    task = "task1"
    split = "train"
    model_name = "ResNet50"
    sequence = "deeplesion"

    data_path = f"/home/moonsurfer/Code/foundation-cancer-image-biomarker/data/preprocessing/deeplesion/annotations/{task}_{split}.csv"

    dataset = DeepLesionDataset(args=None, path=data_path, transform="inference", radius=25)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)  
    
    for checkpoint_path in glob(f"/home/moonsurfer/Code/3DMedicalDINO/results/{model_name}*checkpoint.pth"):

        print(f"Loading checkpoint: {checkpoint_path}")   

        _, model, _ = get_model(model_name=model_name, sequence=sequence)
        model.load_state_dict(torch.load(checkpoint_path, map_location="cuda", weights_only=False)["teacher"], strict=False)
        model.cuda()

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

        torch.save(features, f"/home/moonsurfer/Code/3DMedicalDINO/features/{task}_{split}_features.pth")
        torch.save(labels, f"/home/moonsurfer/Code/3DMedicalDINO/features/{task}_{split}_labels.pth")
