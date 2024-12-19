from torch.utils.data import Dataset
import torchio as tio
import torch
import monai
import os
from monai.transforms import Compose
from pathlib import Path
from glob import glob


class MedicalDataset(Dataset):
    def __init__(self, args, sequence, transform, volumes=None, segmentations=None):
        super().__init__()

        self.sequence = sequence

        self.transform = transform

        if self.transform:
            self.transforms = MedicalDataAugmentationDINO(local_crops_number=args.local_crops_number)

        if (volumes == None) & (segmentations==None):
        
            self.volumes = []
            self.segmentations = []

            # UCSF - All        
            self.volumes.extend(sorted(glob("/home/johannes/Code/3DMedicalDINO/data/UCSF-PDGM-Dataset/UCSF-PDGM/*/*/*T1c_skull_normalized.nii.gz")))
            self.segmentations.extend(sorted(glob("/home/johannes/Code/3DMedicalDINO/data/UCSF-PDGM-Dataset/UCSF-PDGM/*/*/segmentation_skull_normalized_hd_glio.nii.gz")))          
            
            # Erasmus - All
            self.volumes.extend(sorted(glob("/home/johannes/Code/3DMedicalDINO/data/erasmus/*/preop/*space-sri_t1c.nii.gz")))
            self.segmentations.extend(sorted(glob("/home/johannes/Code/3DMedicalDINO/data/erasmus/*/preop/*space-sri_seg.nii.gz")))        
        
        else:
            self.volumes = volumes
            self.segmentations = segmentations
    
    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx):

        t1c = self.volumes[idx]
        parent_path = Path(t1c).parent

        if "UCSF" in str(parent_path):
            t1w = glob(os.path.join(parent_path, "*T1_skull_normalized.nii.gz"))[0]
            t2w = glob(os.path.join(parent_path, "*T2_skull_normalized.nii.gz"))[0]
            t2f = glob(os.path.join(parent_path, "*FLAIR_skull_normalized.nii.gz"))[0]

        elif "erasmus" in str(parent_path):
            t1w = glob(os.path.join(parent_path, "*preop_space-sri_t1.nii.gz"))[0]
            t2w = glob(os.path.join(parent_path, "*preop_space-sri_t2.nii.gz"))[0]
            t2f = glob(os.path.join(parent_path, "*preop_space-sri_flair.nii.gz"))[0]
        
        elif "tcga" in str(parent_path):
            t1w = glob(os.path.join(parent_path, "*preop_space-sri_t1.nii.gz"))[0]
            t2w = glob(os.path.join(parent_path, "*preop_space-sri_t2.nii.gz"))[0]
            t2f = glob(os.path.join(parent_path, "*preop_space-sri_flair.nii.gz"))[0]

        subject = tio.Subject(
            t1c = tio.ScalarImage(t1c),
            t1w = tio.ScalarImage(t1w),
            t2w = tio.ScalarImage(t2w),
            t2f = tio.ScalarImage(t2f),
            seg = tio.LabelMap(self.segmentations[idx])
        )

        subject_aligned = tio.ToCanonical()(subject)
        subject_cropped = tio.CropOrPad(mask_name="seg")(subject_aligned)

        match self.sequence:
            case "T1":
                img_crop = subject_cropped.t1c.tensor
            
            case "T2":
                img_crop = subject_cropped.t2f.tensor
            
            case "T1T2":
                img_crop = torch.concatenate([subject_cropped.t1c.tensor, subject_cropped.t2f.tensor], dim=0)

            case "T1T1":
                img_crop = torch.concatenate([subject_cropped.t1c.tensor, subject_cropped.t1c.tensor], dim=0)

            case "T2T2":
                img_crop = torch.concatenate([subject_cropped.t2f.tensor, subject_cropped.t2f.tensor], dim=0)

            case "T1T2T1T2":
                img_crop = torch.concatenate([subject_cropped.t1c.tensor, 
                                              subject_cropped.t2f.tensor,
                                              subject_cropped.t1c.tensor, 
                                              subject_cropped.t2f.tensor], dim=0)
                
            case "all":
                img_crop = torch.concatenate([subject_cropped.t1c.tensor,
                                              subject_cropped.t1w.tensor,
                                              subject_cropped.t2w.tensor,
                                              subject_cropped.t2f.tensor], dim=0)        
        
        if self.transform:
            img_crop_list = self.transforms(img_crop)
            return img_crop_list
    
        else:
            return img_crop, t1c
    

class MedicalDataAugmentationDINO():
    def __init__(self, local_crops_number=0):
        
        # rotate = monai.transforms.RandRotate(prob=0.2, range_x=10, range_y=10, range_z=10)
        # flip = monai.transforms.RandFlip(prob=0.2, spatial_axis=1)
        # scale = monai.transforms.RandZoom(prob=0.2, min_zoom=0.7, max_zoom=1.4)
        gaussian_noise = monai.transforms.RandGaussianNoise()
        gaussian_blur = monai.transforms.RandGaussianSmooth(prob=0.2, sigma_x=(0.5, 1.0), sigma_y=(0.5, 10.0), sigma_z=(0.5, 1.0))
        contrast = monai.transforms.RandAdjustContrast()
        intensity = monai.transforms.RandScaleIntensity(factors=(2, 10))
        histogram_shift = monai.transforms.RandHistogramShift()
        
        normalize = monai.transforms.NormalizeIntensity()       
        totensor = monai.transforms.ToTensor()

        crop_global = monai.transforms.RandSpatialCrop(roi_size=96)
        pad_global = monai.transforms.SpatialPad(spatial_size=96)
        
        crop_local = monai.transforms.RandSpatialCrop(roi_size=32)
        pad_local = monai.transforms.SpatialPad(spatial_size=32)

        # first global crop
        self.global_transfo1 = Compose([gaussian_noise, gaussian_blur, contrast, intensity, histogram_shift, crop_global, normalize, pad_global, totensor])

        # second global crop
        self.global_transfo2 = Compose([gaussian_noise, gaussian_blur, contrast, intensity, histogram_shift, crop_global, normalize, pad_global, totensor])
        
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = Compose([gaussian_noise, gaussian_blur, contrast, intensity, histogram_shift, crop_local, normalize, pad_local, totensor])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops
