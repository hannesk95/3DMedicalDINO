from torch.utils.data import Dataset
import torchio as tio
import torch
import monai
import os
from monai.transforms import Compose
from pathlib import Path
from glob import glob
import pandas as pd
import SimpleITK as sitk
from utils import resample_image_to_spacing, slice_image, RandomResizedCrop3D
from monai.transforms import ToTensor, EnsureChannelFirst, NormalizeIntensity, SpatialPad, CenterSpatialCrop


class GliomaDataset(Dataset):
    def __init__(self, args, sequence, transform, volumes=None, segmentations=None):
        super().__init__()

        self.sequence = sequence

        self.transform = transform

        if self.transform == "train":
            self.transforms = GliomaDataAugmentationDINO(local_crops_number=args.local_crops_number)
        elif self.transform == "inference":
            self.transforms = Compose([CenterSpatialCrop(roi_size=(96, 96, 96)),                
                                       NormalizeIntensity(),
                                       SpatialPad(spatial_size=(96, 96, 96)),
                                       ToTensor(),])

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
        
        if self.transform == "train":
            img_crop_list = self.transforms(img_crop)
            return img_crop_list
    
        else:
            img_crop = self.transforms(img_crop)
            return img_crop, t1c
        
class GliomaDataAugmentationDINO():
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
        

class DeepLesionDataset(Dataset):
    def __init__(self, args, path, transform, radius=35):
        super().__init__()

        self.transform = transform

        if self.transform == "train":
            self.transforms = DeepLesionDataAugmentationDINO(local_crops_number=args.local_crops_number)
        elif self.transform == "inference":
            self.transforms = Compose([ToTensor(),
                                       EnsureChannelFirst(channel_dim='no_channel'),
                                       NormalizeIntensity(subtrahend=-1024, divisor=3072),
                                       SpatialPad(spatial_size=(50, 50, 50))])
        else:
            self.transforms = None

        self.annotations = pd.read_csv(path)
        self.resample_spacing = (1, 1, 1)
        self.orient = True
        self.orient_patch = True
        self.radius = radius

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        
        # Get a row from the CSV file
        row = self.annotations.iloc[idx]
        image_path = row["image_path"]
        image = sitk.ReadImage(str(image_path))
        image = resample_image_to_spacing(image, self.resample_spacing, -1024) if self.resample_spacing is not None else image

        centroid = (row["coordX"], row["coordY"], row["coordZ"])
        centroid = image.TransformPhysicalPointToContinuousIndex(centroid)
        centroid = [int(d) for d in centroid]

        # Orient all images to LPI orientation
        image = sitk.DICOMOrient(image, "LPI") if self.orient else image

        # Extract positive with a specified radius around centroid
        patch_idx = [(c - self.radius, c + self.radius) for c in centroid]
        patch_image = slice_image(image, patch_idx)

        patch_image = sitk.DICOMOrient(patch_image, "LPS") if self.orient_patch else patch_image

        array = sitk.GetArrayFromImage(patch_image)   

        # tensor = torch.unsqueeze(torch.from_numpy(array), dim=0) if self.transform is False else self.transforms(torch.unsqueeze(torch.from_numpy(array), dim=0))

        tensor = torch.unsqueeze(torch.from_numpy(array), dim=0) if self.transforms is None else self.transforms(array)

        if self.transform == "inference":
            return tensor, image_path
        else:
            return tensor

        # if self.transform:
        #     tensor = self.transforms(torch.unsqueeze(torch.from_numpy(array), dim=0))
        #     tensor = [temp.as_tensor() for temp in tensor]
        #     return tensor 
        
        # else:
        #     # tensor = torch.unsqueeze(torch.from_numpy(array), dim=0)

        #     my_transform = Compose([
        #         ToTensor(),
        #         EnsureChannelFirst(channel_dim='no_channel'),
        #         NormalizeIntensity(subtrahend=-1024, divisor=3072),
        #         # SpatialPad(spatial_size=(50, 50, 50)),
        #     ])
            
        #     tensor = my_transform(torch.unsqueeze(torch.from_numpy(array), dim=0))
        #     return tensor, image_path        

           
    

class DeepLesionDataAugmentationDINO():
    def __init__(self, local_crops_number=0):

        crop_resize_global = RandomResizedCrop3D(size=50)
        crop_resize_local = RandomResizedCrop3D(size=20)
        flip = monai.transforms.RandAxisFlip(prob=0.5)
        shift = monai.transforms.RandHistogramShift(prob=0.5)
        smooth = monai.transforms.RandGaussianSmooth(prob=0.5)
        pad_global = monai.transforms.SpatialPad(spatial_size=(50,50,50))
        pad_local = monai.transforms.SpatialPad(spatial_size=(20,20,20))
        threshold = monai.transforms.ThresholdIntensity(threshold=-1024, cval=-1024)
        normalize = monai.transforms.NormalizeIntensity(subtrahend=-1024, divisor=3072)
        channel_first = monai.transforms.EnsureChannelFirst(channel_dim='no_channel')
        totensor = monai.transforms.ToTensor()

        # first global crop
        self.global_transfo1 = Compose([totensor, channel_first, 
                                        crop_resize_global, flip, shift, smooth, pad_global, threshold, normalize])

        # second global crop
        self.global_transfo2 = Compose([totensor, channel_first, 
                                        crop_resize_global, flip, shift, smooth, pad_global, threshold, normalize])
        
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = Compose([totensor, channel_first, 
                                      crop_resize_local, flip, shift, smooth, pad_local, threshold, normalize])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops

