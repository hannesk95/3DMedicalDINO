import monai
import torch
from typing import Literal, Union
from torch.nn import Module

import monai.networks
from monai.networks.nets import resnet


def get_model(model_name: Literal["DenseNet121", "ResNext50", "ResNet50"], sequence: str) -> Union[Module, int]:

    if sequence == "deeplesion":
        in_channels = 1
    
    else:    
        if sequence != "all":
            in_channels = len(sequence) // 2
        else:
            in_channels = 4

    

    match model_name:
        case "DenseNet121":
            student = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=in_channels, out_channels=2).features
            student = torch.nn.Sequential(*list(student.children()), torch.nn.AdaptiveAvgPool3d(output_size=1), torch.nn.Flatten())
            teacher = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=in_channels, out_channels=2).features
            teacher = torch.nn.Sequential(*list(teacher.children()), torch.nn.AdaptiveAvgPool3d(output_size=1),torch.nn.Flatten())
            embedding_dim = 1024            
        
        case "ResNext50":
            student = monai.networks.nets.SEResNext50(spatial_dims=3, in_channels=in_channels)
            student = torch.nn.Sequential(*list(student.children())[:-1], torch.nn.Flatten())            
            teacher = monai.networks.nets.SEResNext50(spatial_dims=3, in_channels=in_channels)
            teacher = torch.nn.Sequential(*list(teacher.children())[:-1], torch.nn.Flatten())
            embedding_dim = 2048

        case "ResNet50_plain":
            student = resnet.resnet50(spatial_dims=3, n_input_channels=in_channels, num_classes=2)
            student = torch.nn.Sequential(*list(student.children())[:-1], torch.nn.Flatten())             
            teacher = resnet.resnet50(spatial_dims=3, n_input_channels=in_channels, num_classes=2)
            teacher = torch.nn.Sequential(*list(teacher.children())[:-1], torch.nn.Flatten()) 
            embedding_dim = 2048

        case "ResNet50":
            student = monai.networks.nets.resnet.resnet50(pretrained=False, n_input_channels=1, widen_factor=2, conv1_t_stride=2, feed_forward=False)            
            teacher = monai.networks.nets.resnet.resnet50(pretrained=False, n_input_channels=1, widen_factor=2, conv1_t_stride=2, feed_forward=False)
            embedding_dim = 4096

        case "ConvNeXt":
            raise NotImplementedError("ConvNeXt not implemented yet")
    
    return [student, teacher, embedding_dim]
