import monai
import torch
from typing import Literal, Union
from torch.nn import Module

import torch
import torch.nn as nn

import monai.networks
from monai.networks.nets import resnet


def get_model(model_name: Literal["DenseNet121", "ResNeXt50", "ResNet50", "ConvNeXt"], sequence: str) -> Union[Module, int]:

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
        
        case "ResNeXt50":
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
            student = monai.networks.nets.resnet.resnet50(pretrained=False, n_input_channels=in_channels, widen_factor=2, conv1_t_stride=2, feed_forward=False)            
            teacher = monai.networks.nets.resnet.resnet50(pretrained=False, n_input_channels=in_channels, widen_factor=2, conv1_t_stride=2, feed_forward=False)
            embedding_dim = 4096

        case "ConvNeXt":
            params = {
                "custom": {"depth": [2, 2, 6, 2], "dims": [96, 192, 384, 768]},
                "tiny": {"depth": [3, 3, 9, 3], "dims": [96, 192, 384, 768]},
                "small": {"depth": [3, 3, 27, 3], "dims": [96, 192, 384, 768]},
                "base": {"depth": [3, 3, 27, 3], "dims": [128, 256, 512, 1024]},
                "large": {"depth": [3, 3, 27, 3], "dims": [192, 384, 768, 1536]},
                "xlarge": {"depth": [3, 3, 27, 3], "dims": [256, 512, 1024, 2048]}
                }

            model_size = "large"
            student = ConvNeXt3D(in_chans=in_channels, depths=params[model_size]["depth"], dims=params[model_size]["dims"])
            teacher = ConvNeXt3D(in_chans=in_channels, depths=params[model_size]["depth"], dims=params[model_size]["dims"])
            embedding_dim = 1536
    
    return [student, teacher, embedding_dim]


# https://github.com/HusnuBarisBaydargil/ConvNext-Medical-Imaging/blob/main/convnext.py

class ConvNeXtBlockBase(nn.Module):
    def __init__(self, dim, conv_layer, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = conv_layer(dim, dim, kernel_size=7, padding=3, groups=dim) 
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim) 
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, *range(2, x.ndim), 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, -1, *range(1, x.ndim - 1)) 
        x = shortcut + self.drop_path(x)
        return x

class ConvNeXtBase(nn.Module):
    def __init__(self, in_chans, num_classes, depths, dims, conv_layer, pool_layer):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            conv_layer(in_chans, dims[0], kernel_size=4, stride=4),
            nn.GroupNorm(num_groups=1, num_channels=dims[0])
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.GroupNorm(num_groups=1, num_channels=dims[i]),
                conv_layer(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlockBase(dim=dims[i], conv_layer=conv_layer) for _ in range(depths[i])]
            )
            self.stages.append(stage)

        self.norm = nn.GroupNorm(num_groups=1, num_channels=dims[-1])
        # self.head = nn.Linear(dims[-1], num_classes)
        self.pool_layer = pool_layer

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = x.mean(dim=self.pool_layer)
        x = self.norm(x)
        # x = self.head(x)
        return x


class ConvNeXt3D(ConvNeXtBase):
    def __init__(self, in_chans=1, num_classes=2, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]):
        super().__init__(in_chans, num_classes, depths, dims, conv_layer=nn.Conv3d, pool_layer=[-1, -2, -3])

class ConvNeXt2D(ConvNeXtBase):
    def __init__(self, in_chans=1, num_classes=2, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]):
        super().__init__(in_chans, num_classes, depths, dims, conv_layer=nn.Conv2d, pool_layer=[-1, -2])


if __name__ == "__main__":

    params = {
        "custom": {"depth": [2, 2, 6, 2], "dims": [96, 192, 384, 768]},
        "tiny": {"depth": [3, 3, 9, 3], "dims": [96, 192, 384, 768]},
        "small": {"depth": [3, 3, 27, 3], "dims": [96, 192, 384, 768]},
        "base": {"depth": [3, 3, 27, 3], "dims": [128, 256, 512, 1024]},
        "large": {"depth": [3, 3, 27, 3], "dims": [192, 384, 768, 1536]},
        "xlarge": {"depth": [3, 3, 27, 3], "dims": [256, 512, 1024, 2048]}        
    }

    model_size = "custom"
    model = ConvNeXt3D(depths=params[model_size]["depth"], dims=params[model_size]["dims"])

    x = torch.randn(1, 1, 128, 128, 128) 
    output = model(x)

    print(f"Output shape: {output.shape}") 
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
