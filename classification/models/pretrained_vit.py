import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

#By defaultn torch  typically uses a cache directory inside your home directory ~/.cache/torch/hub
# torch.hub.set_dir("/goinfre/ryagoub/torch_cache")this command tell torch to use this directory instead
# mkdir -p /goinfre/$USER/torch_cache


class Pretrained_VisionTransformers(nn.Module):
    def __init__(self):
        super().__init__()
        torch.hub.set_dir("/goinfre/ryagoub/torch_cache")
        weights = ViT_B_16_Weights.DEFAULT
        self.vit = vit_b_16(weights=weights)
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.heads.head = nn.Linear(768, 8)

    def forward(self, x):
        return self.vit(x)
