import torch.nn as nn
import timm
import torch
import numpy  as np


# 定义模型类
class TimmModel(nn.Module):
    def __init__(self, model_name='convnext_base.fb_in22k_ft_in1k_384', pretrained=False, img_size=384):
        super(TimmModel, self).__init__()
        self.img_size = img_size
        if "vit" in model_name:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, img1, img2=None):
        if img1 is not None and img2 is not None:
            image_features1 = self.model(img1)
            image_features2 = self.model(img2)
            return image_features1, image_features2
        elif img2 is None:
            image_features1 = self.model(img1)
            return image_features1, None
        else:
            image_features2 = self.model(img2)
            return None, image_features2


