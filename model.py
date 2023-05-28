import torch
import torch.nn as nn
import numpy as np
import torchvision
from collections import OrderedDict


class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.class_num = 10

        ### initialize the BaseModel --------------------------------
        resnet18 = torchvision.models.resnet18(pretrained=True)
        layer1 = [module for module in resnet18.layer1.modules() if not isinstance(module, nn.Sequential)]
        self.backbone = nn.Sequential(OrderedDict([
            ('conv1', resnet18.conv1),
            ('bn1', resnet18.bn1),
            ('relu', resnet18.relu),
            ('maxpool', resnet18.maxpool),
            ('layer1_index0', layer1[0])]))
        self.dyn = nn.Sequential(nn.Linear(in_features=4096, out_features=576),
                                 nn.Tanh())
        self.dc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.cls = nn.Linear(in_features=64, out_features=10)

    def forward(self, imgs, with_dyn=True):
        if with_dyn:
            ''' with dynamic convolutional layer '''
            ### complete the forward path --------------------
            v = self.backbone(imgs)
            assert list(v.shape) == [imgs.shape[0], 64, 8, 8]  # Sanity check
            v_flat = torch.flatten(v, 1)
            w = self.dyn(v_flat)
            w = torch.norm(w, dim=0)
            w_conv = w.view(1, 64, 3, 3)
            self.dc.weight.data = w_conv
            out = self.dc(v)
            out = torch.flatten(out, 1)
            cls_scores = self.cls(out)

        else:
            ''' without dynamic convolutional layer '''
            ### complete the forward path --------------------
            v = self.backbone(imgs)
            assert list(v.shape) == [imgs.shape[0], 64, 8, 8]  # Sanity check
            v = self.dc(v)
            v = torch.flatten(v, 1)
            cls_scores = self.cls(v)

        return cls_scores


class ImprovedModel(nn.Module):
    def __init__(self, args):
        super(ImprovedModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.class_num = 10

        ### initialize the ImprovedModel --------------------------------
        resnet18 = torchvision.models.resnet18(pretrained=True)
        layer1 = [module for module in resnet18.layer1.modules() if not isinstance(module, nn.Sequential)]
        self.backbone = nn.Sequential(OrderedDict([
            ('conv1', resnet18.conv1),
            ('bn1', resnet18.bn1),
            ('relu', resnet18.relu),
            ('maxpool', resnet18.maxpool),
            ('layer1_index0', layer1[0])]))
        self.dc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(in_features=64, out_features=64)
        self.bn = nn.Dropout2d()
        self.cls = nn.Linear(in_features=64, out_features=10)

    def forward(self, imgs, with_dyn=True):
        ''' without dynamic convolutional layer '''
        v = self.backbone(imgs)
        assert list(v.shape) == [imgs.shape[0], 64, 8, 8]  # Sanity check
        v = self.dc(v)
        v = torch.flatten(v, 1)
        out = self.fc(v)
        out = self.bn(out)
        cls_scores = self.cls(out)

        return cls_scores


'''self.dc_additional = nn.Sequential(
            nn.Linear(in_features=1, out_features=64),
            nn.Dropout2d())'''