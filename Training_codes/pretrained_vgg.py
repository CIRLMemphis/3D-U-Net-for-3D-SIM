import torch
import torch.nn as nn
from config_train_3D import *
import torchvision.models as models

def perceptual_loss(pred_maps, gt_maps):
    for i in range(len(pred_maps)):
        if i == 0:
            perp_loss = F.mse_loss(gt_maps[i], pred_maps[i])
        else:
            perp_loss += F.mse_loss(gt_maps[i], pred_maps[i])
    
    return perp_loss

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

class PretrainedVGG(nn.Module):
    def __init__(self, content_layers=['conv_1', 'conv_3', 'conv_5', 'conv_9', 'conv_13']):
        super().__init__()
        norm_mean = torch.tensor([0.485, 0.456, 0.406])#.to(device)
        norm_std = torch.tensor([0.229, 0.224, 0.225])#.to(device)
        self._content_layers = set(content_layers)
        vgg =  models.vgg19(pretrained=True).features
        norm_layer = Normalization(norm_mean, norm_std)
        norm_layer.name="norm"
        self._vgg_layers = []
        self._vgg_layers.append(norm_layer)

        i = 0
        for layer in vgg.children():
            if isinstance(layer, nn.Conv2d):

                i += 1
                
                name = 'conv_{}'.format(i)
                # layer.padding = (0, 0)

            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            layer.name = name
            
            self._vgg_layers.append(layer)
        
        self._vgg_layers = nn.ModuleList(self._vgg_layers)

        #print("self._vgg_layers", self._vgg_layers)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        output_layers = []
        for layer in self._vgg_layers:
            x = layer(x)

            if layer.name in self._content_layers:
                output_layers.append(x)
            if len(output_layers) == len(self._content_layers):
                return output_layers

        return output_layers

