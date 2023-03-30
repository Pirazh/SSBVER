import warnings
import argparse
from copy import deepcopy

import numpy as np
from PIL import Image
import cv2
import os.path as osp

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torchvision import transforms as TF

from models import resnet, resnet_ibn_a


class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, neck, target_layers):
        self.model = model
        self.neck = neck
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.feature_extractor = FeatureExtractor(self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        output_map, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.neck(output)
        return output, output_map[0]

def show_cam_on_image(img, mask, save_path=None):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(save_path, np.uint8(255 * cam))


class GradCam:
    def __init__(self, model_q, model_g, neck_q, neck_g, target_layer_names, use_cuda=True):
        self.model_q = model_q
        self.model_g = model_g
        self.neck_q = neck_q
        self.neck_g = neck_g
        self.model_q.eval()
        self.model_g.eval()
        self.neck_q.eval()
        self.neck_g.eval()
        self.cuda = use_cuda
        self.extractor = ModelOutputs(self.model_q, self.neck_q, target_layer_names)

    def __call__(self, input_q, input_g):
        self.model_q.eval()
        self.model_g.eval()
        self.neck_q.eval()
        self.neck_g.eval()
        self.model_q.zero_grad()
        self.model_g.zero_grad()
        self.neck_q.zero_grad()
        self.neck_g.zero_grad()
        query_features, output_map = self.extractor(input_q)
        gallery_features = self.neck_g(self.model_g(input_g))
        query_features = F.normalize(query_features, dim=1)
        gallery_features = F.normalize(gallery_features, dim=1)
        one_hot = (query_features * gallery_features).sum()
        grad = torch.autograd.grad(one_hot, output_map,
                                retain_graph=True, create_graph=True)[0]
        
        weights = grad.detach().mean(dim=2).mean(dim=2).unsqueeze(dim=-1).unsqueeze(dim=-1)
        cam = weights * output_map.detach()
        cam = cam.sum(dim=1)
        cam = torch.where(cam > 0, cam, torch.zeros_like(cam))
        cam = cam[0].cpu().numpy()
        """
        #one_hot.backward(retain_graph=True)
        #grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        grads_val = grad.cpu().data.numpy()

        target = output_map[-1] # 2048 * 16 * 16
        target = target.cpu().data.numpy()

        weights = np.mean(grads_val, axis=(2, 3))[0, :] # 2048
        cam = np.zeros(target.shape[1:], dtype=np.float32)


        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        """
        cam = cv2.resize(cam, (256, 256))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        return cam


def get_model(args):
    if args.model_arc == 'resnet50':
        model = resnet.__dict__[args.model_arc](args)
    elif args.model_arc == 'resnet50_ibn_a':
        model = resnet_ibn_a.__dict__[args.model_arc](args)
    try:
        ckpt = torch.load(args.ckpt_path, map_location='cpu')['student_ema']
    except:
        ckpt = torch.load(args.ckpt_path, map_location='cpu')['teacher']
    state_dict, bn_dict = {}, {}
    for k,v in ckpt.items():
        if 'backbone' in k:
            state_dict[k.replace('module.backbone.', '')] = v
        if 'bottleneck' in k:
            bn_dict[k.replace('module.reid_head.bottleneck.', '')] = v
    m = model.load_state_dict(state_dict)
    print(m)
    model = model.cuda()
    model.eval()

    embed_dim = model.layer4[2].conv3.weight.shape[0]
    bottleneck = nn.BatchNorm1d(embed_dim).cuda()
    bottleneck.bias.requires_grad_(False)
    m = bottleneck.load_state_dict(bn_dict)
    print(m)
    return model, bottleneck

def build_tf(args):
    tf = TF.Compose([TF.Resize((256, 256), interpolation=Image.BICUBIC),
                TF.ToTensor(),
                TF.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    return tf

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_arc', type=str, default='resnet50_ibn_a', 
                        choices=['resnet50_ibn_a', 'resnet50'])
    parser.add_argument('--last_stride', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, default='./results/VeRi_resnet50_ibn_a/ckpt_epoch_046.pth')
    parser.add_argument('--query_path', type=str, default='')
    parser.add_argument('--gallery_path', type=str, default='')
    parser.add_argument('--save_path', type=str, default='./')
    args = parser.parse_args()

    model_q, bottleneck_q = get_model(args)
    model_g, bottleneck_g = deepcopy(model_q), deepcopy(bottleneck_q)
    

    im_transform = build_tf(args)

    query_im = im_transform(Image.open(args.query_path)).unsqueeze(0).cuda()
    query_im.requires_grad_(True)
    gallery_im = im_transform(Image.open(args.gallery_path)).unsqueeze(0).cuda()
    gallery_im.requires_grad_(True)

    path_to_query_img, query_img_name = osp.split(args.query_path)
    path_to_gallery_img, gallery_img_name = osp.split(args.gallery_path)

    save_path_query = osp.join(args.save_path, 
                    query_img_name.strip('.jpg') + '_query_proposed.jpg')
    save_path_gallery = osp.join(args.save_path, 
                    gallery_img_name.strip('.jpg') + '_gallery_proposed.jpg')

    grad_cam_q = GradCam(model_q, model_g, bottleneck_q, bottleneck_g, 
                    target_layer_names=["layer4"])
    grad_cam_g = GradCam(model_g, model_q, bottleneck_g, bottleneck_q ,
                    target_layer_names=["layer4"])

    mask_q = grad_cam_q(query_im, gallery_im)
    
    query_im = query_im.detach()
    gallery_im = gallery_im.detach()
    
    
    mask_g = grad_cam_g(gallery_im, query_im)

    img_q = cv2.imread(args.query_path, 1)
    img_q = np.float32(cv2.resize(img_q, (256, 256))) / 255
    
    img_g = cv2.imread(args.gallery_path, 1)
    img_g = np.float32(cv2.resize(img_g, (256, 256))) / 255

    show_cam_on_image(img_q, mask_q, save_path=save_path_query)
    show_cam_on_image(img_g, mask_g, save_path=save_path_gallery)