import argparse
import os
import warnings

import cv2
import numpy as np
import torch
import torch.autograd as autograd
from torchvision import transforms as TF
from PIL import Image
import matplotlib.pyplot as plt

from configs import get_configs
from tools import log
from data import build_data
from models import build_models


def build_tf():
    tf = TF.Compose([TF.Resize((256, 256), interpolation=Image.BICUBIC),
                TF.ToTensor(),
                TF.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    return tf

def show_heatmap_on_image(img, mask):
    """both img and mask should be between 0,1"""
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

def main(args):

    _, _, num_train_classes = build_data(args)
    student, teacher = build_models(args, num_classes=num_train_classes)
    ckpt = torch.load(args.test_ckpt)
    student.load_state_dict(ckpt[args.test_model], strict=False)
    print('{} model is loaded from {}'.format(args.test_model, args.test_ckpt))
    student = student.module.backbone
    student.eval()

    layer_to_filter_id = {}
    ind = 0
    for _, (name, param) in enumerate(student.named_parameters()):
        if len(param.size()) == 4:
            for j in range(param.size()[0]):
                if name not in layer_to_filter_id:
                    layer_to_filter_id[name] = [ind + j]
                else:
                    layer_to_filter_id[name].append(ind + j)
            ind += param.size()[0]
    
    total = 0
    for layer in layer_to_filter_id:
        total += len(layer_to_filter_id[layer])
    print('Total filters:', total)
    print('Total layers:', len(layer_to_filter_id))
    
    im_transform = build_tf()

    query_im = im_transform(Image.open(args.query_path)).unsqueeze(0).cuda()
    query_im.requires_grad_(True)
    gallery_im = im_transform(Image.open(args.gallery_path)).unsqueeze(0).cuda()
    gallery_im.requires_grad_(True)


    student.eval()
    student.zero_grad()
    query_feat = student(query_im)
    gallery_feat = student(gallery_im)

    sim_score = torch.mm(query_feat, gallery_feat.T) / (query_feat.norm() * gallery_feat.norm())
    print(sim_score.item())

    gradients_query = autograd.grad(sim_score, query_im, create_graph=True)
    gradients_gallery = autograd.grad(sim_score, gallery_im, create_graph=True)

    grads_gall_to_save = gradients_gallery[0].detach().cpu()
    grads_quer_to_save = gradients_query[0].detach().cpu()
    
    grads_gall_to_save, _ = grads_gall_to_save.max(dim=1)
    grads_quer_to_save, _ = grads_quer_to_save.max(dim=1)

    grads_gall_to_save = grads_gall_to_save.detach().cpu().numpy().reshape((256, 256))
    grads_quer_to_save = grads_quer_to_save.detach().cpu().numpy().reshape((256, 256))
    
    grads_gall_to_save = np.abs(grads_gall_to_save)
    grads_quer_to_save = np.abs(grads_quer_to_save)

    grads_gall_to_save[grads_gall_to_save > np.percentile(grads_gall_to_save, 99)] = np.percentile(grads_gall_to_save, 99)
    grads_gall_to_save[grads_gall_to_save < np.percentile(grads_gall_to_save, 90)] = np.percentile(grads_gall_to_save, 90)

    grads_quer_to_save[grads_quer_to_save > np.percentile(grads_quer_to_save, 99)] = np.percentile(grads_quer_to_save, 99)
    grads_quer_to_save[grads_quer_to_save < np.percentile(grads_quer_to_save, 90)] = np.percentile(grads_quer_to_save, 90)

    _, gallery_name = os.path.split(args.gallery_path)
    _, query_name = os.path.split(args.query_path)


    plt.figure()
    plt.imshow(grads_gall_to_save)
    plt.axis('off')
    grad_gall_save_path = './saliency/figures/{}_grad_gallery.jpg'.format(gallery_name.split('.')[0])
    #plt.savefig(grad_gall_save_path, bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.imshow(grads_quer_to_save)
    plt.axis('off')
    grad_quer_save_path = './saliency/figures/{}_grad_query.jpg'.format(query_name.split('.')[0])
    #plt.savefig(grad_quer_save_path, bbox_inches='tight')
    plt.close()



    grads_gall_to_save = (grads_gall_to_save - np.min(grads_gall_to_save)) / (np.max(grads_gall_to_save) - np.min(grads_gall_to_save))
    grads_quer_to_save = (grads_quer_to_save - np.min(grads_quer_to_save)) / (np.max(grads_quer_to_save) - np.min(grads_quer_to_save))


    #Superimpose gradient heatmap
    reference_gall_image_to_compare = np.array(Image.open(args.gallery_path).resize((256,256))).astype(np.float) / 255
    gradients_gall_heatmap = np.ones_like(grads_gall_to_save) - grads_gall_to_save
    gradients_gall_heatmap = cv2.GaussianBlur(gradients_gall_heatmap, (3, 3), 0)
    heatmap_gall_superimposed = show_heatmap_on_image(reference_gall_image_to_compare, gradients_gall_heatmap)

    plt.figure()
    plt.imshow(grads_gall_to_save)
    plt.imshow(heatmap_gall_superimposed)
    plt.axis('off')
    im_gall_save_path = './saliency/figures/{}_superimposed_gallery.jpg'.format(gallery_name.split('.')[0])
    plt.savefig(im_gall_save_path, bbox_inches='tight')
    plt.close()


    #Superimpose gradient heatmap
    reference_quer_image_to_compare = np.array(Image.open(args.query_path).resize((256,256))).astype(np.float) / 255
    gradients_quer_heatmap = np.ones_like(grads_quer_to_save) - grads_quer_to_save
    gradients_quer_heatmap = cv2.GaussianBlur(gradients_quer_heatmap, (3, 3), 0)
    heatmap_quer_superimposed = show_heatmap_on_image(reference_quer_image_to_compare, gradients_quer_heatmap)
    plt.figure()
    plt.imshow(grads_quer_to_save)
    plt.imshow(heatmap_quer_superimposed)
    plt.axis('off')
    im_quer_save_path = './saliency/figures/{}_superimposed_query.jpg'.format(query_name.split('.')[0])
    plt.savefig(im_quer_save_path, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(description='', 
                                        parents=[get_configs()])
    main(parser.parse_args())