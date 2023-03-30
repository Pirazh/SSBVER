import os
import sys
import argparse
import warnings

import skimage.io
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from PIL import Image
import pdb

from configs import get_configs
from att_tools import display_instances


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(description='Visualize Self-Attention maps for ViTs', 
                                        parents=[get_configs()])
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids

    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

    from data import build_data
    from models import build_models
    _, val_loader, num_train_classes = build_data(args)
    student, _ = build_models(args, num_classes=num_train_classes)
    ckpt = torch.load(args.test_ckpt)
    m = student.load_state_dict(ckpt[args.test_model])
    for p in student.parameters():
        p.requires_grad = False
    student.eval()
    print('Student model is loaded from {} with this message: {}'.format(args.test_ckpt, m))
    if os.path.isfile(args.image_path):
        with open(args.image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
    else:
        print(f"Provided image path {args.image_path} is non valid.")
        sys.exit(1)
    transform = val_loader.dataset.transform
    img = transform(img)
    w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
    img = img[:, :w, :h].unsqueeze(0)
    w_featmap = img.shape[-2] // args.patch_size
    h_featmap = img.shape[-1] // args.patch_size

    attentions = student.module.backbone.get_last_selfattention(img.cuda())
    nh = attentions.shape[1] # number of head
    
    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    attentions /= attentions.max(dim=1)[0].unsqueeze(dim=1)
    attentions = attentions.sum(dim=0).unsqueeze(dim=0)

    nh = attentions.shape[0]

    if args.threshold is not None:
        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - args.threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        # interpolate
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

    # save attentions heatmaps
    os.makedirs(args.output_attn_dir, exist_ok=True)
    torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(args.output_attn_dir, "img.png"))
    for j in range(nh):
        fname = os.path.join(args.output_attn_dir, "attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format='png')
        print(f"{fname} saved.")

    if args.threshold is not None:
        image = skimage.io.imread(os.path.join(args.output_attn_dir, "img.png"))
        for j in range(nh):
            display_instances(image, th_attn[j], fname=os.path.join(args.output_attn_dir, "mask_th" + str(args.threshold) + "_head" + str(j) +".png"), blur=False)