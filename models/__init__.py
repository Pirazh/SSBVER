from email.policy import strict
import torch
from torch import nn

from . import vit, resnet, resnet_ibn_a, swin_transformer, convnext
from .head import SSLHead, ReIDHead
from .wrapper import MultiCropWrapper


def build_models(args, num_classes=None):
    if 'vit' in args.model_arc:
        student = vit.__dict__[args.model_arc](patch_size=args.patch_size,
                                            drop_path_rate=args.drop_path_rate)
        teacher = vit.__dict__[args.model_arc](patch_size=args.patch_size)
        embed_dim = student.embed_dim

    elif args.model_arc == 'resnet50':
        student = resnet.__dict__[args.model_arc](args)
        teacher = resnet.__dict__[args.model_arc](args)
        embed_dim = student.layer4[2].conv3.weight.shape[0]

    elif args.model_arc == 'resnet50_ibn_a':
        student = resnet_ibn_a.__dict__[args.model_arc](args)
        teacher = resnet_ibn_a.__dict__[args.model_arc](args)
        embed_dim = student.layer4[2].conv3.weight.shape[0]
    
    elif 'swin' in args.model_arc:
        student = swin_transformer.__dict__[args.model_arc](args)
        teacher = swin_transformer.__dict__[args.model_arc](args)
        embed_dim = student.norm.weight.shape[0]
    
    elif 'convnext' in args.model_arc:
        student = convnext.__dict__[args.model_arc](pretrained=True)
        teacher = convnext.__dict__[args.model_arc](pretrained=True)
        embed_dim = student.norm.weight.shape[0]

    student_ssl_head = SSLHead(in_dim=embed_dim,
                                out_dim=args.ssl_dim,
                                use_bn=args.use_bn_in_head,
                                norm_last_layer=args.norm_last_layer)
    
    student_reid_head = ReIDHead(num_classes=num_classes,
                        embed_dim=embed_dim,
                        neck=args.neck,
                        neck_feat=args.neck_feat)

    teacher_ssl_head = SSLHead(in_dim=embed_dim,
                                out_dim=args.ssl_dim,
                                use_bn=args.use_bn_in_head)
    
    teacher_reid_head = ReIDHead(num_classes=num_classes,
                        embed_dim=embed_dim,
                        neck=args.neck,
                        neck_feat=args.neck_feat)

    student = MultiCropWrapper(student, student_ssl_head, student_reid_head)
    teacher = MultiCropWrapper(teacher, teacher_ssl_head, teacher_reid_head,
                                                            is_student=False)

    if args.pretrained:
        if args.pretrained_method == 'ImageNet':
            if args.model_arc == 'vit_small':
                ckpt = torch.load('./models/pretrained_weights/deit_small.pth')['model']
            elif args.model_arc == 'vit_base':
                ckpt = torch.load('./models/pretrained_weights/jx_vit_base_p16_224-80ecf9dd.pth')
            elif args.model_arc == 'resnet50':
                ckpt = torch.load('./models/pretrained_weights/resnet50-0676ba61.pth')
            elif args.model_arc == 'resnet101':
                ckpt = torch.load('./models/pretrained_weights/resnet101-63fe2227.pth')
            elif args.model_arc == 'resnet50_ibn_a':
                ckpt = torch.load('./models/pretrained_weights/resnet50_ibn_a-d9d0bb7b.pth', map_location='cpu')
            elif args.model_arc == 'swin_base':
                ckpt = torch.load('./models/pretrained_weights/swin_base_patch4_window7_224_22k.pth', 
                        map_location='cpu')['model']
            
        m = student.backbone.load_state_dict(ckpt, strict=False)
        print('Student model is loaded by pretrained Weights with this message: {}'.format(m))

    student, teacher = student.cuda(), teacher.cuda()
    
    student = nn.DataParallel(student)
    teacher = nn.DataParallel(teacher)
    
    m = teacher.load_state_dict(student.state_dict(), strict=False)
    print(m)
    
    for p in teacher.parameters():
        p.requires_grad = False

    print('student and teacher models are ready!')
    return student, teacher





