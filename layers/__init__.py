import torch.nn.functional as F
from torch import nn

from .loss import CrossEntropyLabelSmooth, TripletLoss, SSLLoss, CMPTLoss


def build_loss_fn(args, num_classes=None):
    if args.label_smoothing:
        cls_loss = CrossEntropyLabelSmooth(num_classes=num_classes,
                                            epsilon=args.label_smoothing_eps)
    
    else:
        cls_loss = nn.CrossEntropyLoss()

    triplet_loss = TripletLoss(use_margin=args.use_margin,
                                margin=args.triplet_loss_margin)
    
    if args.ssl_loss_lambda > 0:
        ssl_loss = SSLLoss(
                    out_dim=args.ssl_dim,
                    ncrops=args.local_crops_num + 2,
                    warmup_teacher_temp=args.warmup_teacher_temp,
                    teacher_temp=args.teacher_temp,
                    warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
                    student_temp=args.student_temp,
                    nepochs=args.epochs)
    else:
        ssl_loss = None
    
    if args.cmpt_loss_lambda:
        cmpt_loss = CMPTLoss(ncrops=args.local_crops_num + 2)

    def loss_fn(cls_score, feat, target, student_out, teacher_out, epoch):
        id_loss = args.id_loss_lambda * cls_loss(cls_score, target)
        trip_loss = args.triplet_loss_lambda * triplet_loss(feat, target)
        
        if args.ssl_loss_lambda > 0:
            ssl_loss_ssl = args.ssl_loss_lambda * ssl_loss(student_out, teacher_out, epoch)
        else:
            ssl_loss_ssl = None
        
        if args.cmpt_loss_lambda > 0:
            ssl_loss_cmpt = args.cmpt_loss_lambda * cmpt_loss(student_out, teacher_out)
        else:
            ssl_loss_cmpt = None
        
        return id_loss, trip_loss, ssl_loss_ssl, ssl_loss_cmpt, ssl_loss
    
    return loss_fn