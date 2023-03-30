import torch

from .utils import get_params_groups, cosine_scheduler, gamma_scheduler

def build_solver(args, student):
    params_groups = get_params_groups(student, wd=args.weight_decay)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(params_groups) 
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, 
                                    lr=0, 
                                    momentum=0.9, 
                                    nesterov=True)
    else:
        raise NameError('Optimizer {} is not applicable!'.format(args.optimzier))
    
    if args.scheduler == 'gamma':
        lr_schedule = gamma_scheduler(base_value=args.lr,
                                    warmup_factor=args.warmup_factor,
                                    warmup_epochs=args.warmup_epochs,
                                    epochs=args.epochs,
                                    gamma=args.gamma,
                                    milestones=args.milestones)

    elif args.scheduler == 'cosine':
        lr_schedule = cosine_scheduler(base_value=args.lr,
                                    final_value=args.min_lr,
                                    warmup_factor=args.warmup_factor,
                                    warmup_epochs=args.warmup_epochs,
                                    epochs=args.epochs)
    
    return optimizer, lr_schedule