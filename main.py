import argparse
import os
import warnings

from configs import get_configs
from tools import log

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids

    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

    from tools import utils
    from data import build_data
    from models import build_models
    from layers import build_loss_fn
    from solver import build_solver
    from engine.trainer import train

    utils.fix_random_seeds(seed=args.seed)
    utils.create_folder(args)
    utils.save_configs(args)

    logger = log.setup_logger(args)
    if args.is_train:
        logger.info("Using {} GPUs".format(len(args.device_ids.split(','))))
        logger.info('Training Configurations:')
        for k, v in args.__dict__.items():
            logger.info('{}: {}'.format(k, v))
        
    train_loader, val_loader, num_train_classes = build_data(args)
    student, teacher = build_models(args, num_classes=num_train_classes)

    if args.is_train:
        loss_fn = build_loss_fn(args, num_classes=num_train_classes)
        optimizer, lr_scheduler = build_solver(args, student)

        train(args=args,
                train_loader=train_loader,
                val_loader=val_loader,
                student=student,
                teacher=teacher,
                loss_fn=loss_fn,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                logger=logger)
    
    else:
        import torch
        from engine.evaluator import do_eval
        ckpt = torch.load(args.test_ckpt)
        student.load_state_dict(ckpt[args.test_model], strict=False)
        print('{} model is loaded from {}'.format(args.test_model, args.test_ckpt))
        do_eval(args=args,
                val_loader=val_loader,
                model=student)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(description='SSLBVER', 
                                        parents=[get_configs()])
    main(parser.parse_args())



