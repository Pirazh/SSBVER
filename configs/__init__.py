import argparse

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

def get_configs():
    parser = argparse.ArgumentParser(description='SSLBVER', add_help=False)
    # ----------------------------------------
    # Data Configuration
    # ----------------------------------------
    parser.add_argument('--data_root', default='', type=str, required=True)
    parser.add_argument('--dataset', default='VeRi', type=str,
                                choices=['VeRi', 'VehicleID', 'VeRiWild']) 
    # ----------------------------------------
    # Data Transformation
    # ----------------------------------------
    parser.add_argument('--pixel_mean', type=float, nargs='+', 
                                        default=(0.485, 0.456, 0.406))
    parser.add_argument('--pixel_std', type=float, nargs='+', 
                                        default=(0.229, 0.224, 0.225))
    parser.add_argument('--multi_crop', type=bool_flag, default=True)
    parser.add_argument('--local_crops_num', type=int, default=5)
    parser.add_argument('--global_crop_scale', type=float, 
                                        nargs='+', default=(0.8, 1.))
    parser.add_argument('--local_crop_scale', type=float, 
                                        nargs='+', default=(0.1, 0.4))
    parser.add_argument('--train_global_size', type=int, 
                                        nargs='+', default=(256, 256))
    parser.add_argument('--train_local_size', type=int, nargs='+', 
                                        default=(112, 112))
    parser.add_argument('--test_size', type=int, nargs='+', 
                                        default=(256, 256))
    parser.add_argument('--rea', type=bool_flag, default=True, 
                                help='Random Erasing Augmentation Technique')
    parser.add_argument('--rea_prob', type=float, default=0.5)
    parser.add_argument('--rea_values', type=float, nargs='+', 
                                        default=(0.485, 0.456, 0.406))
    parser.add_argument('--hflip', type=bool_flag, default=True, 
                                help='Horizontal Flip Augmentation')
    parser.add_argument('--hflip_prob', type=float, default=0.5)
    parser.add_argument('--brightness', type=float, default=0.2)
    parser.add_argument('--contrast', type=float, default=0.2)
    parser.add_argument('--saturation', type=float, default=0.1)
    parser.add_argument('--hue', type=float, default=0.02)
    parser.add_argument('--colorjitter_prob', type=float, default=0.1)
    parser.add_argument('--gaussian_blur_prob', type=float, default=0.1)
    parser.add_argument('--pad_size', type=int, default=5, 
                        help='Number of Pixels to be padded with the image')
    # ----------------------------------------
    # Data Loader
    # ----------------------------------------
    parser.add_argument('--num_instances', type=int, default=16, 
                    help='Number of images for each unique identity in a batch')
    parser.add_argument('--num_workers', type=int, default=16)
    # ----------------------------------------
    # Model Configuration
    # ----------------------------------------
    parser.add_argument('--model_arc', type=str, default='vit_small', 
                    choices=['vit_small', 'resnet50', 'resnet50_ibn_a', 
                    'resnet101', 'vit_base', 'swin_base', 'convnext_base'])
    parser.add_argument('--pretrained', type=bool_flag, default=False)
    parser.add_argument('--pretrained_method', type=str, 
                            default='ImageNet', choices=['ImageNet', 'DINO'])
    parser.add_argument('--last_stride', type=int, default=1, 
                            help='stride of the last layer in resnet')
    parser.add_argument('--patch_size', type=int, default=16, 
                help='Patch size of a ViT model')
    parser.add_argument('--ssl_dim', type=int, default=65536)
    parser.add_argument('--norm_last_layer', default=True, type=bool_flag,
    help="""Whether or not to weight normalize the last layer of ssl head""")
    parser.add_argument('--use_bn_in_head', default=False, type=bool_flag,
    help="Whether to use batch normalizations in projection head")
    parser.add_argument('--label_smoothing', type=bool_flag, default=False,
    help='Use of label smoothing for the cross entropy loss of student model')
    parser.add_argument('--label_smoothing_eps', type=float, default=0.2,
    help='Label smoothing Value')
    parser.add_argument('--neck', type=str, default='bnneck', 
                choices=['no', 'bnneck'],
                help='the layer type that isolate triplet loss from id loss')
    # ----------------------------------------
    # Optimizer Configuration
    # ----------------------------------------
    parser.add_argument('--optimizer', default='adam', type=str, 
                choices=['adam', 'sgd', 'adamw'])
    parser.add_argument('--warmup_teacher_temp', default=0.0005, type=float,
                help="""Initial value for the teacher temperature""")
    parser.add_argument('--teacher_temp', default=0.001, type=float, 
                help="""Final value after linear warmup.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=10, type=int,
                help='Number of warmup epochs for the teacher temperature')
    parser.add_argument('--student_temp', default=0.1, type=float, 
                help="""student temperature""")
    parser.add_argument('--weight_decay', type=float, default=0.0001, 
                help="""value of the weight decay.""")
    parser.add_argument('--clip_grad', type=float, default=0.0, 
                help="""Maximum gradient norm. 0 for disabling.""")
    parser.add_argument("--lr", default=0.001, type=float, 
                help="""Learning rate at the end of linear warmup.""")
    parser.add_argument('--scheduler', type=str, default='gamma', 
                choices=['cosine', 'gamma'])
    parser.add_argument("--warmup_epochs", default=10, type=int,
                help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument("--warmup_factor", default=0.01, type=float,
                help="Factor of base LR in the begining of training")
    parser.add_argument("--gamma", default=0.1, type=float,
                help="Gamma Decay factor for learning rate")
    parser.add_argument('--milestones', type=int, nargs='+', 
                default=(40, 70, 100))
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""lr at the
        end of optimization with cosine LR schedule.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, 
                help="stochastic depth rate")
    # ----------------------------------------
    # Training Configuration
    # ----------------------------------------
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=120, type=int, 
                help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, 
                help="""Number of epochs to keep the output layer fixed.""")
    parser.add_argument('--momentum_teacher', default=0.9995, type=float, 
                help='Base EMA parameter for teacher update')
    parser.add_argument('--output_dir', default=".", type=str, 
                help='Path to save logs and checkpoints.')
    parser.add_argument('--save_ckpt_freq', default=20, type=int, 
                help='Save checkpoint every [] epochs.')
    parser.add_argument('--eval_freq', default=10, type=int, 
                help='evaluate reid performance every x epochs')
    parser.add_argument('--log_freq', default=20, type=int, 
                help='show training log every [] iteration.')
    parser.add_argument('--cmpt_loss_lambda', type=float, default=1.0,
                help='weight of Compactness objective in training')
    parser.add_argument('--ssl_loss_lambda', type=float, default=1.0,
                help='weight of ssl objective in training')
    parser.add_argument('-ssl_loss', type=bool_flag, default=False, 
                help='whether to use ssl loss')
    parser.add_argument('--id_loss_lambda', type=float, default=1.0, 
                help='weight of ID objective in training')
    parser.add_argument('--id_loss', type=bool_flag, default=False, 
                help='whether to use id loss')
    parser.add_argument('--triplet_loss_lambda', type=float, default=1.0,
                help='weight of triplet objective in training')
    parser.add_argument('--triplet_loss', type=bool_flag, default=False, 
                help='whether to use triplet loss')
    parser.add_argument('--use_margin', type=bool_flag, default=False, 
                help='whether to use softmargin or ranking loss')
    parser.add_argument('--triplet_loss_margin', type=float, default=0.3,
                help='the margin used in hard triplet loss')
    # ----------------------------------------
    # Testing Configuration
    # ----------------------------------------
    parser.add_argument('--plot_dist', type=bool_flag, default=False)
    parser.add_argument('--test_model', type=str, default='student', 
                choices=['student', 'teacher'])
    parser.add_argument('--cython_eval', type=bool_flag, default=False)
    parser.add_argument('--test_ckpt', type=str, default='')
    parser.add_argument('--test_hflip', type=bool_flag, default=False, 
                help='Horizontal Flip Augmentation')
    parser.add_argument('--re_rank', type=bool_flag, default=False)
    parser.add_argument('--k1', type=int, default=7)
    parser.add_argument('--k2', type=int, default=2)
    parser.add_argument('--lambda_value', type=float, default=0.85)
    parser.add_argument('--split', type=str, 
                choices=['small', 'medium', 'large'], 
                help='The test split to perform evaluation', default='small')
    parser.add_argument('--neck_feat', type=str, choices=['before', 'after'], 
                default='after')
    parser.add_argument('--image_path', default=None, type=str, 
                help="Path of the image to load.")
    parser.add_argument('--output_attn_dir', default='', 
                help='Path where to save visualizations.', type=str)
    parser.add_argument("--threshold", type=float, default=None, 
                help="""visulization threshold.""")
    parser.add_argument('--k_salient', default=20, type=int, 
                help='K salient filters for visualization')
    parser.add_argument('--boost_factor', default=2, type=float, 
                help='amount to boost activations')
    parser.add_argument('--query_path', default='', type=str)
    parser.add_argument('--gallery_path', default='', type=str)
    # ----------------------------------------
    # Other Configuration
    # ----------------------------------------
    parser.add_argument('--device_ids', default='0,1', type=str)
    parser.add_argument('--is_train', default=False, type=bool_flag)
    parser.add_argument('--seed', default=0, type=int, 
                help='Fix the random number generator seed')
    return parser

