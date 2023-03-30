from PIL import Image
from torchvision import transforms as TF
                                    
from .transform import RandomErasing

def build_trasform(args, is_train=True):
    if is_train:
        flip_and_color_jitter = TF.Compose([
            TF.RandomHorizontalFlip(p=args.hflip_prob),
            TF.RandomApply([TF.ColorJitter(brightness=args.brightness, 
                                        contrast=args.contrast, 
                                        saturation=args.saturation, 
                                        hue=args.hue)], 
                                        p=args.colorjitter_prob)])
        normalize = TF.Compose([TF.ToTensor(),
                                TF.Normalize(args.pixel_mean, args.pixel_std)])
        
        global_tf1 = TF.Compose([TF.Resize(args.train_global_size,
                                            interpolation=Image.BICUBIC),
                                TF.RandomHorizontalFlip(p=args.hflip_prob),
                                TF.Pad(args.pad_size),
                                TF.RandomCrop(args.train_global_size),
                                normalize,
                                RandomErasing(args.rea_prob, 
                                                mean=args.pixel_mean)])
        
        global_tf2 = TF.Compose([TF.RandomResizedCrop(args.train_global_size[0],
                                                    scale=args.global_crop_scale,
                                                    interpolation=Image.BICUBIC),
                                flip_and_color_jitter,
                                normalize])
        
        local_tf = TF.Compose([TF.RandomResizedCrop(args.train_local_size[0],
                                                    scale=args.local_crop_scale,
                                                    interpolation=Image.BICUBIC),
                                flip_and_color_jitter,
                                normalize])
        class Transform(object):
            def __init__(self):
                self.g1 = global_tf1
                self.g2 = global_tf2
                self.l = local_tf
                self.num_local_crops = args.local_crops_num
            
            def __call__(self, image):
                crops = []
                crops.append(self.g1(image))
                crops.append(self.g2(image))
                for _ in range(self.num_local_crops):
                    crops.append(self.l(image))
                return crops
        tf = Transform()
        return tf
    
    else:
        return TF.Compose([TF.Resize(args.test_size, interpolation=Image.BICUBIC),
                            TF.ToTensor(),
                            TF.Normalize(args.pixel_mean, args.pixel_std)])
