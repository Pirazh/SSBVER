from torch.utils.data import DataLoader

from .datasets import init_dataset, torch_dataset
from .transforms import build_trasform
from .sampler.triplet import TripletSampler

def build_data(args):
    train_transforms = build_trasform(args, is_train=True)
    val_transforms = build_trasform(args, is_train=False)
    dataset = init_dataset(args)
    train_set = torch_dataset.ReIDDataset(dataset, train_transforms, is_train=True)
    train_sampler = TripletSampler(dataset.trainset,
                                    args.batch_size,
                                    args.num_instances)
    train_loader = DataLoader(dataset=train_set,
                                sampler=train_sampler,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)
    val_set = torch_dataset.ReIDDataset(dataset, val_transforms, is_train=False)
    val_loader = DataLoader(dataset=val_set,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False)
    num_train_classes = dataset.num_vids 

    return train_loader,val_loader, num_train_classes
