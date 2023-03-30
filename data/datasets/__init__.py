import os.path as osp

from .veri import VeRi
from .vehicleid import VehicleID
from .veriwild import VeRiWild

__factory = {'VeRi': VeRi,
             'VehicleID': VehicleID,
             'VeRiWild': VeRiWild}


def init_dataset(args):
    if not args.dataset in __factory.keys():
        raise KeyError("Unknown dataset : {}".format(args.dataset))
    
    data_path = {'VeRi': osp.join(args.data_root, 'VeRi'),
        'VehicleID': osp.join(args.data_root, 'VehicleID'),
        'VeRiWild': osp.join(args.data_root, 'VeRI-Wild')}[args.dataset]
    
    return __factory[args.dataset](args, data_path)
