import os
import os.path as osp

from .base import BaseImageDataset

class VehicleID(BaseImageDataset):
    def __init__(self, args, data_path, verbose=True):
        super(VehicleID).__init__()
        split_mapper = {'small':'800', 'medium':'1600', 'large': '2400'}
        split = split_mapper[args.split]
        self.data_path = data_path
        train_file = osp.join(data_path, 'Train.txt')
        gallery_file = osp.join(data_path, 'Gallery_{}.txt'.format(split))
        query_file = osp.join(data_path, 'Query_{}.txt'.format(split))

        self.trainset = self._process_dir(train_file, relabel=True)
        self.galleryset = self._process_dir(gallery_file, relabel=False)
        self.queryset = self._process_dir(query_file, relabel=False)

        self.num_vids, self.num_imgs, self.num_cams = \
                                        self.get_imagedata_info(self.trainset)
        self.num_vids_q, self.num_imgs_q, self.num_cams_q = \
                                        self.get_imagedata_info(self.queryset)
        self.num_vids_g, self.num_imgs_g, self.num_cams_g = \
                                        self.get_imagedata_info(self.galleryset)
        
        if verbose:
            self.print_dataset_statistics(args, self.trainset, 
                                                self.queryset, 
                                                self.galleryset)

    
    def _process_dir(self, label_fl, relabel=False):
        items = open(label_fl, 'r').readlines()
        data = []
        vids = set()
        for item in items:
            vid = item.split(' ')[0].split('/')[-2]
            im_path = osp.join(self.data_path, item.split(' ')[0])
            data.append((im_path, int(vid), -1))
            vids.add(int(vid))
        
        vid2label = {vid: label for label, vid in enumerate(vids)}
        dataset = []
        for item in data:
            im_path, vid, camid = item
            if relabel:
                vid = vid2label[vid]
            dataset.append((im_path, vid, camid))
            
        return dataset