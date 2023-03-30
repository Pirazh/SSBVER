import os
import os.path as osp

from .base import BaseImageDataset

class VeRiWild(BaseImageDataset):
    def __init__(self, args, data_path, verbose=True):
        super(VeRiWild).__init__()
        split_mapper = {'small':'3000', 'medium':'5000', 'large': '10000'}
        split = split_mapper[args.split]
        self.data_path = data_path
        train_file = osp.join(data_path, 'train_test_split/train_list.txt')
        gallery_file = osp.join(data_path, 'train_test_split/test_{}.txt'.format(split))
        query_file = osp.join(data_path, 'train_test_split/test_{}_query.txt'.format(split))

        self.cam_dict = {item.split(';')[0]: int(item.split(';')[1]) for item in 
            open(osp.join(data_path, 'train_test_split', 'vehicle_info.txt'), 'r').readlines()[1:]}

        
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
            vid = item.split('/')[0]
            camid = self.cam_dict[item.strip()]
            im_path = osp.join(self.data_path, 'images', item.strip() + '.jpg')
            data.append((im_path, int(vid), int(camid)))
            vids.add(int(vid))
        
        vid2label = {vid: label for label, vid in enumerate(vids)}
        dataset = []
        for item in data:
            im_path, vid, camid = item
            if relabel:
                vid = vid2label[vid]
            dataset.append((im_path, vid, camid))
        
        return dataset