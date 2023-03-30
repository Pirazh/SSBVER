import os
import os.path as osp

from .base import BaseImageDataset


class VeRi(BaseImageDataset):
    def __init__(self, args, data_path, verbose=True):
        super(VeRi).__init__()
        train_path = osp.join(data_path, 'image_train')
        gallery_path = osp.join(data_path, 'image_test')
        query_path = osp.join(data_path, 'image_query')

        self.trainset = self._process_dir(train_path, relabel=True)
        self.galleryset = self._process_dir(gallery_path, relabel=False)
        self.queryset = self._process_dir(query_path, relabel=False)

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

    def _process_dir(self, dir_path, relabel=False):
        data = []
        vids = set()
        for item in os.listdir(dir_path):
            if item.endswith('.jpg'):
                vid, camid, _, _ = item.split('_')
                im_path = osp.join(dir_path, item)
                data.append((im_path, int(vid), int(camid[1:])))
                vids.add(int(vid))
              
        vid2label = {vid: label for label, vid in enumerate(vids)}
        dataset = []
        for item in data:
            im_path, vid, camid = item
            if relabel:
                vid = vid2label[vid]
            dataset.append((im_path, vid, camid))
        return dataset