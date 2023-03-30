import copy
import random
from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import Sampler


class TripletSampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.num_vids_per_batch = batch_size // self.num_instances
        self.batch_size = batch_size

        self.vid_index = defaultdict(list)

        for index, info in enumerate(data_source):
            vid = info[1]
            self.vid_index[vid].append(index)

        self.vids = list(self.vid_index.keys())

        self.length = 0
        for vid in self.vids:
            idxs = self.vid_index[vid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances
    
    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for vid in self.vids:
            idxs = copy.deepcopy(self.vid_index[vid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[vid].append(batch_idxs)
                    batch_idxs = []

        avai_vids = copy.deepcopy(self.vids)
        final_idxs = []

        while len(avai_vids) >= self.num_vids_per_batch:
            selected_vids = random.sample(avai_vids, self.num_vids_per_batch)
            for vid in selected_vids:
                batch_idxs = batch_idxs_dict[vid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[vid]) == 0:
                    avai_vids.remove(vid)
                    
        self.length = len(final_idxs)
        return iter(final_idxs)
    
    def __len__(self):
        return self.length