import torch
from torch import nn

class MultiCropWrapper(nn.Module):
    
    def __init__(self, backbone, ssl_head, reid_head=None, is_student=True):
        super(MultiCropWrapper, self).__init__()
        self.backbone = backbone
        self.ssl_head = ssl_head
        self.reid_head = reid_head
        self.is_student = is_student

    def forward(self, x, vids=None):
        
        if self.training:
            # convert to list
            if not isinstance(x, list):
                x = [x]
            idx_crops = torch.cumsum(torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1], 0)
            num_crops = idx_crops[-1]
            start_idx, output = 0, torch.empty(0).to(x[0].device)
            for end_idx in idx_crops:
                _out = self.backbone(torch.cat(x[start_idx: end_idx]))
                # accumulate outputs
                output = torch.cat((output, _out))
                start_idx = end_idx
            # Run the head forward on the concatenated features.
            
            if self.is_student:
                if self.ssl_head is not None:
                    return self.ssl_head(output), \
                        self.reid_head(torch.cat(
                                [output.chunk(num_crops)[0],
                                    output.chunk(num_crops)[1]])), vids.repeat(2)
                        # self.reid_head(output.chunk(num_crops)[0])
                else:
                    return output, \
                        self.reid_head(output.chunk(num_crops)[0])
            else:
                if self.ssl_head is not None:
                    return self.ssl_head(output)
                else:
                    return output
        else:
            return self.reid_head(self.backbone(x))