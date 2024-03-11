import os
import glob
import torch
from monai.transforms import MapTransform

def data_list(abs_path_data):
    test_images = sorted(glob.glob(os.path.join(abs_path_data, "*.nii.gz")))
    test_data = [{"image": image} for image in test_images]

    return test_data


class SingleChannel(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            meta_data = d["image"].meta
            d_shape = d[key].shape
            d1 = torch.zeros(d_shape[1], d_shape[2], d_shape[3])
            d1[d[key][1, :, :, :] >= 0.5] = 2
            d1[d[key][0, :, :, :] >= 0.5] = 1
            d1[d[key][2, :, :, :] >= 0.5] = 3
            d1 = torch.unsqueeze(d1, 0)
            d[key] = d1
            d[key].meta = meta_data

        return d
