import os
import glob

def get_data_list(abs_path):
    data_list = sorted(glob.glob(os.path.join(abs_path, "*.nii.gz")))
    image_list = [{"image": image} for image in data_list]
    return image_list