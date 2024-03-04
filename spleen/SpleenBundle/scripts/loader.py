import os
import glob
from .tools import get_transform_train, get_transform_val
from monai.data import CacheDataset, DataLoader, Dataset


def get_files(data_dir):
    train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
    # train_files, val_files = data_dicts[:-12], data_dicts[-12:]
    train_files, val_files = data_dicts[:10], data_dicts[-5:]
    train_val_files = {"train_files":train_files, "val_files": val_files}
    return train_val_files

def get_test_images(test_dir):
    test_images = sorted(glob.glob(os.path.join(test_dir, "*.nii.gz")))
    test_data = [{"image": image} for image in test_images]
    return test_data

# def train_val_loader(train_val_files):
#
#     train_files = train_val_files['train_files']
#     val_files = train_val_files['val_files']
#     train_transforms = get_transform_train(-57, 164, (1.5, 1.5, 2.0), (96, 96, 96), 1)
#     val_transforms = get_transform_val(-57, 164, (1.5, 1.5, 2.0))
#     train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
#     # train_ds = Dataset(data=train_files, transform=train_transforms)
#     val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
#     # val_ds = Dataset(data=val_files, transform=val_transforms)
#     train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)
#     val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)
#     data_loader = {"train_ds":train_ds, "train_loader":train_loader, "val_loader":val_loader}
#     return data_loader

def train_val_loader(train_val_files, a_min, a_max, iso_patch_size_tr, voxel_space1, voxel_space2, voxel_space3, posneg_sample, n_worker, batch_tr, batch_val):
    patch_size = (iso_patch_size_tr, iso_patch_size_tr, iso_patch_size_tr)
    voxel_space = (voxel_space1, voxel_space2, voxel_space3)
    train_files = train_val_files['train_files']
    val_files = train_val_files['val_files']
    train_transforms = get_transform_train(a_min, a_max, voxel_space, patch_size, posneg_sample)
    val_transforms = get_transform_val(a_min, a_max, voxel_space=(1.5, 1.5, 2.0))
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=n_worker)
    # train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=n_worker)
    # val_ds = Dataset(data=val_files, transform=val_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_tr, shuffle=True, num_workers=n_worker)
    val_loader = DataLoader(val_ds, batch_size=batch_val, num_workers=n_worker)
    data_loader = {"train_ds":train_ds, "train_loader":train_loader, "val_loader":val_loader}
    return data_loader