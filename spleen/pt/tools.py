import numpy as np
from monai.utils import first
import matplotlib.pyplot as plt
from monai.data import DataLoader, Dataset
from monai.transforms import (AsDiscrete,
                              AsDiscreted,
                              RandAffined,
                              EnsureChannelFirstd,
                              Compose,
                              CropForegroundd,
                              LoadImaged,
                              Orientationd,
                              RandCropByPosNegLabeld,
                              SaveImaged,
                              ScaleIntensityRanged,
                              Spacingd,
                              Invertd)



def get_transform_train(a_min, a_max, voxel_space, patch_size, posneg_sample):
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=a_min,
                a_max=a_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=voxel_space, mode=("bilinear", "nearest")),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                pos=1,
                neg=1,
                num_samples=posneg_sample,
                image_key="image",
                image_threshold=0,
            ),
            # user can also add other random transforms
            RandAffined(
                keys=['image', 'label'],
                mode=('bilinear', 'nearest'),
                prob=1.0,
                spatial_size=patch_size,
                rotate_range=(0, 0, np.pi/15),
                scale_range=(0.1, 0.1, 0.1)),
        ]
    )
    return train_transforms
def get_transform_val(a_min, a_max, voxel_space):
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=a_min,
                a_max=a_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=voxel_space, mode=("bilinear", "nearest")),
        ]
    )
    return val_transforms

def orig_space_transform(a_min, a_max, voxel_space):
    val_org_transforms = Compose([LoadImaged(keys=["image", "label"]),
                                  EnsureChannelFirstd(keys=["image", "label"]),
                                  Orientationd(keys=["image"], axcodes="RAS"),
                                  Spacingd(keys=["image"], pixdim=voxel_space, mode="bilinear"),
                                  ScaleIntensityRanged(keys=["image"],
                                                       a_min=a_min,
                                                       a_max=a_max,
                                                       b_min=0.0,
                                                       b_max=1.0,
                                                       clip=True),
                                  CropForegroundd(keys=["image"], source_key="image")
                                  ]
    )
    return val_org_transforms

def get_post_transform(val_org_transforms):
    post_transforms = Compose([
        Invertd(
                keys="pred",
                transform=val_org_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
                device="cpu",
            ),
        AsDiscreted(keys="pred", argmax=True, to_onehot=2),
        AsDiscreted(keys="label", to_onehot=2)
    ])

    return post_transforms

def get_test_transform(a_min, a_max, voxel_space):
    test_org_transforms = Compose(
        [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=voxel_space, mode="bilinear"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=a_min,
                a_max=a_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
        ]
    )
    return test_org_transforms

def save_test_preds(test_org_transforms, save_dir):
    post_transforms_save = Compose(
        [
            Invertd(
                keys="pred",
                transform=test_org_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            AsDiscreted(keys="pred", argmax=True, to_onehot=2),
            SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=save_dir,
                       output_postfix="seg", resample=False,
                       separate_folder=False),
        ]
    )
    return post_transforms_save


def santiy_check_data(val_files, val_transforms):
    check_ds = Dataset(data=val_files, transform=val_transforms)
    check_loader = DataLoader(check_ds, batch_size=1)
    check_data = first(check_loader)
    image, label = (check_data["image"][0][0], check_data["label"][0][0])
    print(f"image shape: {image.shape}, label shape: {label.shape}")
    # plot the slice [:, :, 80]
    plt.figure("check", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("image")
    plt.imshow(image[:, :, 80], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("label")
    plt.imshow(label[:, :, 80])
    plt.show()

    return None