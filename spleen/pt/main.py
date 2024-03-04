import os
import glob
import torch
from network import UNET
from trainer import training
from evaluate import evaluation
from inference import pred_test
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism
from monai.data import CacheDataset, DataLoader, Dataset
from tools import get_transform_train, get_transform_val, orig_space_transform
from tools import get_post_transform, save_test_preds, get_test_transform, santiy_check_data



rand_seed = 4321
set_determinism(seed=rand_seed)

# data_dir = '/mnt/workspace/projects/16_MonaiLabel/datasets/Task09_Spleen'
data_dir = "/mnt/workspace/projects/6_Monai/3_bundle_practice/0_exp1/3_nodule/pt_lightning/Task06_Lung"
root_dir = './'
save_dir = './outputs'

train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
train_files, val_files = data_dicts[:-12], data_dicts[-12:]

# a_min = -57
# a_max = 164
a_min = -1000
a_max = 500
voxel_space = (1.5, 1.5, 2.0)
patch_size_tr = (96, 96, 96)
posneg_sample = 1
batch_tr = 2
batch_val = 1
n_worker = 4
max_epochs = 1000
val_interval = 2
sw_batch_size = 4
patch_size_Val = (160, 160, 160)
device = torch.device("cuda:0")
abs_path_checkpoint = './best_metric_model_lung.pt'
abs_path_checkpoint_ts = './best_metric_model_lung.ts'
fig_save_path = './plots.png'

train_transforms = get_transform_train(a_min, a_max, voxel_space, patch_size_tr, posneg_sample)
val_transforms = get_transform_val(a_min, a_max, voxel_space)

santiy_check_data(val_files, val_transforms)

train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=n_worker)
# train_ds = Dataset(data=train_files, transform=train_transforms)
val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=n_worker)
# val_ds = Dataset(data=val_files, transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=batch_tr, shuffle=True, num_workers=n_worker)
val_loader = DataLoader(val_ds, batch_size=batch_val, num_workers=n_worker)

# standard PyTorch program style: create UNet, DiceLoss and Adam optimizer

model = UNET()
model.to(device)
loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)
dice_metric = DiceMetric(include_background=False, reduction="mean")

model = training(max_epochs, model, train_loader, optimizer, loss_function,
          val_interval, val_loader, dice_metric,
          train_ds, device, patch_size_Val, sw_batch_size,
          abs_path_checkpoint, abs_path_checkpoint_ts, fig_save_path)

# abs_path_checkpoint = './Well_trained/best_metric_model.pth'

val_org_transforms = orig_space_transform(a_min, a_max, voxel_space)
val_org_ds = Dataset(data=val_files, transform=val_org_transforms)
val_org_loader = DataLoader(val_org_ds, batch_size=batch_val, num_workers=n_worker)
post_transforms = get_post_transform(val_org_transforms)
evaluation(abs_path_checkpoint, device, model, val_org_loader, patch_size_Val, sw_batch_size, post_transforms, dice_metric)

## inference
test_images = sorted(glob.glob(os.path.join(data_dir, "imagesTs", "*.nii.gz")))
test_data = [{"image": image} for image in test_images]
test_org_transforms = get_test_transform(a_min, a_max, voxel_space)
test_org_ds = Dataset(data=test_data, transform=test_org_transforms)
test_org_loader = DataLoader(test_org_ds, batch_size=batch_val, num_workers=n_worker)
post_transforms_save = save_test_preds(test_org_transforms, save_dir)
pred_test(model, abs_path_checkpoint, test_org_loader, patch_size_Val, sw_batch_size, device, post_transforms_save)


