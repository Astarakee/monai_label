import torch
from monai.metrics import DiceMetric
from monai.data import decollate_batch
from monai.handlers.utils import from_engine
from monai.inferers import sliding_window_inference



def evaluation(abs_path_checkpoint, device, model, val_org_loader, iso_patch_size_val, sw_batch_size, post_transforms):

    model.to(device)
    model.load_state_dict(torch.load(abs_path_checkpoint))
    model.eval()
    patch_size_Val = (iso_patch_size_val, iso_patch_size_val, iso_patch_size_val)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    with torch.no_grad():
        for val_data in val_org_loader:
            val_inputs = val_data["image"].to(device)
            val_data["pred"] = sliding_window_inference(val_inputs, patch_size_Val, sw_batch_size, model)
            val_data = [post_transforms(i) for i in decollate_batch(val_data)]
            val_outputs, val_labels = from_engine(["pred", "label"])(val_data)
            # compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=val_labels)

        # aggregate the final mean dice result
        metric_org = dice_metric.aggregate().item()
        # reset the status for next validation round
        dice_metric.reset()

    print("Metric on original image spacing: ", metric_org)

    return None