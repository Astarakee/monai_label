import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference

def pred_test(model, abs_path_checkpoint, test_org_loader, iso_patch_size_val, sw_batch_size, device, post_transforms_save):

    patch_size_Val = (iso_patch_size_val, iso_patch_size_val, iso_patch_size_val)
    model.to(device)
    model.load_state_dict(torch.load(abs_path_checkpoint))
    model.eval()
    with torch.no_grad():
        for test_data in test_org_loader:
            test_inputs = test_data["image"].to(device)
            sw_batch_size = sw_batch_size
            test_data["pred"] = sliding_window_inference(test_inputs, patch_size_Val, sw_batch_size, model)
            test_data = [post_transforms_save(i) for i in decollate_batch(test_data)]

    return None