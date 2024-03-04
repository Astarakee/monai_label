import torch
import matplotlib.pyplot as plt
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.data import decollate_batch
from monai.transforms import AsDiscrete, Compose
from monai.inferers import sliding_window_inference


def plots_training(epoch_loss_values, val_interval, metric_values, save_path):
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.savefig(save_path)
    plt.show()
    plt.close()

    return None

def save_checkpoint(metric, model, best_metric, epoch, abs_path_checkpoint, best_metric_epoch):
    if metric > best_metric:
        best_metric = metric
        best_metric_epoch = epoch + 1
        torch.save(model.state_dict(), abs_path_checkpoint)
        print("saved new best metric model")
    else:
        None

    return best_metric_epoch
def validation(val_inputs, val_labels,patch_size_Val, sw_batch_size, model, post_label, post_pred):
    val_outputs = sliding_window_inference(val_inputs, patch_size_Val, sw_batch_size, model)
    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
    return val_outputs, val_labels

def training(max_epochs, model, val_interval, data_loader,
             device, iso_patch_size_val, sw_batch_size,
             abs_path_checkpoint, fig_save_path):


    model.to(device)
    patch_size_Val = (iso_patch_size_val, iso_patch_size_val, iso_patch_size_val)
    train_ds = data_loader['train_ds']
    train_loader = data_loader['train_loader']
    val_loader = data_loader['val_loader']
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    val_outputs, val_labels = validation(val_inputs, val_labels, patch_size_Val, sw_batch_size, model, post_label, post_pred)
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()

                metric_values.append(metric)
                best_metric_epoch = save_checkpoint(metric, model, best_metric, epoch, abs_path_checkpoint, best_metric_epoch)
                print(f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                      f"\nbest mean dice: {best_metric:.4f} "
                      f"at epoch: {best_metric_epoch}")
    print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")
    plots_training(epoch_loss_values, val_interval, metric_values, fig_save_path)

    return model