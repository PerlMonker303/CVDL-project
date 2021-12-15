import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model import UNet
from model_resunet import ResUNet
from model_resunetpp import ResUNetPlusPlus
from utils import save_checkpoint, compute_metrics, save_predictions_as_imgs, get_loaders, load_checkpoint, \
    compute_metrics_different_thresholds, plot_confusion_matrices_different_thresholds, display_result_graph, \
    plot_roc_curve_custom, append_tensor_scalar, DiceLoss


# Hyper-parameters
torch.manual_seed(0)  # DO NOT CHANGE
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # do not modify
BATCH_SIZE = 1  # modify based on your system
NUM_EPOCHS = 150
NUM_WORKERS = 1  # modify based on your system
IMAGE_HEIGHT = 512  # originally 1920
IMAGE_WIDTH = 512  # originally 1080
PIN_MEMORY = True  # DO NOT CHANGE
LOAD_MODEL = True  # modify if you'd like to start from an existing model
LOAD_PATH = "saved_models/remove_0.pth.tar"  # "../runs/quadro/unet/att2/att2_41.pth.tar"  # the path to the existing model
SAVE_PATH = "saved_models/remove.pth.tar"  # where to save the model
IS_INFERENCE = False  # true if you want to see the results for a specific model from LOAD_PATH; LOAD_MODEL must be true
# data directories
# TRAIN_IMG_DIR = "data_rail/train_images/"  # "data_cells/ACDC2017/training_2017jpg/"
# TRAIN_MASK_DIR = "data_rail/train_masks/"  # "data_cells/ACDC2017/training_2017_gt/"
# VAL_IMG_DIR = "data_rail/val_images/"  # "data_cells/ACDC2017/validation_2017jpg/"
# VAL_MASK_DIR = "data_rail/val_masks/"  # "data_cells/ACDC2017/validation_2017_gt/"
IMAGE_DIR = "../../croppedDataset/images"  # "../../dataset/jpgs/rs19_val/"
MASK_DIR = "../../croppedDataset/groundTruth"  # "../../dataset/uint8/rs19_val/"
IMAGE_SAVE_DIR = "saved_images/"  # where to save the validation results
NO_CLASSES = 1  # number of output classes besides the background - keep it 1 for both problems
BETA = 37.7584
THRESHOLD = 0.5

PLOT_CONFUSION_MATRICES = False
PLOT_ROC = False
PLOT_GRAPHS = True
SAVE_GRAPHS = True  # if this is True, then plots will be saved to GRAPHS_SAVE_DIR, if False then displayed
GRAPHS_SAVE_DIR = "training_results/rails/"
SMOOTH = 1

# ignored if LOAD_MODEL = True
TOTAL_IMAGES = 20
TRAIN_IMAGES = 10
VAL_IMAGES = 5
SHUFFLE_IMAGES = True


def weighted_bce(y_pred, y_true):
    weights = (y_true * BETA) + 1.
    bce = nn.BCELoss(reduction='none')(y_pred, y_true)
    wbce = torch.mean(bce * weights)
    return wbce


def dice_loss(y_pred, y_true):
    # dice loss - iou_bg increases
    a = 2 * torch.sum(y_pred * y_true)
    b = torch.sum(torch.pow(y_pred, 2)) + torch.sum(torch.pow(y_true, 2))
    div = a / b
    return 1 - div


#  called on each epoch
def train_fn(loader, model, optimizer, loss_fn, scaler):
    # declare a loop variable for displaying training details to the console
    loop = tqdm(loader)
    loss = 0

    for batch_idx, (data, targets) in enumerate(loop):
        # convert the tensors to the current device (cpu/gpu)
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        # forward pass
        predictions = model(data)  # compute predictions
        predictions = torch.reshape(predictions, targets.size())  # reshape predictions
        loss = loss_fn(predictions, targets.float())  # compute the loss

        # backward pass
        optimizer.zero_grad()  # reset the gradients
        scaler.scale(loss).backward()  # backward pass
        scaler.step(optimizer)  # perform the backward steps using the optimizer
        scaler.update()  # update the scaler

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

    return loss


def main():
    # Declaring the model - choose the one you need
    global TOTAL_IMAGES, VAL_IMAGES, TRAIN_IMAGES
    model = UNet(in_channels=1, out_channels=NO_CLASSES).to(DEVICE)
    # model = ResUNet(in_channels=1, out_channels=NO_CLASSES).to(DEVICE)
    # model = ResUNetPlusPlus(in_channels=1, out_channels=NO_CLASSES).to(DEVICE)

    # Note: ResUNetPlusPlus works with 152x152 (cells) or 224x224 (rails) sizes for images

    # Declaration of the loss function
    # loss_fn = weighted_bce  # weighted BCE
    # loss_fn = dice_loss
    loss_fn = DiceLoss()

    # Declaration of the optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Setup for temporary variables
    best_dice = 0  # best dice score obtained so far - for the validation set
    epoch_offset = 0  # offset of the nr of epochs (> 0 in case LOAD_MODEL = true)
    # Arrays for keeping track of the history
    loss_history = torch.tensor([])
    iou_history_train = torch.tensor([])
    iou_history_val = torch.tensor([])
    iou_rails_history_train = torch.tensor([])
    iou_rails_history_val = torch.tensor([])
    iou_bg_history_train = torch.tensor([])
    iou_bg_history_val = torch.tensor([])
    dice_history_train = torch.tensor([])
    dice_history_val = torch.tensor([])
    thresholds = np.arange(0.2, 1, 0.1)
    indices = None

    if LOAD_MODEL:  # Loading the model
        epoch_offset, loss_history, iou_history_train, iou_history_val, iou_rails_history_train, iou_rails_history_val, \
        iou_bg_history_train, iou_bg_history_val, dice_history_train, dice_history_val, indices, config \
            = load_checkpoint(torch.load(LOAD_PATH), model)
        TOTAL_IMAGES, TRAIN_IMAGES, VAL_IMAGES = config

    # Data loaders initialization
    train_loader, val_loader, indices = get_loaders(
        IMAGE_DIR,
        MASK_DIR,
        BATCH_SIZE,
        NUM_WORKERS,
        PIN_MEMORY,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        TOTAL_IMAGES,
        TRAIN_IMAGES,
        VAL_IMAGES,
        SHUFFLE_IMAGES,
        indices
    )

    if LOAD_MODEL:  # Loading the model
        # Compute metrics related to the loaded model (validation set only)
        best_iou, best_iou_rails, best_iou_bg, best_dice = compute_metrics(val_loader, model, threshold=THRESHOLD)
        if IS_INFERENCE:
            print(best_iou, best_iou_rails, best_iou_bg, best_dice)
            save_predictions_as_imgs(-1, BATCH_SIZE, val_loader, model, 0, 0, 0,
                                     threshold=THRESHOLD, folder=IMAGE_SAVE_DIR, rows=2)
            return

    # Declare a gradient scaler
    scaler = torch.cuda.amp.GradScaler()

    # Start of the training process
    for epoch in range(epoch_offset, epoch_offset + NUM_EPOCHS):
        print(f'[Epoch: {epoch} started ...]')
        # compute the loss
        loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        # compute loss difference
        if len(loss_history) > 0:
            loss_dif = loss / loss_history[-1] - 1
            print(f'[Loss difference: {loss_dif:.4f}%]')
        # append the loss
        loss_history = append_tensor_scalar(loss_history, loss)

        # compute metrics and save them
        print(f'[Training metrics ...]')
        iou_train, iou_rails_train, iou_bg_train, dice_train = compute_metrics(train_loader, model, threshold=THRESHOLD)
        print(f"IoU: {iou_train:.2f}%")
        print(f"IoU Rails: {iou_rails_train:.2f}%")
        print(f"IoU Background: {iou_bg_train:.2f}%")
        print(f"Dice score: {dice_train:.2f}%")
        iou_history_train = append_tensor_scalar(iou_history_train, iou_train)
        iou_rails_history_train = append_tensor_scalar(iou_rails_history_train, iou_rails_train)
        iou_bg_history_train = append_tensor_scalar(iou_bg_history_train, iou_bg_train)
        dice_history_train = append_tensor_scalar(dice_history_train, dice_train)
        print(f'[... done]')
        print(f'[Validation metrics ...]')
        iou_val, iou_rails_val, iou_bg_val, dice_val = compute_metrics(val_loader, model, threshold=THRESHOLD)
        print(f"IoU: {iou_val:.2f}%")
        print(f"IoU Rails: {iou_rails_val:.2f}%")
        print(f"IoU Background: {iou_bg_val:.2f}%")
        print(f"Dice score: {dice_val:.2f}%")
        iou_history_val = append_tensor_scalar(iou_history_val, iou_val)
        iou_rails_history_val = append_tensor_scalar(iou_rails_history_val, iou_rails_val)
        iou_bg_history_val = append_tensor_scalar(iou_bg_history_val, iou_bg_val)
        dice_history_val = append_tensor_scalar(dice_history_val, dice_val)
        print(f'[... done]')

        if PLOT_ROC:
            roc_auc = plot_roc_curve_custom(val_loader, model, epoch, save_graphs=SAVE_GRAPHS, save_dir=SAVE_PATH)
            print(f"ROC-AUC: {(roc_auc*100):.2f}%")
        if PLOT_CONFUSION_MATRICES:
            plot_confusion_matrices_different_thresholds(val_loader, model, thresholds, epoch)
        compute_metrics_different_thresholds(val_loader, model, thresholds, epoch, save_graphs=SAVE_GRAPHS, save_dir=SAVE_PATH)

        # save the model every 5 epochs
        if epoch % 5 == 0:
            print("[Saving checkpoint ...]")
            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "loss_history": loss_history,
                "iou_history_train": iou_history_train.cpu().detach(),
                "iou_history_val": iou_history_val.cpu().detach(),
                "iou_rails_history_train": iou_rails_history_train.cpu().detach(),
                "iou_rails_history_val": iou_rails_history_val.cpu().detach(),
                "iou_bg_history_train": iou_bg_history_train.cpu().detach(),
                "iou_bg_history_val": iou_bg_history_val.cpu().detach(),
                "dice_history_train": dice_history_train.cpu().detach(),
                "dice_history_val": dice_history_val.cpu().detach(),
                "indices": indices,
                "config": (TOTAL_IMAGES, TRAIN_IMAGES, VAL_IMAGES)
            }
            save_checkpoint(checkpoint, f"saved_models/remove_{epoch}.pth.tar")
            print(f'[... done]')


        # save predictions
        print("[Saving predictions ...]")
        save_predictions_as_imgs(epoch, BATCH_SIZE, val_loader, model, dice_val, iou_val, iou_rails_val,
                                 threshold=THRESHOLD, folder=IMAGE_SAVE_DIR, rows=8)
        print(f'[... done]')

        # display graphs
        if PLOT_GRAPHS:
            print("[Displaying graphs ...]")
            display_result_graph(loss_history, dice_history_train, dice_history_val, iou_history_train, iou_history_val,
                                 iou_rails_history_train, iou_rails_history_val, save_graphs=SAVE_GRAPHS, save_dir=GRAPHS_SAVE_DIR)
            print(f'[... done]')

        print(f'[Epoch: {epoch} ... done]')


if __name__ == "__main__":
    main()
