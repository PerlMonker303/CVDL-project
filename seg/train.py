import pickle

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
    plot_roc_curve_custom, append_tensor_scalar, compute_ratio

# Hyper-parameters
torch.manual_seed(0)  # DO NOT CHANGE
np.random.seed(0)  # DO NOT CHANGE
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # do not modify
BATCH_SIZE = 1  # modify based on your system
NUM_EPOCHS = 150
NUM_WORKERS = 1  # modify based on your system
IMAGE_HEIGHT = 512  # originally 1920
IMAGE_WIDTH = 512  # originally 1080
PIN_MEMORY = True  # DO NOT CHANGE
LOAD_MODEL = False  # modify if you'd like to start from an existing model
LOAD_PATH = "../runs/quadro/unet/att12/att12_16.pth.tar"  #  "saved_models/remove_0.pth.tar"   # the path to the existing model
SAVE_PATH = "saved_models/remove.pth.tar"  # where to save the model
IS_INFERENCE = False  # true if you want to see the results for a specific model from LOAD_PATH; LOAD_MODEL must be true
# data directories
IMAGE_DIR = "./data_rail/overfit_selected/train_images/"  # "../../croppedDataset/images"  # "../../dataset/jpgs/rs19_val/"
MASK_DIR = "./data_rail/overfit_selected/train_masks/"# "../../croppedDataset/groundTruth"  # "../../dataset/uint8/rs19_val/"
IMAGE_SAVE_DIR = "saved_images/"  # where to save the validation results
NO_CLASSES = 1  # number of output classes besides the background - keep it 1 for both problems
BETA = 37.7584
THRESHOLD = 0.5

PLOT_CONFUSION_MATRICES = False
PLOT_ROC = True
PLOT_GRAPHS = True
SAVE_GRAPHS = True  # if this is True, then plots will be saved to GRAPHS_SAVE_DIR, if False then displayed
GRAPHS_SAVE_DIR = "training_results/rails/"
SMOOTH = 1

# ignored if LOAD_MODEL = True
TOTAL_IMAGES = 4
TRAIN_IMAGES = 3
VAL_IMAGES = 1
SHUFFLE_IMAGES = False
USE_PREDEFINED_DISTRIBUTION = False
INDICES_DISTRIBUTION_PATH = "distro.pickle"


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
def train_fn(train_loader, model, optimizer, loss_fn, scaler):
    # declare a loop variable for displaying training details to the console
    loop = tqdm(train_loader)
    loss = 0
    train_loss = 0

    for batch_idx, (data, targets) in enumerate(loop):
        # convert the tensors to the current device (cpu/gpu)
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        # forward pass
        predictions = model(data)  # compute predictions
        predictions = torch.reshape(predictions, targets.size())  # reshape predictions
        loss = loss_fn(predictions, targets.float())  # compute the loss
        train_loss += loss.item()

        # backward pass
        optimizer.zero_grad()  # reset the gradients
        scaler.scale(loss).backward()  # backward pass
        scaler.step(optimizer)  # perform the backward steps using the optimizer
        scaler.update()  # update the scaler

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

    return torch.tensor(train_loss / len(train_loader))


def get_validation_loss(val_loader, model, loss_fn):
    torch.cuda.empty_cache()
    val_loss = 0

    model.eval()
    # validation loss
    for data, targets in val_loader:
        # convert the tensors to the current device (cpu/gpu)
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        # forward pass
        predictions = model(data)  # compute predictions
        predictions = torch.reshape(predictions, targets.size())  # reshape predictions
        val_loss += loss_fn(predictions, targets.float()).item()  # compute the loss

    model.train()

    return torch.tensor(val_loss / len(val_loader))


def print_metrics(iou, iou_rails, iou_bg, dice):
    print(f"IoU: {iou:.2f}%")
    print(f"IoU Rails: {iou_rails:.2f}%")
    print(f"IoU Background: {iou_bg:.2f}%")
    print(f"Dice score: {dice:.2f}%")


def main():
    global TOTAL_IMAGES, VAL_IMAGES, TRAIN_IMAGES
    # Declaring the model - choose the one you need
    model = UNet(in_channels=1, out_channels=NO_CLASSES).to(DEVICE)
    # model = ResUNet(in_channels=1, out_channels=NO_CLASSES).to(DEVICE)
    # model = ResUNetPlusPlus(in_channels=1, out_channels=NO_CLASSES).to(DEVICE)

    # Declaration of the loss function
    loss_fn = weighted_bce  # weighted BCE
    # loss_fn = dice_loss
    # loss_fn = DiceLoss()

    # Declaration of the optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Declaration of the scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)

    # Setup for temporary variables
    best_dice = 0  # best dice score obtained so far - for the validation set
    epoch_offset = 0  # offset of the nr of epochs (> 0 in case LOAD_MODEL = true)
    # Arrays for keeping track of the history
    loss_history_train = torch.tensor([])
    loss_history_val = torch.tensor([])
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
        epoch_offset, loss_history_train, loss_history_val, iou_history_train, iou_history_val, iou_rails_history_train, iou_rails_history_val, \
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

    # use a predefined data distribution
    if USE_PREDEFINED_DISTRIBUTION:
        filehandler = open(INDICES_DISTRIBUTION_PATH, 'rb')
        object = pickle.load(filehandler)
        indices = object["indices"]
        config = object["config"]
        print(f"[INFO]Using predefined config: {config}")
        print(indices)

    if LOAD_MODEL and IS_INFERENCE:  # INFERENCE
        # Compute metrics related to the loaded model (validation set only)
        iou, iou_rails, iou_bg, dice = compute_metrics(val_loader, model, threshold=THRESHOLD)
        print_metrics(iou, iou_rails, iou_bg, dice)
        roc_auc = plot_roc_curve_custom(val_loader, model, -1, save_graphs=SAVE_GRAPHS, save_dir=GRAPHS_SAVE_DIR)
        print(f"ROC-AUC: {(roc_auc * 100):.2f}%")
        save_predictions_as_imgs(-1, BATCH_SIZE, val_loader, model, 0, 0, 0,
                                 threshold=THRESHOLD, folder=IMAGE_SAVE_DIR, rows=2)
        return

    # Declare a gradient scaler
    scaler = torch.cuda.amp.GradScaler()

    # Start of the training process
    for epoch in range(epoch_offset, epoch_offset + NUM_EPOCHS):
        print(f'[Epoch: {epoch} started ...]')
        # compute the loss
        loss_train = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        loss_val = get_validation_loss(val_loader, model, loss_fn)
        print(f'[Training loss: {loss_train:.2f}]')
        print(f'[Validation loss: {loss_val:.2f}]')
        # compute loss difference
        if len(loss_history_train) > 0:
            loss_dif = loss_train / loss_history_train[-1] - 1
            print(f'[Loss difference: {loss_dif:.4f}%]')
        # append the loss
        loss_history_train = append_tensor_scalar(loss_history_train, loss_train)
        loss_history_val = append_tensor_scalar(loss_history_val, loss_val)

        # compute metrics and save them
        print(f'[Training metrics ...]')
        iou_train, iou_rails_train, iou_bg_train, dice_train = compute_metrics(train_loader, model, threshold=THRESHOLD)
        print_metrics(iou_train, iou_rails_train, iou_bg_train, dice_train)
        iou_history_train = append_tensor_scalar(iou_history_train, iou_train)
        iou_rails_history_train = append_tensor_scalar(iou_rails_history_train, iou_rails_train)
        iou_bg_history_train = append_tensor_scalar(iou_bg_history_train, iou_bg_train)
        dice_history_train = append_tensor_scalar(dice_history_train, dice_train)
        print(f'[... done]')
        print(f'[Validation metrics ...]')
        iou_val, iou_rails_val, iou_bg_val, dice_val = compute_metrics(val_loader, model, threshold=THRESHOLD)
        print_metrics(iou_val, iou_rails_val, iou_bg_val, dice_val)
        iou_history_val = append_tensor_scalar(iou_history_val, iou_val)
        iou_rails_history_val = append_tensor_scalar(iou_rails_history_val, iou_rails_val)
        iou_bg_history_val = append_tensor_scalar(iou_bg_history_val, iou_bg_val)
        dice_history_val = append_tensor_scalar(dice_history_val, dice_val)
        print(f'[... done]')

        if PLOT_ROC:
            roc_auc = plot_roc_curve_custom(val_loader, model, epoch, save_graphs=SAVE_GRAPHS, save_dir=GRAPHS_SAVE_DIR)
            print(f"ROC-AUC: {(roc_auc*100):.2f}%")
        if PLOT_CONFUSION_MATRICES:
            plot_confusion_matrices_different_thresholds(val_loader, model, thresholds, epoch)
        compute_metrics_different_thresholds(val_loader, model, thresholds, epoch, save_graphs=SAVE_GRAPHS, save_dir=GRAPHS_SAVE_DIR)

        # save the model every 5 epochs
        if epoch % 5 == 0:
            print("[Saving checkpoint ...]")
            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "loss_history_train": loss_history_train.cpu().detach(),
                "loss_history_val": loss_history_val.cpu().detach(),
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
            display_result_graph(loss_history_train, loss_history_val, dice_history_train, dice_history_val, iou_history_train, iou_history_val,
                                 iou_rails_history_train, iou_rails_history_val, save_graphs=SAVE_GRAPHS, save_dir=GRAPHS_SAVE_DIR)
            print(f'[... done]')

        # call scheduler
        scheduler.step(loss_train)

        print(f'[Epoch: {epoch} ... done]')


if __name__ == "__main__":
    main()
