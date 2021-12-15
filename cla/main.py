import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from dataset import CustomDataset
from models.vgg import VGG_net
from models.resnet import ResNet50
from utils import format_alpha, append_tensor_scalar, compute_metrics, display_result_graph, load_checkpoint, \
    save_checkpoint, display_metrics

# Hyper-parameters
torch.manual_seed(0)  # DO NOT CHANGE
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # do not modify
BATCH_SIZE = 8  # modify based on your system
NUM_EPOCHS = 150
NUM_WORKERS = 1  # modify based on your system
PIN_MEMORY = True  # DO NOT CHANGE
LOAD_MODEL = False  # modify if you'd like to start from an existing model
LOAD_PATH = "saved_models/vgg/test_remove_2.pth.tar"  # the path to the existing model
SAVE_PATH = "saved_models/vgg/"  # where to save the model
IS_INFERENCE = False  # true if you want to see the results for a specific model from LOAD_PATH; LOAD_MODEL must be true
# data directories
BASE_DATA_DIR = './data/'
NO_CLASSES = 1  # number of output classes besides the background - keep it 1 for both problems


CLASSES = ['switch-left', 'switch-right']
ALPHA = 1.45
USE_MASKS = False
PLOT_GRAPHS = True
SAVE_GRAPHS = True  # if this is True, then plots will be saved to GRAPHS_SAVE_DIR, if False then displayed
GRAPHS_SAVE_DIR = f'training_results/a_{format_alpha(ALPHA)}/'
LABELS_PATH = './data/labels_mtn.txt'

TOTAL_IMAGES = 20
TRAIN_IMAGES = 10
VAL_IMAGES = 5


def get_loaders(
    image_dir,
    labels_path,
    alpha,
    batch_size,
    num_workers=4,
    pin_memory=True,
    total_images=20,
    train_images=10,
    val_images=5,
    shuffle_images=False
):
    if train_images + val_images > total_images:
        print('[ERROR: total_images must be >= than train_images + val_images]')
        return None
    # generate indices
    indices = np.arange(total_images)
    if shuffle_images:
        np.random.shuffle(indices)

    train_ds = CustomDataset(
        image_dir=image_dir,
        alpha=alpha,
        labels_path=labels_path,
        indices=indices[:train_images]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CustomDataset(
        image_dir=image_dir,
        alpha=alpha,
        labels_path=labels_path,
        indices=indices[train_images:train_images + val_images]
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


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
        targets = F.one_hot(targets, 2).float()
        loss = loss_fn(predictions, targets)  # compute the loss

        # backward pass
        optimizer.zero_grad()  # reset the gradients
        scaler.scale(loss).backward()  # backward pass
        scaler.step(optimizer)  # perform the backward steps using the optimizer
        scaler.update()  # update the scaler

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

    return loss


def main():
    image_dir = f'{BASE_DATA_DIR}a_{format_alpha(ALPHA)}/image'
    if USE_MASKS:
        image_dir = f'{BASE_DATA_DIR}a_{format_alpha(ALPHA)}/mask'
    # Data loaders initialization
    train_loader, val_loader = get_loaders(
        image_dir,
        LABELS_PATH,
        ALPHA,
        BATCH_SIZE,
        NUM_WORKERS,
        PIN_MEMORY,
        TOTAL_IMAGES,
        TRAIN_IMAGES,
        VAL_IMAGES
    )

    # Model declaration
    # model = VGG_net(in_channels=1, no_classes=2, type='VGG_custom').to(DEVICE)
    model = ResNet50(1, 2).to(DEVICE)

    # Declaration of the optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Declaration of the loss function
    loss_fn = nn.BCEWithLogitsLoss()

    # Setup for temporary variables
    best_dice = 0  # best dice score obtained so far - for the validation set
    epoch_offset = 0  # offset of the nr of epochs (> 0 in case LOAD_MODEL = true)

    # Arrays for keeping track of the history
    loss_history = torch.tensor([])
    accuracies_history_train = torch.tensor([])
    accuracies_history_val = torch.tensor([])
    precision_history_train = torch.tensor([])
    precision_history_val = torch.tensor([])
    recall_history_train = torch.tensor([])
    recall_history_val = torch.tensor([])
    f1_history_train = torch.tensor([])
    f1_history_val = torch.tensor([])
    auc_history_train = torch.tensor([])
    auc_history_val = torch.tensor([])

    if LOAD_MODEL: # Loading the model
        epoch_offset, loss_history, accuracies_history_train, accuracies_history_val, precision_history_train, \
        precision_history_val, recall_history_train, recall_history_val, f1_history_train, f1_history_val, \
        auc_history_train, auc_history_val \
            = load_checkpoint(torch.load(LOAD_PATH), model)
        # Compute metrics related to the loaded model (validation set only)
        accuracies, precision, recall, f1, auc = compute_metrics(train_loader, model)
        if IS_INFERENCE:
            display_metrics(accuracies, precision, recall, f1, auc)
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
        accuracies_train, precision_train, recall_train, f1_train, auc_train = compute_metrics(train_loader, model)
        display_metrics(accuracies_train, precision_train, recall_train, f1_train, auc_train)
        accuracies_history_train = append_tensor_scalar(accuracies_history_train, accuracies_train)
        precision_history_train = append_tensor_scalar(precision_history_train, precision_train)
        recall_history_train = append_tensor_scalar(recall_history_train, recall_train)
        f1_history_train = append_tensor_scalar(f1_history_train, f1_train)
        auc_history_train = append_tensor_scalar(auc_history_train, auc_train)
        print(f'[... done]')
        print(f'[Validation metrics ...]')
        accuracies_val, precision_val, recall_val, f1_val, auc_val = compute_metrics(val_loader, model)
        display_metrics(accuracies_val, precision_val, recall_val, f1_val, auc_val)
        accuracies_history_val = append_tensor_scalar(accuracies_history_val, accuracies_val)
        precision_history_val = append_tensor_scalar(precision_history_val, precision_val)
        recall_history_val = append_tensor_scalar(recall_history_val, recall_val)
        f1_history_val = append_tensor_scalar(f1_history_val, f1_val)
        auc_history_val = append_tensor_scalar(auc_history_val, auc_val)
        print(f'[... done]')

        # display graphs
        if PLOT_GRAPHS:
            print("[Displaying graphs ...]")
            display_result_graph(loss_history, accuracies_history_train, accuracies_history_val, precision_history_train,
                precision_history_val, recall_history_train, recall_history_val, f1_history_train, f1_history_val,
                auc_history_train, auc_history_val, save_graphs=SAVE_GRAPHS, save_dir=GRAPHS_SAVE_DIR)
            print(f'[... done]')

        # save the checkpoint
        print("[Saving checkpoint ...]")
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "loss_history": loss_history,
            "accuracies_history_train": accuracies_history_train.cpu().detach(),
            "accuracies_history_val": accuracies_history_val.cpu().detach(),
            "precision_history_train": precision_history_train.cpu().detach(),
            "precision_history_val": precision_history_val.cpu().detach(),
            "recall_history_train": recall_history_train.cpu().detach(),
            "recall_history_val": recall_history_val.cpu().detach(),
            "f1_history_train": f1_history_train.cpu().detach(),
            "f1_history_val": f1_history_val.cpu().detach(),
            "auc_history_train": auc_history_train.cpu().detach(),
            "auc_history_val": auc_history_val.cpu().detach()
        }
        save_checkpoint(checkpoint, f"{SAVE_PATH}test_remove_{epoch}.pth.tar")
        print(f'[... done]')

        print(f'[Epoch: {epoch} ... done]')


if __name__ == "__main__":
    main()
