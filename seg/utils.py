import cv2
import torch
import torchvision
from torch.utils.data import DataLoader
from dataset import CustomDataset
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, RocCurveDisplay
import torch.nn as nn
import cv2


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # do not modify


def append_tensor_scalar(t1, t2):
    t2p = torch.unsqueeze(t2, 0)
    return torch.cat((t1.to(DEVICE), t2p.to(DEVICE)))


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint["epoch"], torch.as_tensor(checkpoint["loss_history"]), torch.as_tensor(checkpoint["iou_history_train"]),\
           torch.as_tensor(checkpoint["iou_history_val"]), torch.as_tensor(checkpoint["iou_rails_history_train"]), \
           torch.as_tensor(checkpoint["iou_rails_history_val"]), torch.as_tensor(checkpoint["iou_bg_history_train"]),\
           torch.as_tensor(checkpoint["iou_bg_history_val"]), torch.as_tensor(checkpoint["dice_history_train"]), \
           torch.as_tensor(checkpoint["dice_history_val"]), checkpoint["indices"], checkpoint["config"]


def get_loaders(
    image_dir,
    mask_dir,
    batch_size,
    num_workers=4,
    pin_memory=True,
    image_width=224,
    image_height=224,
    total_images=20,
    train_images=10,
    val_images=5,
    shuffle_images=False,
    indices=None
):
    if train_images + val_images > total_images:
        print('[ERROR: total_images must be >= than train_images + val_images]')
        return None
    # generate indices
    if indices is None:
        indices = np.arange(total_images)
        if shuffle_images:
            np.random.shuffle(indices)

    train_ds = CustomDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_dim=(image_width, image_height),
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
        mask_dir=mask_dir,
        image_dim=(image_width, image_height),
        indices=indices[train_images:train_images + val_images]
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader, indices


def compute_metrics_different_thresholds(loader, model, thresholds, epoch, save_graphs=False, save_dir='/'):

    iou = torch.tensor([]).to(DEVICE)
    iou_rails = torch.tensor([]).to(DEVICE)
    iou_bg = torch.tensor([]).to(DEVICE)
    dice = torch.tensor([]).to(DEVICE)
    for th in thresholds:
        results = compute_metrics(loader, model, threshold=th)
        iou = append_tensor_scalar(iou, results[0])
        iou_rails = append_tensor_scalar(iou_rails, results[1])
        iou_bg = append_tensor_scalar(iou_bg, results[2])
        dice = append_tensor_scalar(dice, results[3])

    plt.plot(thresholds, iou.cpu().detach(), label="IoU")
    plt.plot(thresholds, iou_rails.cpu().detach(), label="IoU Rails")
    plt.plot(thresholds, iou_bg.cpu().detach(), label="IoU Background")
    plt.plot(thresholds, dice.cpu().detach(), label="Dice")
    plt.title(f'Different thresholds comparison - epoch {epoch}')
    plt.xlabel('Thresholds')
    plt.ylabel('Scores')
    plt.legend()
    if save_graphs:
        plt.savefig(f'{save_dir}th_epoch_{epoch}.png')
    else:
        plt.show()


def compute_metrics(loader, model, threshold = 0.5):
    '''
    Computes the dice and iou scores
    '''
    tp = torch.tensor(0).to(DEVICE)
    tp_bg = torch.tensor(0).to(DEVICE)
    tp_rails = torch.tensor(0).to(DEVICE)
    num_pixels = torch.tensor(0).to(DEVICE)
    num_pixels_rails = torch.tensor(0).to(DEVICE)
    epsilon = 1e-6
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            # y = torch.unsqueeze(y.to(DEVICE), 1)
            y = y.to(DEVICE)
            preds = model(x)
            preds = torch.where(preds > threshold, 1, 0)
            tp = torch.add(tp, torch.sum(torch.where(preds == y, 1, 0)))
            tp_rails = torch.add(tp_rails, torch.sum(torch.mul(preds, y)))
            preds_copy = preds.clone()
            y_copy = y.clone()
            preds_copy[preds == 0] = 1
            preds_copy[preds != 0] = 0
            y_copy[y == 0] = 1
            y_copy[y != 0] = 0
            tp_bg = torch.add(tp_bg, torch.sum(torch.mul(preds_copy, y_copy)))
            num_pixels = torch.add(num_pixels, torch.numel(preds))
            num_pixels_rails = torch.add(num_pixels_rails, torch.sum(torch.where(y == 1, 1, 0)))

    iou = torch.mul(torch.div(tp, num_pixels), 100)
    iou_rails = torch.mul(torch.div(tp_rails, num_pixels_rails), 100)
    iou_bg = torch.mul(torch.div(tp_bg, (torch.subtract(num_pixels, num_pixels_rails))), 100)
    dice_score = torch.mul(torch.div((torch.add(torch.mul(tp, 2), epsilon)), (torch.add(tp,  torch.add(num_pixels, epsilon)))), 100)
    model.train()

    return iou, iou_rails, iou_bg, dice_score


def save_predictions_as_imgs(
    epoch, batch_size, loader, model, score_dice, score_iou, score_iou_rail, threshold=0.5, folder="saved_images/", rows=4
):
    # rows = maximum no. of samples to display
    image_path = f'{folder}/dice{score_dice:.2f}_iou{score_iou:.2f}_iourail{score_iou_rail:.2f}_e{epoch}.png'
    total_x = torch.tensor([]).to(DEVICE)
    total_y = torch.tensor([]).to(DEVICE)
    total_preds = torch.tensor([]).to(DEVICE)
    model.eval()

    for idx, (x, y) in enumerate(loader):
        if idx * batch_size > rows:
            break
        y = y.to(DEVICE)
        x = x.to(DEVICE)
        # x = x.squeeze(1)
        # x = torch.swapaxes(x, 2, 3)
        # x = torch.swapaxes(x, 1, 2)
        with torch.no_grad():
            preds = model(x)
            preds = (preds > threshold).float()
            total_preds = torch.cat((total_preds, preds), 0)
            total_y = torch.cat((total_y, y), 0)
            total_x = torch.cat((total_x, x), 0)

    interleaved = torch.tensor([]).to(DEVICE)
    for idx in range(len(total_preds)):
        #unsqueezed = total_y[idx].unsqueeze(0)
        interleaved = torch.cat((interleaved, total_x[idx] / 255, total_y[idx], total_preds[idx]), 0)

    interleaved = interleaved.unsqueeze(1)
    total_preds_grid = torchvision.utils.make_grid(interleaved, nrow=3, pad_value=200)
    torchvision.utils.save_image(total_preds_grid, image_path)

    model.train()


def inference(model, image_dim):
    # used for timing the model
    x = torch.randn((1920, 1080, 3))
    x = x.cpu().detach().numpy()
    x = cv2.resize(x, dsize=image_dim, interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(x)
    x = np.expand_dims(x, axis=0)
    x = np.swapaxes(x, 2, 3)
    x = np.swapaxes(x, 1, 2)
    x = torch.from_numpy(x)
    start = time.time()
    _ = model(x)
    end = time.time()
    return end - start


def compute_ratio(train_loader, val_loader):
    # offline computation
    bg_pixels = 0
    rail_pixels = 0
    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).unsqueeze(1)
            bg_pixels += (y == 0).sum()
            rail_pixels += (y == 1).sum()

        for x, y in val_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).unsqueeze(1)
            bg_pixels += (y == 0).sum()
            rail_pixels += (y == 1).sum()

    return bg_pixels / rail_pixels
    # 0.0265 for rail_pixels / bg_pixels
    # 37.7584 for bg_pixels / rail_pixels


def plot_confusion_matrices_different_thresholds(loader, model, thresholds, epoch):
    predictions_total = [np.array([])] * len(thresholds)
    y_total = np.array([])
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).unsqueeze(1)
            y_total = np.append(y_total, y.cpu().detach().numpy())
            preds = model(x)
            for idx, th in enumerate(thresholds):
                preds_th = (preds > th).float()
                predictions_total[idx] = np.append(predictions_total[idx], preds_th.cpu().detach().numpy())

    model.train()

    nrows = int(len(thresholds) / 2) + 1
    if len(thresholds) % 2 == 0:
        nrows -= 1
    fig, ax = plt.subplots(nrows=nrows, ncols=2)
    fig.set_figheight(16)
    fig.set_figwidth(15)
    for idx, th in enumerate(thresholds):
        row = int(idx / 2)
        col = idx % 2
        preds_current = predictions_total[idx]
        conf_mat = confusion_matrix(y_total, preds_current)
        tn, fp, fn, tp = conf_mat.ravel()
        sns.heatmap(conf_mat, annot=True, cmap='Blues', ax=ax[row][col])
        ax[row][col].title.set_text(f'Threshold: {th:.1f}')
        ax[row][col].set_xlabel('Actual Values ')
        ax[row][col].set_ylabel('\nPredicted Values')
        ax[row][col].xaxis.set_ticklabels(['False', 'True'])
        ax[row][col].yaxis.set_ticklabels(['False', 'True'])
    plt.title(f"Different thresholds for epoch {epoch}")
    plt.show()


def plot_roc_curve_custom(loader, model, epoch, save_graphs=False, save_dir='/'):
    predictions_total = torch.tensor([]).to(DEVICE)
    y_total = torch.tensor([]).to(DEVICE)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).unsqueeze(1)
            y_total = torch.cat((y_total, torch.flatten(y)))
            preds = model(x)
            predictions_total = torch.cat((predictions_total, torch.flatten(preds)))

    t1 = y_total.cpu().detach().numpy()
    t2 = predictions_total.cpu().detach()
    fpr, tpr, threhsolds = roc_curve(t1, t2)
    roc_auc = roc_auc_score(t1, t2)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="Segmentation estimator")
    display.plot()
    if save_graphs:
        # plt.savefig(f'{save_dir}roc_epoch_{epoch}')
        pass
    else:
        plt.show()

    return roc_auc


def display_result_graph(loss_history, dice_history_train, dice_history_val, iou_history_train, iou_history_val, iou_rails_history_train,
                         iou_rails_history_val, save_graphs=False, save_dir='/'):
    fig, ax = plt.subplots(nrows=2, ncols=2)
    # loss graph
    ax[0][0].plot(loss_history.cpu().detach().numpy())
    ax[0][0].set_xlabel('Epochs')
    ax[0][0].set_ylabel('Loss')
    # dice graph
    ax[0][1].plot(dice_history_train.cpu().detach().numpy(), label="Training")
    ax[0][1].plot(dice_history_val.cpu().detach().numpy(), label="Validation")
    ax[0][1].set_xlabel('Epochs')
    ax[0][1].set_ylabel('Dice score')
    # iou graph
    ax[1][0].plot(iou_history_train.cpu().detach().numpy(), label="Training")
    ax[1][0].plot(iou_history_val.cpu().detach().numpy(), label="Validation")
    ax[1][0].set_xlabel('Epochs')
    ax[1][0].set_ylabel('IoU score')
    # iou rails graph
    ax[1][1].plot(iou_rails_history_train.cpu().detach().numpy(), label="Training")
    ax[1][1].plot(iou_rails_history_val.cpu().detach().numpy(), label="Validation")
    ax[1][1].set_xlabel('Epochs')
    ax[1][1].set_ylabel('IoU Rails score')
    plt.legend()
    if save_graphs:
        plt.savefig(f'{save_dir}epoch_{len(loss_history.cpu().detach().numpy()) - 1}')
    else:
        plt.show()


# PyTorch
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice