import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, recall_score, precision_score

np.seterr(divide='ignore', invalid='ignore')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # do not modify


def format_alpha(alpha):
    return int(round(alpha * 100, 2))


def enhance_index(index):
    if index == 0:
        return '00000'
    digits = 0
    res = ''
    cp = index
    while cp > 0:
        digits += 1
        cp = (int)(cp / 10)

    while digits < 5:
        res += '0'
        digits += 1

    res += str(index)

    return res


def append_tensor_scalar(t1, t2):
    t2p = torch.unsqueeze(t2, 0)
    return torch.cat((t1.to(DEVICE), t2p.to(DEVICE)))


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint["epoch"], \
           torch.as_tensor(checkpoint["loss_history_train"]), \
           torch.as_tensor(checkpoint["loss_history_val"]), \
           torch.as_tensor(checkpoint["accuracies_history_train"]),\
           torch.as_tensor(checkpoint["accuracies_history_val"]), \
           torch.as_tensor(checkpoint["precision_history_train"]), \
           torch.as_tensor(checkpoint["precision_history_val"]), \
           torch.as_tensor(checkpoint["recall_history_train"]), \
           torch.as_tensor(checkpoint["recall_history_val"]), \
           torch.as_tensor(checkpoint["f1_history_train"]), \
           torch.as_tensor(checkpoint["f1_history_val"]), \
           torch.as_tensor(checkpoint["auc_history_train"]), \
           torch.as_tensor(checkpoint["auc_history_val"])


def compute_metrics(loader, model):
    '''
    Returns [accuracy_left, accuracy_right, accuracy_mean],
            [precision_left, precision_right, precision_mean],
            [recall_left, recall_right, recall_mean],
            [f1_left, f1_right, f1_mean],
            auc
    '''
    predictions_total = np.array([])
    y_total = np.array([])
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).unsqueeze(1)
            y_total = np.append(y_total, y.cpu().detach().numpy())
            preds = model(x)
            preds = torch.unsqueeze(torch.argmax(preds, 1), 1)
            predictions_total = np.append(predictions_total, preds.cpu().detach().numpy())

    model.train()

    '''
    # METRICS FOR SWITCH PROBLEM
    conf_mat = confusion_matrix(y_total, predictions_total)
    recall = np.diag(conf_mat) / np.sum(conf_mat, axis=1)
    # recall = recall_score(y_total, predictions_total, average='samples')
    precision = np.diag(conf_mat) / np.sum(conf_mat, axis=0)
    # precision = precision_score(y_total, predictions_total, average='samples')
    f1 = 2 * recall * precision / (recall + precision)
    recall_mean = np.mean(recall)
    precision_mean = np.mean(precision)
    f1_mean = np.mean(f1)
    recall = np.append(recall, recall_mean)
    precision = np.append(precision, precision_mean)
    f1 = np.append(f1, f1_mean)
    accuracies = conf_mat.diagonal()/conf_mat.sum(axis=1)
    accuracy_mean = accuracy_score(y_total, predictions_total)
    accuracies = np.append(accuracies, accuracy_mean)
    auc = roc_auc_score(y_total, predictions_total)

    tn, fp, fn, tp = conf_mat.ravel()
    print(f"[CONF MAT: tp:{tp} fp:{fp} fn:{fn} tn:{tn}]")

    return torch.tensor(accuracies * 100), \
           torch.tensor(precision * 100), \
           torch.tensor(recall * 100), \
           torch.tensor(f1 * 100), \
           torch.tensor(auc * 100)

    '''

    # METRICS FOR CIFAR-10 PROBLEM
    conf_mat = confusion_matrix(y_total, predictions_total)
    accuracies = conf_mat.diagonal() / conf_mat.sum(axis=1)
    precision = np.diag(conf_mat) / np.sum(conf_mat, axis=0)
    recall = np.diag(conf_mat) / np.sum(conf_mat, axis=1)
    f1 = 2 * recall * precision / (recall + precision)
    print(accuracies)

    return torch.tensor([np.mean(accuracies),np.mean(accuracies),np.mean(accuracies)]),\
           torch.tensor([np.mean(precision),np.mean(precision),np.mean(precision)]),\
           torch.tensor([np.mean(recall),np.mean(recall),np.mean(recall)]),\
           torch.tensor([np.mean(f1),np.mean(f1),np.mean(f1)]),\
           torch.tensor(0)


def display_metrics(accuracies, precision, recall, f1, auc):
    print(f"Accuracies: [{accuracies[0]:.2f}%, {accuracies[1]:.2f}%, {accuracies[2]:.2f}%]")
    print(f"Precision: [{precision[0]:.2f}%, {precision[1]:.2f}%, {precision[2]:.2f}%]")
    print(f"Recall: [{recall[0]:.2f}%, {recall[1]:.2f}%, {recall[2]:.2f}%]")
    print(f"F1: [{f1[0]:.2f}%, {f1[1]:.2f}%, {f1[2]:.2f}%]")
    print(f"AUC: {auc:.2f}%")


def display_result_graph(loss_history_train, loss_history_val, accuracies_history_train, accuracies_history_val, precision_history_train,
                         precision_history_val, recall_history_train, recall_history_val, f1_history_train,
                         f1_history_val, auc_history_train, auc_history_val,
                         save_graphs=False, save_dir='/'):
    fig, ax = plt.subplots(nrows=3, ncols=2)
    # loss graph
    ax[0][0].plot(loss_history_train.cpu().detach().numpy(), label="Training")
    ax[0][0].plot(loss_history_val.cpu().detach().numpy(), label="Validation")
    ax[0][0].set_xlabel('Epochs')
    ax[0][0].set_ylabel('Loss')
    # accuracy graph
    ax[0][1].plot(accuracies_history_train[:, 2].cpu().detach().numpy(), label="Training")
    ax[0][1].plot(accuracies_history_val[:, 2].cpu().detach().numpy(), label="Validation")
    ax[0][1].set_xlabel('Epochs')
    ax[0][1].set_ylabel('Accuracy score')
    # precision graph
    ax[1][0].plot(precision_history_train[:, 2].cpu().detach().numpy(), label="Training")
    ax[1][0].plot(precision_history_val[:, 2].cpu().detach().numpy(), label="Validation")
    ax[1][0].set_xlabel('Epochs')
    ax[1][0].set_ylabel('Precision score')
    # recall graph
    ax[1][1].plot(recall_history_train[:, 2].cpu().detach().numpy(), label="Training")
    ax[1][1].plot(recall_history_val[:, 2].cpu().detach().numpy(), label="Validation")
    ax[1][1].set_xlabel('Epochs')
    ax[1][1].set_ylabel('Recall score')
    # f1 graph
    ax[2][0].plot(f1_history_train[:, 2].cpu().detach().numpy(), label="Training")
    ax[2][0].plot(f1_history_val[:, 2].cpu().detach().numpy(), label="Validation")
    ax[2][0].set_xlabel('Epochs')
    ax[2][0].set_ylabel('F1 score')
    # auc graph
    ax[2][1].plot(auc_history_train.cpu().detach().numpy(), label="Training")
    ax[2][1].plot(auc_history_val.cpu().detach().numpy(), label="Validation")
    ax[2][1].set_xlabel('Epochs')
    ax[2][1].set_ylabel('Auc score')

    if save_graphs:
        plt.savefig(f'{save_dir}epoch_{len(loss_history_train.cpu().detach().numpy())-1}')
    else:
        plt.show()
