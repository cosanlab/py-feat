import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch.nn as nn


def dice_loss(pred, target, smooth=1):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    # https://github.com/ZhiwenShao/PyTorch-JAANet
    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth)) / iflat.size(0)


def au_softmax_loss(input, target, weight=None, size_average=True, reduce=True):
    # https://github.com/ZhiwenShao/PyTorch-JAANet
    classify_loss = nn.NLLLoss(
        size_average=size_average, reduce=reduce, ignore_index=9)

    for i in range(input.size(2)):

        t_input = input[:, :, i]
        t_target = target[:, i].long()

        t_loss = classify_loss(t_input, t_target)
        if weight is not None:
            t_loss = t_loss * weight[i]
        t_loss = torch.unsqueeze(t_loss, 0)
        if i == 0:
            loss = t_loss
        else:
            loss = torch.cat((loss, t_loss), 0)

    if size_average:
        return loss.mean()
    else:
        return loss.sum()


def au_dice_loss(input, target, weight=None, smooth=1, size_average=True):
    # https://github.com/ZhiwenShao/PyTorch-JAANet
    for i in range(input.size(2)):
        # input is log_softmax, t_input is probability
        t_input = (input[:, 1, i]).exp()
        t_target = (target[:, i]).float()
        t_loss = dice_loss(t_input, t_target, smooth)
        if weight is not None:
            t_loss = t_loss * weight[i]
        t_loss = torch.unsqueeze(t_loss, 0)
        if i == 0:
            loss = t_loss
        else:
            loss = torch.cat((loss, t_loss), 0)

    if size_average:
        return loss.mean()
    else:
        return loss.sum()


def calculate_AU_weight(master_dataframe):
    """
    Calculates the AU weight according to a occurence dataframe 
    inputs: 
        occurence_df: a pandas dataframe containing occurence of each AU. See BP4D+
    """
    #occurence_df = occurence_df.rename(columns = {'two':'new_name'})
    occurence_df = master_dataframe[[
        "1", "2", "4", "6", "7", "10", "12", "14", "15", "17", "23", "24"]]

    weight_mtrx = np.zeros((occurence_df.shape[1], 1))
    for i in range(occurence_df.shape[1]):
        weight_mtrx[i] = np.sum(occurence_df.iloc[:, i]
                                > 0) / float(occurence_df.shape[0])
    weight_mtrx = 1.0/weight_mtrx

    print(weight_mtrx)
    weight_mtrx[weight_mtrx == np.inf] = 0
    print(np.sum(weight_mtrx)*len(weight_mtrx))
    weight_mtrx = weight_mtrx / np.sum(weight_mtrx) * len(weight_mtrx)

    return weight_mtrx


def AU_detection_evalv2(loader, drml_net, use_gpu=True):
    # https://github.com/ZhiwenShao/PyTorch-JAANet
    missing_label = 9
    for i, batch in enumerate(loader):
        img, label = batch
        if use_gpu:
            img, label = img.cuda(), label.cuda()

        pred_au = drml_net(img)
        pred_au = (pred_au[:, 1, :]).exp()
        if i == 0:
            all_pred_au = pred_au.data.cpu().float()
            all_au = label.data.cpu().float()
        else:
            all_pred_au = torch.cat(
                (all_pred_au, pred_au.data.cpu().float()), 0)
            all_au = torch.cat((all_au, label.data.cpu().float()))
    AUoccur_pred_prob = all_pred_au.data.numpy()
    AUoccur_actual = all_au.data.numpy()

    # AUs
    AUoccur_pred = np.zeros(AUoccur_pred_prob.shape)
    AUoccur_pred[AUoccur_pred_prob < 0.5] = 0
    AUoccur_pred[AUoccur_pred_prob >= 0.5] = 1

    AUoccur_actual = AUoccur_actual.transpose((1, 0))
    AUoccur_pred = AUoccur_pred.transpose((1, 0))

    f1score_arr = np.zeros(AUoccur_actual.shape[0])
    acc_arr = np.zeros(AUoccur_actual.shape[0])

    for i in range(AUoccur_actual.shape[0]):
        curr_actual = AUoccur_actual[i]
        curr_pred = AUoccur_pred[i]
        new_curr_actual = curr_actual[curr_actual != missing_label]
        new_curr_pred = curr_pred[curr_actual != missing_label]
        f1score_arr[i] = f1_score(new_curr_actual, new_curr_pred)
        acc_arr[i] = accuracy_score(new_curr_actual, new_curr_pred)

    return f1score_arr, acc_arr, AUoccur_actual, AUoccur_pred
