"""
This module implements evaluation of the model on the test set and creates the result file.
"""

import torch
import json

from utils.metrics import *
import os

from sklearn import preprocessing
# , multilabel_confusion_matrix
from sklearn.metrics import label_ranking_average_precision_score, confusion_matrix
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, jaccard_score, label_ranking_loss, coverage_error
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve, roc_auc_score, auc


def network_eval(model, Iplus, Iminus, M, loader, dataset, write_folder, device, hyp):

    model.eval()

    for i, (x, y) in enumerate(loader):

        x = x.to(device)
        y = y.to(device)

        model.eval()

        constrained_output = model(x.float(), Iplus, Iminus, M, device)
        predicted = constrained_output.data > 0.5

        # Move output and label back to cpu to be processed by sklearn
        predicted = predicted.to('cpu')
        cpu_constrained_output = constrained_output.to('cpu')
        y = y.to('cpu')
        if i == 0:
            predicted_test = predicted
            constr_test = cpu_constrained_output
            y_test = y
        else:
            predicted_test = torch.cat((predicted_test, predicted), dim=0)
            constr_test = torch.cat(
                (constr_test, cpu_constrained_output), dim=0)
            y_test = torch.cat((y_test, y), dim=0)

    different_from_0 = (y_test.sum(0) != 0).clone().detach()
    y_test = y_test[:, different_from_0]
    constr_test = constr_test[:, different_from_0]
    predicted_test = predicted_test[:, different_from_0]


    average_prob = []
    for i in range(predicted_test.shape[0]):
        pred = predicted_test[i].float()
        y = y_test[i].float()
        y_np = y.cpu().numpy()
        pred_np = pred.cpu().numpy()
        average_prob.append(
            sum(y_np * pred_np + abs(1 - y_np) * (1 - pred_np)) / len(pred_np)
        )
    average_prob = sum(average_prob) / len(average_prob)
    
    acc = accuracy_score(y_test, predicted_test)
    hamming = hamming_loss(y_test, predicted_test)
    multilabel_accuracy = jaccard_score(
        y_test, predicted_test, average='micro')
    ranking_loss = label_ranking_loss(y_test, constr_test.data)
    avg_precision = label_ranking_average_precision_score(
        y_test, constr_test.data)
    cov_error = (coverage_error(y_test, constr_test.data) - 1) / \
        constr_test.shape[1]
    one_err = one_error(y_test, constr_test.data)

    print("starting writing....")

    if not os.path.exists(write_folder):
        os.makedirs(write_folder)

    file_path = os.path.join(write_folder, dataset + '.json')

    data = {
        'split': hyp['split'],
        'seed': hyp['seed'],
        'best_epoch': hyp['best_epoch'],
        'hidden_dim': hyp['hidden_dim'],
        'acc':acc,
        'hamming': hamming,
        'multilabel_accuracy': multilabel_accuracy,
        'ranking_loss': ranking_loss,
        'cov_error': cov_error,
        'avg_precision': avg_precision,
        'one_err': one_err,
        "probability": float(average_prob),
        'END': 'END',
        'hyp': hyp
    }

    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = []
    existing_data.append(data)
    with open(file_path, 'w') as f:
        json.dump(existing_data, f, indent=4)
