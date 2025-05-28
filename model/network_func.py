"""
This module implements evaluation of the model on the test set and creates the result file.
"""

import torch
import json
import math

from utils.metrics import *
import os
import numpy as np

from sklearn import preprocessing
# , multilabel_confusion_matrix
from sklearn.metrics import label_ranking_average_precision_score, confusion_matrix
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, jaccard_score, label_ranking_loss, coverage_error
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve, roc_auc_score, auc

def check_top_n(predicted_distr, actual, combinations_valid, n = 20):
    """Checks top n combination whether they are correct
    """
    n_test_samples = len(actual)
    n_classes = len(actual[0])
    predicted = np.zeros(shape = (n_test_samples, n_classes))
    for index, output_distr in enumerate(predicted_distr):
        top_n_pred = np.argsort(output_distr)[-n:][::-1]
        correct = np.all(combinations_valid[top_n_pred] == actual[index].cpu().numpy()[np.newaxis, :], axis=1)
        if correct.sum()>0:
            predicted[index,:] = combinations_valid[top_n_pred[np.argmax(correct)]]
        else:
            predicted[index,:] = combinations_valid[top_n_pred[0]]
    return accuracy_score(actual, predicted)

def read_constrains(path):
    data = ""
    constraints = []

    with open(path, "r") as file:
        data = file.read()
        data = [constraint.split(" ")[:-1] for constraint in data.split('\n')]
        for constraint in data[:-1]:
            head = constraint[1]
            positive_body = []
            negative_body = []
            for cl in constraint[3:]:
                if "n" in cl:
                    negative_body.append(cl[1:])
                else:
                    positive_body.append(cl)
            constraints.append((positive_body, negative_body, head))
    return constraints

def check_if_satisfies_constraints(combination, constraints) -> bool:
    for constraint in constraints:
        to_skip = False
        positive_body = [int(x) for x in constraint[0]]
        negative_body = [int(x) for x in constraint[1]]
        head = int(constraint[2])

        if len(combination.shape) > 1:
            combination = combination[0]

        if combination[head] == 1:
            continue
        
        for ind in positive_body:
            if combination[ind] == 0:
                to_skip = True
                break

        if to_skip:
            continue

        for ind in negative_body:
            if combination[ind] == 1:
                to_skip = True
                break

        if to_skip:
            continue

        if combination[head] == 0:
            return False
    
    return True

def load_valid_combinations(path, n_classes, include_constraints=True):
    all_combinations = np.array([np.expand_dims(np.array(list(format(i, f"0{n_classes}b")), dtype=int), 0)
                                for i in range(2**n_classes)])
    if not include_constraints:
        all_combinations = all_combinations.reshape(-1, n_classes)
        return all_combinations
    
    # Read constraints from the file
    constraints = read_constrains(path)
    valid_constraint = []
    for comb in all_combinations:
        if check_if_satisfies_constraints(comb, constraints):
            valid_constraint.append(comb[0])
    return valid_constraint

def generate_distribution_independent_preds(preds, combinations_valid):
    distributions = np.zeros((len(preds), len(combinations_valid)))
    for index, pred in enumerate(preds):
        pred = np.array(pred)[np.newaxis, :]
        combinations_valid = np.array(combinations_valid)
        distr = np.prod((combinations_valid * pred) + ((1 - combinations_valid) * (1 - pred)), axis=1)
        distributions[index] = distr * (1/distr.sum())
    return distributions


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

    average_prob = []
    for i in range(predicted_test.shape[0]):
        pred = constr_test.data[i].float()
        y = y_test[i].float()
        y_np = y.cpu().numpy()
        pred_np = pred.cpu().numpy()
        average_prob.append(
            math.prod(y_np * pred_np + abs(1 - y_np) * (1 - pred_np))
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

    
    if dataset in ["emotions", "scene", "yeast"]:
        combinations_valid = np.array(load_valid_combinations(f"data\{dataset}\{dataset}_constraints.txt", len(y_test[0]), include_constraints=False))
        distr = generate_distribution_independent_preds(constr_test.data, combinations_valid)
        acc1 = check_top_n(distr, y_test, combinations_valid, n=1)
        acc2 = check_top_n(distr, y_test, combinations_valid, n=2)
        acc5 = check_top_n(distr, y_test, combinations_valid, n=5)
        acc10 = check_top_n(distr, y_test, combinations_valid, n=10)
    else:
        acc1, acc2, acc5, acc10 = 0, 0, 0, 0
    data = {
        'split': hyp['split'],
        'seed': hyp['seed'],
        'best_epoch': hyp['best_epoch'],
        'hidden_dim': hyp['hidden_dim'],
        'acc':acc,
        "acc1": acc1,
        "acc2": acc2,
        "acc5": acc5,
        "acc10": acc10,
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
