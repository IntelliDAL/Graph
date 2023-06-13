import os
import numpy as np
import torch
from texttable import Texttable
from scipy import stats
from sklearn.metrics import f1_score, precision_score, recall_score

def calculate_metrics(predicted_labels,true_labels, task_type=2):
    true_labels_np = true_labels
    predicted_labels_np = predicted_labels

    if task_type >2:

        f1 = f1_score(true_labels_np, predicted_labels_np, average='macro')
        precision = precision_score(true_labels_np, predicted_labels_np, average='macro')
        recall = recall_score(true_labels_np, predicted_labels_np, average='macro')
    elif task_type ==2:
        true_labels_np = true_labels.squeeze()
        predicted_labels_np = torch.argmax(predicted_labels,dim=1).squeeze()
        f1 = f1_score(true_labels_np, predicted_labels_np)
        precision = precision_score(true_labels_np, predicted_labels_np)
        recall = recall_score(true_labels_np, predicted_labels_np)
    else:
        raise ValueError("Invalid task_type. Supported values are 'multi' and 'binary'.")

    return f1, precision, recall


def computing_precision_ks(trues, predictions, ks, inclusive=True, rm=0):
    assert trues.shape == predictions.shape
    m, n = trues.shape

    precision_ks = np.zeros((m, len(ks)))
    inclusive_final_true_ks = np.zeros((m, len(ks)))
    inclusive_final_pred_ks = np.zeros((m, len(ks)))

    for i in range(m):

        for k_idx, k in enumerate(ks):
            assert (type(k) is int and 0 < k < n)
            true_ids, true_k = top_k_ids(trues, i, k, inclusive, rm)
            pred_ids, pred_k = top_k_ids(predictions, i, k, inclusive, rm)
            precision_ks[i][k_idx] = min(len(set(true_ids).intersection(set(pred_ids))), k) / k
            inclusive_final_true_ks[i][k_idx] = true_k
            inclusive_final_pred_ks[i][k_idx] = pred_k
    return np.mean(precision_ks, axis=0), np.mean(inclusive_final_true_ks, axis=0), np.mean(inclusive_final_pred_ks,
                                                                                            axis=0)


def metrics_spearmanr_rho(true, predication):
    assert true.shape == predication.shape
    rho, p_val = stats.spearmanr(true, predication)
    return rho


def metrics_kendall_tau(true, predication):
    assert true.shape == predication.shape
    tau, p_val = stats.kendalltau(true, predication)
    return tau


def metrics_mean_square_error(true, predication):
    assert true.shape == predication.shape
    mse = (np.square(true - predication).mean())
    return mse


def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.set_cols_dtype(['t', 't'])
    t.add_rows([['Parameter', 'Value']] + [[k.replace('_', ' ').capitalize(), args[k]] for k in keys])
    print(t.draw())


def top_k_ids(data, k, inclusive, rm):
    """
    :param data: input
    :param k:
    :param inclusive: whether to be tie inclusive or not.
        For example, the ranking may look like this:
        7 (sim_score=0.99), 5 (sim_score=0.99), 10 (sim_score=0.98), ...
        If tie inclusive, the top 1 results are [7, 9].
        Therefore, the number of returned results may be larger than k.
        In summary,
            len(rtn) == k if not tie inclusive;
            len(rtn) >= k if tie inclusive.
    :param rm: 0
    :return: for a query, the ids of the top k database graph
    ranked by this model.
    """
    sort_id_mat = np.argsort(-data)
    n = sort_id_mat.shape[0]
    if k < 0 or k >= n:
        raise RuntimeError('Invalid k {}'.format(k))
    if not inclusive:
        return sort_id_mat[:k]
    # Tie inclusive.
    dist_sim_mat = data
    while k < n:
        cid = sort_id_mat[k - 1]
        nid = sort_id_mat[k]
        if abs(dist_sim_mat[cid] - dist_sim_mat[nid]) <= rm:
            k += 1
        else:
            break
    return sort_id_mat[:k]


def prec_at_ks(true_r, pred_r, ks, rm=0):
    """
    Ranking-based. prec@ks.
    :param true_r: result object indicating the ground truth.
    :param pred_r: result object indicating the prediction.
    :param ks: k
    :param rm: 0
    :return: precision at ks.
    """
    true_ids = top_k_ids(true_r, ks, inclusive=True, rm=rm)
    pred_ids = top_k_ids(pred_r, ks, inclusive=True, rm=rm)
    ps = min(len(set(true_ids).intersection(set(pred_ids))), ks) / ks
    return ps


def ranking_func(data):
    sort_id_mat = np.argsort(-data)
    n = sort_id_mat.shape[0]
    rank = np.zeros(n)
    for i in range(n):
        finds = np.where(sort_id_mat == i)
        fid = finds[0][0]
        while fid > 0:
            cid = sort_id_mat[fid]
            pid = sort_id_mat[fid - 1]
            if data[pid] == data[cid]:
                fid -= 1
            else:
                break
        rank[i] = fid + 1

    return rank


def calculate_ranking_correlation(rank_corr_function, prediction, target):
    """
    Calculating specific ranking correlation for predicted values.
    :param rank_corr_function: Ranking correlation function.
    :param prediction: Vector of predicted values.
    :param target: Vector of ground-truth values.
    :return ranking: Ranking correlation value.
    """
    r_prediction = ranking_func(prediction)
    r_target = ranking_func(target)

    return rank_corr_function(r_prediction, r_target).correlation


def write_log_file(file_name_path, log_str, print_flag=True):
    if print_flag:
        print(log_str)
    if log_str is None:
        log_str = 'None'
    if os.path.isfile(file_name_path):
        with open(file_name_path, 'a+') as log_file:
            log_file.write(log_str + '\n')
    else:
        with open(file_name_path, 'w+') as log_file:
            log_file.write(log_str + '\n')


def log_args(file_name_path, args):
    v = vars(args)
    with open(file_name_path, 'w+') as log_file:
        for k in v.keys():
            log_file.write(str(k) + ':' + str(v[k]) + '\n')


def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return '\nmake dirs # {} '.format(directory)
    else:
        return "\ndirs already exist! Cannot be created"


def plt_mat(t):
    import matplotlib.pyplot as plt
    plt.imshow(t.detach().cpu().numpy())
    plt.colorbar()
    plt.show()
