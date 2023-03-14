import numpy as np
import pandas as pd
import pickle

import os
from sklearn.model_selection import KFold
from sklearn.model_selection import KFold


def remove_space(adj):
    for index, row in adj.items():
        for i in range(len(row)):
            row[i] = float(str(row[i]).replace(' ', ''))
    adj = pd.DataFrame(adj, dtype=np.float64)
    return adj

def dealwith_nan(adj):
    a = adj.iloc[0, 0]
    b = type(adj.iloc[0, 0])
    print(adj.iloc[0, 0], type(adj.iloc[0, 0]))
    ss = adj.values[0, 0]
    adj = adj.replace(ss, '1')
    adj = remove_space(adj)

    return adj
if __name__ == "__main__":
    path = "../dataset/"
    label_path = "../dataset/final_id_tag_dict.csv"
    label_dataframe = pd.read_csv(label_path)
    mapping1 = {
        1: 'CN',
        7: 'CN',
        9: 'CN',
        2: 'MCI',
        4: 'MCI',
        8: 'MCI',
        3: 'AD',
        5: 'AD',
        6: 'AD'
    }
    label_dataframe.replace({'baseline_DXCHANGE': mapping1}, inplace=True)
    label_dataframe.replace({'m06_DXCHANGE': mapping1}, inplace=True)
    label_dataframe.replace({'m12_DXCHANGE': mapping1}, inplace=True)
    label_dataframe.replace({'m24_DXCHANGE': mapping1}, inplace=True)
    label_dataframe.replace({'m36_DXCHANGE': mapping1}, inplace=True)
    label_dataframe.replace({'m48_DXCHANGE': mapping1}, inplace=True)

    mapping2 = {
        'CN': 0,
        'MCI': 1,
        'AD': 2
    }
    label_dataframe.replace({'baseline_DXCHANGE': mapping2}, inplace=True)
    label_dataframe.replace({'m06_DXCHANGE': mapping2}, inplace=True)
    label_dataframe.replace({'m12_DXCHANGE': mapping2}, inplace=True)
    label_dataframe.replace({'m24_DXCHANGE': mapping2}, inplace=True)
    label_dataframe.replace({'m36_DXCHANGE': mapping2}, inplace=True)
    label_dataframe.replace({'m48_DXCHANGE': mapping2}, inplace=True)
    ids = label_dataframe["id"]
    column_ori = ["CDRSB", "ADAS11", "MMSE", "RAVLT_immediate",
                  "RAVLT_learning", "ADAS13", "RAVLT_forgetting",
                  "RAVLT_perc_forgetting", "MOCA"]
    column_baseline = ["baseline_CDRSB", "baseline_ADAS11", "baseline_MMSE", "baseline_RAVLT_immediate",
                       "baseline_RAVLT_learning", "baseline_ADAS13", "baseline_RAVLT_forgetting",
                       "baseline_RAVLT_perc_forgetting", "basleine_MOCA"]
    column_m06 = ["m06_CDRSB", "m06_ADAS11", "m06_MMSE", "m06_RAVLT_immediate", "m06_RAVLT_learning",
                  "m06_ADAS13", "m06_RAVLT_forgetting", "m06_RAVLT_perc_forgetting", "m06_MOCA"]
    column_m12 = ["m12_CDRSB", "m12_ADAS11", "m12_MMSE", "m12_RAVLT_immediate", "m12_RAVLT_learning",
                  "m12_ADAS13", "m12_RAVLT_forgetting", "m12_RAVLT_perc_forgetting", "m12_MOCA"]
    column_m24 = ["m24_CDRSB", "m24_ADAS11", "m24_MMSE", "m24_RAVLT_immediate", "m24_RAVLT_learning",
                  "m24_ADAS13", "m24_RAVLT_forgetting", "m24_RAVLT_perc_forgetting", "m24_MOCA"]
    column_m36 = ["m36_CDRSB", "m36_ADAS11", "m36_MMSE", "m36_RAVLT_immediate", "m36_RAVLT_learning",
                  "m36_ADAS13", "m36_RAVLT_forgetting", "m36_RAVLT_perc_forgetting", "m36_MOCA"]
    column_m48 = ["m48_CDRSB", "m48_ADAS11", "m48_MMSE", "m48_RAVLT_immediate", "m48_RAVLT_learning",
                  "m48_ADAS13", "m48_RAVLT_forgetting", "m48_RAVLT_perc_forgetting", "m48_MOCA"]
    list_adj_all = []
    list_series_all = []
    list_label_all = []

    for id in ids:
        data_adj_baseline = pd.read_csv("../dataset/baseline_ROICorrelation_"+str(id)+".txt",header=None).iloc[0:90,:90]
        data_adj_m06 = pd.read_csv("../dataset/m06_ROICorrelation_"+str(id)+".txt",header=None).iloc[0:90,:90]
        data_adj_m12 = pd.read_csv("../dataset/m12_ROICorrelation_"+str(id)+".txt",header=None).iloc[0:90,:90]

        data_adj_baseline = dealwith_nan(data_adj_baseline)
        data_adj_m06 = dealwith_nan(data_adj_m06)
        data_adj_m12 = dealwith_nan(data_adj_m12)
        if np.sum(np.sum(data_adj_baseline==1.0))>(8100*0.6) or np.sum(np.sum(data_adj_m06==1.0))>(8100*0.6) or np.sum(np.sum(data_adj_m12==1.0))>(8100*0.6):
            continue
        # data_adj_baseline[abs(data_adj_baseline) < 0.3] = 0
        # data_adj_m06[abs(data_adj_m06) < 0.3] = 0
        # data_adj_m12[abs(data_adj_m12) < 0.3] = 0

        data_x_adj = np.stack((data_adj_baseline.values, data_adj_m06.values, data_adj_m12.values),axis=0)

        data_series_baseline = pd.read_csv("../dataset/baseline_ROISignals_"+str(id)+".txt",header=None)
        data_series_m06 = pd.read_csv("../dataset/m06_ROISignals_"+str(id)+".txt",header=None)
        data_series_m12 = pd.read_csv("../dataset/m12_ROISignals_"+str(id)+".txt",header=None)
        # if  data_series_baseline.empty or data_series_m06.empty or data_series_m12.empty:
        #     continue
        if data_series_baseline.shape[0]!=135 or data_series_m06.shape[0]!=135 or data_series_m12.shape[0]!=135:
            print("aaaaa")
            continue

        data_x_series = np.stack((data_series_baseline.values, data_series_m06.values, data_series_m12.values),axis=0)
        data_adj_baseline.replace(np.nan,'0')
        data_adj_m06.replace(np.nan,'0')
        data_adj_m12.replace(np.nan,'0')
        # data_adj_baseline.replace(-np.inf,np.nan)
        # data_adj_m06.replace(-np.inf,np.nan)
        # data_adj_m12.replace(-np.inf,np.nan)
        if data_adj_baseline.isnull().values.any() or \
                data_adj_m06.isnull().values.any() or \
                data_adj_m12.isnull().values.any() :
            continue
            print(file_name, "after has nan")


        if not np.isfinite(data_adj_baseline.values).all() \
                and not np.isfinite(data_adj_baseline.values).all() \
                and not np.isfinite(data_adj_baseline.values).all():
            continue
            print(file_name, "after has inf")

        label = label_dataframe[label_dataframe["id"]==id].iloc[:,1:]
        list_adj_all.append(data_x_adj)
        list_series_all.append(data_x_series)
        list_label_all.append(label)
data_adj = np.asarray(list_adj_all)
data_series = np.asarray(list_series_all)
labels = np.asarray(list_label_all)
labels = np.squeeze(labels)    #  a 中所有为 1 的维度删掉

mean  = np.nanmean(labels,axis=0)
std = np.nanstd(labels,axis=0)
# labels[:,6:] =(labels[:,6:] - mean[6:]) / std[6:]

a_dict = {'mean': mean[6:],  'std': std[6:]}

# pickle a variable to a file
file = open('mean_std.pickle', 'wb')
pickle.dump(a_dict, file)
file.close()

filename_adj_data_all = '../dataset/train_test/train_data_adj_all.npy'
filename_series_data_all = '../dataset/train_test/train_data_series_all.npy'
filename_labels_all = '../dataset/train_test/train_label_all.npy'

filename_adj_data_24 = '../dataset/train_test/train_data_adj_all_24.npy'
filename_series_data_24 = '../dataset/train_test/train_data_series_all_24.npy'
filename_labels_24 = '../dataset/train_test/train_label_all_24.npy'

filename_adj_data_36 = '../dataset/train_test/train_data_adj_all_36.npy'
filename_series_data_36 = '../dataset/train_test/train_data_series_all_36.npy'
filename_labels_36  = '../dataset/train_test/train_label_all_36.npy'

filename_adj_data_48 = '../dataset/train_test/train_data_adj_all_48.npy'
filename_series_data_48 = '../dataset/train_test/train_data_series_all_48.npy'
filename_labels_48 = '../dataset/train_test/train_label_all_48.npy'

labels_24 = labels[~np.isnan(labels[:,3])]
data24 = data_adj[~np.isnan(labels[:,3])]
series24 = data_series[~np.isnan(labels[:,3])]

data_24_strue = []
# for i in range(len(data24[0])):
#     data_stru = data_strue(data24[0],"24")
#     data_24_strue.append(data_stru)
# data_24_strue = np.array(data_24_strue)

labels_36 = labels[~np.isnan(labels[:,4])]
data36 = data_adj[~np.isnan(labels[:,4])]
series36 = data_series[~np.isnan(labels[:,4])]

# data_36_strue = []
# for i in range(len(data36[0])):
#     data_stru = data_strue(data36[0],"36")
#     data_36_strue.append(data_stru)
# data_36_strue = np.array(data_36_strue)
#
labels_48 = labels[~np.isnan(labels[:,5])]
data48 = data_adj[~np.isnan(labels[:,5])]
series48 = data_series[~np.isnan(labels[:,5])]

# data_48_strue = []
# for i in range(len(data48[0])):
#     data_stru = data_strue(data48[0],"48")
#     data_48_strue.append(data_stru)
# data_48_strue = np.array(data_48_strue)

# np.save(filename_adj_data_all, data_adj)
# np.save(filename_labels_all, labels)
# np.save(filename_adj_data_24,data_24_strue)
# np.save(filename_labels_24, labels_24)
# np.save(filename_adj_data_36, data_36_strue)
# np.save(filename_labels_36, labels_36)
# np.save(filename_adj_data_48, data_48_strue)
# np.save(filename_labels_48, labels_48)

np.save(filename_adj_data_all, data_adj)
np.save(filename_labels_all, labels)
np.save(filename_adj_data_24,data24)
np.save(filename_series_data_24,series24)
np.save(filename_labels_24, labels_24)

np.save(filename_adj_data_36, data36)
np.save(filename_series_data_36,series36)
np.save(filename_labels_36, labels_36)

np.save(filename_adj_data_48, data48)
np.save(filename_series_data_48, series48)
np.save(filename_labels_48, labels_48)
# jj = 5
# total_fold = 5
# train_index = []
# test_index = []
# kf = KFold(n_splits=total_fold, shuffle=True, random_state=True)
# for j in range(1, jj + 1):
#     i = 0
#     res = []  # 每次五折的均值
#     for train_i, test_i in kf.split(data_adj):
#         i = i+1
#         train_index.append(train_i)
#         test_index.append(test_i)
#         train_data_adj = data_adj[train_i,:,:,:]
#         #train_data_series= data_series[train_i,:,:,:]
#
#         test_data_adj = data_adj[test_i,:,:,:]
#         test_data_series = data_series[test_i,:,:,:]
#
#         train_label = labels[train_i]
#         test_label = labels[test_i]
#
#         filename_train_adj_data = "../dataset/train_test/train_data_adj_"+str(j) + '_' + str(i) + '.npy'
#         filename_test_adj_data = "../dataset/train_test/test_data_adj_"+str(j) + '_' + str(i) + '.npy'
#         #
#         # filename_train_series_data = "../dataset/train_test/train_data_series_"+str(j) + '_' + str(i) + '.npy'
#         # filename_test_series_data = "../dataset/train_test/test_data_series_"+str(j) + '_' + str(i) + '.npy'
#
#         filename_train_labels = "../dataset/train_test/train_label_"+str(j) + '_' + str(i) + '.npy'
#         filename_test_labels = "../dataset/train_test/test_label_"+str(j) + '_' + str(i) + '.npy'
#
#         np.save(filename_train_adj_data, train_data_adj)
#         np.save(filename_test_adj_data, test_data_adj)
#
#         # np.save(filename_train_series_data, train_data_series)
#         # np.save(filename_test_series_data, test_data_series)
#
#         np.save(filename_train_labels, train_label)
#         np.save(filename_test_labels, test_label)