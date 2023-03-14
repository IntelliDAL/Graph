from torch.utils.data import Dataset





class datasets(Dataset):
    def __init__(self, adj, label):
        self.adj_all = adj
        self.labels = label


    def __getitem__(self, idx):

        adj = self.adj_all[idx]
        return_dic = {'adj': adj,
                      'label': self.labels[idx]
                      }

        return return_dic

    def __len__(self):
        return len(self.labels)
class dataset_pro(Dataset):
    def __init__(self, adj, label):
        self.adj_all = adj
        self.labels = label

    def __getitem__(self, idx):

        adj = self.adj_all[idx]
        label = self.labels[idx]
        return adj, label

    def __len__(self):
        return len(self.labels)
class datasets4(Dataset):
    def __init__(self, adj, adj1, label):
        self.adj_all = adj
        self.adj_all1 = adj1
        self.labels = label


    def __getitem__(self, idx):

        adj = self.adj_all[idx]
        adj1 = self.adj_all1[idx]
        return_dic = {'adj': adj,
                      'adj1': adj1,
                      'label': self.labels[idx]
                      }

        return return_dic

    def __len__(self):
        return len(self.labels)

class datasets2(Dataset):
    def __init__(self, adj, adj1, adj2, label):
        self.adj_all = adj
        self.labels = label
        self.adj_all1 = adj1
        self.adj_all2 = adj2


    def __getitem__(self, idx):

        adj = self.adj_all[idx]
        adj1 = self.adj_all1[idx]
        adj2 = self.adj_all2[idx]
        return_dic = {'adj': adj,
                      'adj1': adj1,
                      'adj2': adj2,
                      'label': self.labels[idx]
                      }

        return return_dic

    def __len__(self):
        return len(self.labels)

class datasets2_1(Dataset):
    def __init__(self, adj, adj1, adj2, adj3,label):
        self.adj_all = adj
        self.labels = label
        self.adj_all1 = adj1
        self.adj_all2 = adj2
        self.adj_all3 = adj3


    def __getitem__(self, idx):

        adj = self.adj_all[idx]
        adj1 = self.adj_all1[idx]
        adj2 = self.adj_all2[idx]
        adj3 = self.adj_all3[idx]
        return_dic = {'adj': adj,
                      'adj1': adj1,
                      'adj2': adj2,
                      'adj3': adj3,
                      'label': self.labels[idx]
                      }

        return return_dic

    def __len__(self):
        return len(self.labels)


class datasets3(Dataset):
    def __init__(self, adj, adj1, adj2, adj3, adj4, adj5, label):
        self.adj_all = adj
        self.labels = label
        self.adj_all1 = adj1
        self.adj_all2 = adj2
        self.adj_all3 = adj3
        self.adj_all4 = adj4
        self.adj_all5 = adj5


    def __getitem__(self, idx):

        adj = self.adj_all[idx]
        adj1 = self.adj_all1[idx]
        adj2 = self.adj_all2[idx]
        adj3 = self.adj_all3[idx]
        adj4 = self.adj_all4[idx]
        adj5 = self.adj_all5[idx]
        return_dic = {'adj': adj,
                      'adj1': adj1,
                      'adj2': adj2,
                      'adj3': adj3,
                      'adj4': adj4,
                      'adj5': adj5,
                      'label': self.labels[idx]
                      }

        return return_dic

    def __len__(self):
        return len(self.labels)

class datasets1(Dataset):
    def __init__(self, adj, label, timeseries):
        self.adj_all = adj
        self.labels = label
        self.timeseries = timeseries

    def __getitem__(self, idx):

        adj = self.adj_all[idx]
        return_dic = {'adj': adj,
                      'label': self.labels[idx],
                      'timeseries':self.timeseries[idx]
                      }

        return return_dic

    def __len__(self):
        return len(self.labels)

