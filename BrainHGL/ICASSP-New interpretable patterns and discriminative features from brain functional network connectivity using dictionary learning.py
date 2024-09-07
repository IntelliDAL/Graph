import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import FastICA
import numpy as np

# X = np.load('./MDD_HC_sig.npz.npy')
# Y = np.load('./MDD_HC_sig_label.npy')
bold_signals = np.load('./MDD_HC_sig.npz.npy')
labels = np.load('./MDD_HC_sig_label.npy')
# bold_signals = bold_signal.reshape
# bold_signals.shape = (num_subjects, num_regions, num_timepoints)
# labels.shape = (num_subjects,)


def perform_group_ica(bold_signals, n_components=50):
    num_subjects, num_timepoints, num_regions = bold_signals.shape
    bold_signals_reshaped = bold_signals.reshape(num_subjects * num_timepoints, num_regions)
    
    ica = FastICA(n_components=n_components, max_iter=500)
    components = ica.fit_transform(bold_signals_reshaped)
    components = components.reshape(num_subjects, num_timepoints, n_components)
    return components

def extract_fnc_features(components):
    num_subjects, num_timepoints, n_components = components.shape
    fnc_features = np.zeros((num_subjects, n_components * (n_components - 1) // 2))
    
    for i in range(num_subjects):
        tc = components[i]
        corr_matrix = np.corrcoef(tc.T)
        corr_matrix = np.nan_to_num(corr_matrix)
        upper_triangle_indices = np.triu_indices(n_components, 1)
        fnc_features[i] = corr_matrix[upper_triangle_indices]
    
    return fnc_features

n_components = 50
components = perform_group_ica(bold_signals, n_components)
fnc_features = extract_fnc_features(components)

scaler = StandardScaler()
fnc_features = scaler.fit_transform(fnc_features)
fnc_features = torch.tensor(fnc_features, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(fnc_features, labels, test_size=0.2, random_state=42)

class DictionaryLearningWithClassifier(nn.Module):
    def __init__(self, feature_dim, num_atoms):
        super(DictionaryLearningWithClassifier, self).__init__()
        self.dictionary = nn.Parameter(torch.randn(feature_dim, num_atoms))
        nn.init.xavier_uniform_(self.dictionary)  # 
        self.classifier = nn.Parameter(torch.randn(2, num_atoms))
        nn.init.xavier_uniform_(self.classifier)  
        self.sparse_codes = None

    def forward(self, X, sparsity, labels=None, beta=0.05):
        self.sparse_codes = torch.zeros(X.shape[0], self.dictionary.shape[1])
        for i in range(X.shape[0]):
            residual = X[i].clone()
            for _ in range(sparsity):
                projections = torch.matmul(residual, self.dictionary)
                best_atom = torch.argmax(torch.abs(projections))
                self.sparse_codes[i, best_atom] += projections[best_atom].item()
                residual -= projections[best_atom] * self.dictionary[:, best_atom]
        
        recon = torch.matmul(self.sparse_codes, self.dictionary.T)
        
        if labels is not None:
            predictions = torch.matmul(self.sparse_codes, self.classifier.T)
            classification_loss = nn.CrossEntropyLoss()(predictions, labels)
            reconstruction_loss = nn.MSELoss()(recon, X)
            loss = reconstruction_loss + beta * classification_loss
            return recon, predictions, loss
        
        return recon

    def get_sparse_codes(self):
        return self.sparse_codes

def check_gradients_for_nan(model):
    for param in model.parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print("Warning: NaN gradients found.")
            return True
    return False

def train_model(X_train, y_train, feature_dim, num_atoms, sparsity, num_epochs, learning_rate, beta):
    model = DictionaryLearningWithClassifier(feature_dim, num_atoms)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        recon, predictions, loss = model(X_train, sparsity, y_train, beta)
        loss.backward()

        # nan
        if check_gradients_for_nan(model):
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: NaN detected, stopping training.")
            break

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
    return model

# num_atoms = 100  # 字典原子数量 100
# sparsity = 3    # 稀疏度 10
# num_epochs = 100  # 训练轮数 100
learning_rate = 0.001  # 降低学习率 0.001
beta = 0.05  # 分类损失权重 0.05



def TrainModel(num_atoms, sparsity, num_epochs):
    model = train_model(X_train, y_train, X_train.shape[1], num_atoms, sparsity, num_epochs, learning_rate, beta)

    model.eval()
    with torch.no_grad():
        sparse_codes_train = model.get_sparse_codes().detach().numpy()


    svm = SVC(kernel='linear')
    svm.fit(sparse_codes_train, y_train.numpy())

    y_train_pred = svm.predict(sparse_codes_train)
    train_accuracy = accuracy_score(y_train.numpy(), y_train_pred)
    print(f'Classification Accuracy on Training Data: {train_accuracy:.4f}')

    with torch.no_grad():
        model(X_test, sparsity)  
        sparse_codes_test = model.get_sparse_codes().detach().numpy()

    y_test_pred = svm.predict(sparse_codes_test)
    test_accuracy = accuracy_score(y_test.numpy(), y_test_pred)
    print(f'Classification Accuracy on Test Data: {test_accuracy:.4f}')

for num_atoms in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]:
    for sparsity in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for num_epochs in [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]:
            print('原子数量：', num_atoms, '稀疏度：', sparsity, '训练轮数：', num_epochs)
            TrainModel(num_atoms, sparsity, num_epochs)

