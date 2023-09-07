import torch
import numpy as np
from model import my_model

# data = np.load('./data/635_8A.npy')
# data = np.load('./data/MDD_HC_427_18A.npy')
data = np.load('./data/BP_HC_392_18A.npy')
print(data.shape)

device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

# Model = my_model(1.0, 1.0)
# Model = torch.load('/home/user/JinZhiYong/Brain/model_checkpoint/ABIDE_6_4/epochs20-2023-06-04-21:55:28.pth')
# Model = torch.load('/home/fennel/Brain/model_checkpoint/MDD/MDD-epochs20-a1.0-b1.0-25.02-2023-07-20-12:55:56.pth')
Model = torch.load('/home/fennel/Brain/model_checkpoint/BD/BP-epochs20-a1.0-b1.0-24.98-2023-07-20-12:59:42.pth')
# Model.to(device)

# A=[]

# for i in range(100, 635, 100):
#     data_batch = torch.from_numpy(data[i-100:i]).float().to(device)
#     print(data_batch.shape)
#     a = Model.Encoder(data_batch)
#     a.to("cpu")
#     A.append(a)

# A = torch.tensor(A)
data = torch.from_numpy(data[:]).float().to(device)
A = Model.Encoder(data)
print(A.shape)

A = A.to("cpu").detach().numpy()

score = []

for a in A:
    l = []
    for i in range(116):
        l.append(a[:,i,:].mean())
    score.append(l)

score = np.array(score)
print(score.shape)

np.save("rankBD.npy", score)

# data_df = pd.DataFrame(score)
# writer = pd.ExcelWriter('rank.xlsx')
# data_df.to_excel(writer,'page_1',float_format='%.5f')
# writer.save()