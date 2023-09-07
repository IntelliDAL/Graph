import numpy as np

# X = np.load('./data/all_data_ABIDE.npy')
# print(X.shape)

# A = np.empty([631, 8, 116, 116])

# for k,x in enumerate(X):
#     if k%10==0:
#         print(k)
#     for n, left in enumerate(range(0, 71, 10)):
#         for i in range(116):
#             for j in range(i, 116):
#                 if i==j:
#                     A[k][n][i][j] = 1
#                 else:
#                     A[k][n][i][j] = np.corrcoef(x[i,left:left+100], x[j,left:left+100])[0][1]
#                     A[k][n][j][i] = A[k][n][i][j]

# np.save('./data/ABIDE_aal_8A', A)

import math
new_A = []

A = np.load('./data/ABIDE_aal_8A.npy')
Y = np.load('./data/all_label_ABIDE.npy')
Y620 = []
print(A.shape)
for i, a in enumerate(A):
    if not math.isnan(a.mean()):
        new_A.append(a)
        Y620.append(Y[i])

new_A = np.array(new_A)
Y620 = np.array(Y620)
print(new_A.shape, Y620.shape)
np.save('./data/ABIDE_aal_620_8A.npy', new_A)
np.save('./data/ABIDE_aal_620y.npy', Y620)