import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
plt.style.use('seaborn-whitegrid')
palette = pyplot.get_cmap('Set1')
font1 = {
'weight' : 'normal',
'size'   : 20,
}

data = np.load('./data/s.npy')
data2 = np.load('./data/d.npy')
data0 = np.load('./data/0.npy')

print(data.shape)
print(data2.shape)
print(data0.shape)

x = [i for i in range(1, 9)]
# plt.plot(x, data2[0], label='s', color = 'r', linewidth=0.5)
# plt.plot(x, data[0], label='d', color = 'g', linewidth=0.5)
# plt.savefig('2.png')

# x = [i for i in range(1, 201)]
y1 = data[0].reshape(200, 8)
y2 = data2[0].reshape(200, 8)
# y3 = data0[0].reshape(200, 8)

# plt.plot(x, y2, label='d', color = 'g', linewidth=0.5)
# plt.plot(x, y1, label='s', color = 'r', linewidth=0.5)
# plt.savefig('1.png')

# plt.figure(figsize=(12,10))

# def draw_line(name_of_alg,color_index,datas):
#     color=palette(color_index)
#     avg=np.mean(datas,axis=0)
#     std=np.std(datas,axis=0)
#     r1 = list(map(lambda x: x[0]-x[1], zip(avg, std)))#上方差
#     r2 = list(map(lambda x: x[0]+x[1], zip(avg, std)))#下方差
#     plt.plot(x, avg, color=color,label=name_of_alg,linewidth=3)
#     plt.fill_between(x, r1, r2, color=color, alpha=0.2)

# draw_line("s",1,y1)
# draw_line("d",2,y2)

# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlabel('time steps',fontsize=20)
# plt.ylabel('features',fontsize=20)
# plt.legend(loc='upper left',prop=font1)

# plt.savefig('3.png')

plt.figure(figsize=(12,10))

box_plot1 = plt.boxplot(y1, patch_artist=True)
box_plot2 = plt.boxplot(y2, patch_artist=True)

color_blue = (89/255, 141/255, 191/255, 1.0)
color_green = (112/255, 189/255, 122/255, 1.0)

for box in box_plot1['boxes']:
    box.set(facecolor=color_blue)
for box in box_plot2['boxes']:
    box.set(facecolor=color_green)

boxes = [box_plot1["boxes"][0], box_plot2["boxes"][0]]
labels = ["s", "d"]

# 添加图例
plt.legend(boxes, labels, loc='upper right', prop=font1)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('time steps',fontsize=20)
plt.ylabel('features',fontsize=20)
# plt.legend(loc='upper left', prop=font1)

plt.savefig('4.png')