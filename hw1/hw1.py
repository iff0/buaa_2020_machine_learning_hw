# %%

import os
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from cnn import Cnn
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image as pil
color_b = sns.color_palette("Blues_r")
# Seaborn's palette for drawing charts
img_saving = False
# (DEBUG) whether to save the drawn picture..

# %%

if torch.cuda.is_available():
    print('Now using:', torch.cuda.get_device_name(0))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
print('Device name:', device)

# %%

class Loader():
    def __init__(self, data: np.ndarray, bsz: int, label=None):
        self.data = data
        self.bsz = bsz
        self.p = 0
        self.label = label

    def __iter__(self):
        self.p = 0
        return self

    def __next__(self):
        if self.p >= self.data.shape[0]:
            raise StopIteration
        else:
            pp = min(self.data.shape[0], self.p + self.bsz)
            r1 = self.data[self.p: pp]
            if self.label is None:
                self.p = pp
                return r1
            else:
                r2 = self.label[self.p: pp]
                self.p = pp
                return (r1, r2)

    def __len__(self):
        return self.data.shape[0]

# %%

catyDict = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprise': 6
}

train_x = np.zeros((0, 48, 48))
train_y = np.zeros((0, 1))
train_path = 'train'
caty_ns = []
for c in catyDict:
    asy = os.listdir(os.path.join(train_path, c))
    maty = np.zeros((len(asy), 48, 48))
    for (ind, i) in enumerate(asy):
        x = np.asarray(pil.open(os.path.join(train_path, c, i))).copy()
        if x.std() > 1:
            x = (x - x.mean()) / x.std()
        maty[ind, :, :] = x
    train_x = np.vstack((train_x, maty))
    train_y = np.vstack((train_y, np.full((maty.shape[0], 1), catyDict[c])))
    caty_ns.append(maty.shape[0])
print(train_x.shape)
print(train_y.shape)
plt.figure(figsize=(6, 6))
plt.pie(caty_ns, colors=color_b[0: 7],
        labels=[c for c in catyDict.keys()],
        autopct='%.2f%%')
plt.title('Data labeling pie chart')
if img_saving:
    plt.savefig('1_1.png', bbox_inches='tight')
# plt.show()
# print(apic.size)

# %%

train_x = np.expand_dims(train_x, axis=1)
print(train_x[0])
plt.imshow(train_x[0][0])
per = np.random.permutation(train_x.shape[0])  # 打乱后的行号
rtrain_x = torch.from_numpy(train_x[per])
rtrain_y = torch.from_numpy(train_y[per]).squeeze()
print(per)
plt.figure()
plt.imshow(rtrain_x[0][0])

# %%

cnn = Cnn()
cnn.to(device)
print(cnn)
dummy_input = torch.randn(1, 1, 48, 48, device=device)
torch.onnx.export(cnn, dummy_input, "convnet.onnx", verbose=True, input_names=['input'], output_names=['output'])

# %%

learn_rate = 3e-4
los = nn.CrossEntropyLoss()
loader = Loader(rtrain_x, 64, label=rtrain_y)
optim = torch.optim.Adam(cnn.parameters(), lr=learn_rate)
for ep in range(24):
    print('epoches: %d' % ep)
    for i, (x, y) in enumerate(loader):
        optim.zero_grad()
        x = x.to(device, dtype=torch.float)
        # print(x.shape)
        y = y.to(device, dtype=torch.long)
        # print(y)
        outt = cnn(x)
        loss = los(outt, y)
        # print(loss)
        loss.backward()
        optim.step()
        if (i + 1) % (int(len(loader) / loader.bsz / 100)) == 0:
            print('\r', '%d %%' % ((i + 1) / (len(loader) / loader.bsz) * 100), end="", flush=True)

    print('')
    # acc = torch.Tensor([0.]).squeeze().to(device)
    # for i, (x, y) in enumerate(loader):
    #   o = cnn(x.to(device, dtype=torch.float))
    #  add = torch.sum(torch.argmax(o, 1) == y.to(device, dtype=torch.long)).to(torch.float64)
    # acc += add
    # print('accuary', acc.cpu().numpy() / len(loader) * 100, '%')
acc = torch.Tensor([0.]).squeeze().to(device)
torch.cuda.empty_cache()
# Clear cuda cache
for i, (x, y) in enumerate(loader):
    o = cnn(x.to(device, dtype=torch.float))
    add = torch.sum(torch.argmax(o, 1) == y.to(device, dtype=torch.long)).to(torch.float64)
    acc += add

print('accuary', acc.cpu().numpy() / len(loader) * 100, '%')
torch.save(cnn.state_dict(), 'convnet.pkl')
# dummy_input = torch.randn(1, 1, 48, 48, device=device)
# torch.onnx.export(cnn, dummy_input, "convnet.onnx", verbose=True, input_names=['input'], output_names=['output'])
caty_dict_ = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}
cnn = Cnn()
cnn.load_state_dict(torch.load('convnet.pkl'))
print(cnn)
test_dic = 'test'
df = pd.read_csv('submission.csv')
print(df.head())

for i in range(df.shape[0]):

    s = df['file_name'][i]
    x = np.asarray(pil.open(os.path.join(test_dic, s)))
    x = (x - x.mean()) / x.std()
    x = np.expand_dims(np.expand_dims(x, axis=0), axis=0)
    x = torch.from_numpy(x).to('cpu', dtype=torch.float)
    df['class'][i] = caty_dict_[torch.argmax(cnn(x), 1).numpy()[0]]
    if i % 1000 == 0:
        print(i, s)
print(df['class'].value_counts())
df.to_csv('submission.csv', index=False)
