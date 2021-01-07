#%%

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image as pil
import tensorflow.keras.layers as lays
#%%

caty_dict = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprise': 6
}
caty_dict_ = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

train_x = np.zeros((0, 48, 48))
train_y = np.zeros((0, 1))
train_path = 'train'
caty_ns = []
for c in caty_dict:
    asy = os.listdir(os.path.join(train_path, c))
    maty = np.zeros((len(asy), 48, 48))
    for (ind, i) in enumerate(asy):
        x= np.asarray(pil.open(os.path.join(train_path, c, i))).copy()
        maty[ind, :, :] = x/255.
    train_x = np.vstack((train_x, maty))
    train_y = np.vstack((train_y, np.full((maty.shape[0], 1), caty_dict[c])))
    caty_ns.append(maty.shape[0])
print(train_x.shape)
print(train_y.shape)
#plt.pie(caty_ns)
#plt.show()
idx = [i for i in range(train_x.shape[0])]
np.random.shuffle(idx)
train_x = train_x[idx]
train_y = train_y[idx]

train_x = train_x.reshape(train_x.shape[0], 48, 48, 1)


plt.imshow(train_x[20, :, :, 0], cmap='gray')
print(train_x[20])
plt.show()

model = tf.keras.models.Sequential([
    lays.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding='same',
    ),
    lays.BatchNormalization(),
    lays.LeakyReLU(),
    lays.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding='same',
    ),
    lays.BatchNormalization(),
    lays.LeakyReLU(),
    lays.MaxPool2D(strides=2),
    lays.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding='same'
    ),
    lays.BatchNormalization(),
    lays.LeakyReLU(),
    lays.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding='same'
    ),
    lays.BatchNormalization(),
    lays.LeakyReLU(),
    lays.MaxPool2D(strides=2),
    lays.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding='same'
    ),
    lays.BatchNormalization(),
    lays.LeakyReLU(),
    lays.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding='same'
    ),
    lays.BatchNormalization(),
    lays.LeakyReLU(),
    lays.MaxPool2D(strides=2),

    lays.Conv2D(
        filters=256,
        kernel_size=(3, 3),
        padding='same'
    ),
    lays.BatchNormalization(),
    lays.LeakyReLU(),
    lays.Conv2D(
        filters=256,
        kernel_size=(3, 3),
        padding='same'
    ),
    lays.BatchNormalization(),
    lays.ReLU(),

    lays.Flatten(),

    #lays.Dense(units=512, activation=tf.nn.relu),
    lays.Dropout(0.5),
    lays.Dense(units=1024, activation=tf.nn.relu),
    lays.Dropout(0.25),
    lays.Dense(units=7),
    lays.Softmax()
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=['acc']
)
model.fit(train_x, train_y, epochs=16, batch_size=96)

model.save_weights('my_weights.h5')
print(model.evaluate(train_x, train_y))


test_dic = 'test'
df = pd.read_csv('submission.csv')
print(df.head())

for i in range(df.shape[0]):
    s = df['file_name'][i]
    x = np.asarray(pil.open(os.path.join(test_dic, s)))
    x = x/255.
    x = x.reshape(1, 48, 48, 1)
    #print(np.argmax(model(x), axis=-1))
    df['class'][i] = caty_dict_[np.argmax(model(x), axis=-1)[0]]
    if i % 1000 == 0:
        print(i, s)
print(df['class'].value_counts())
df.to_csv('submission.csv', index=False)
#print(apic.size)

