import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
print(len(os.listdir(r"C:\CNN\People_Clothing_Segmentation\IMAGES")))
print(len(os.listdir(r"C:\CNN\People_Clothing_Segmentation\MASKS")))

# 建立一個 discrete_map 把連續的顏色漸層切成 N 個區間
# 參考自 https://gist.github.com/jakevdp/91077b0cae40f8f8244a

def discrete_cmap(N, base_cmap=None):
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)
"""
# 將圖畫出來看看
img = cv.imread("People Clothing Segmentation/IMAGES/img_0009.png")
mask = cv.imread("People Clothing Segmentation\MASKS\seg_0009.jpeg")
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
mask_gray = mask[:, :, 0]
plt.figure(figsize=(6, 6))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(mask_gray, cmap=discrete_cmap(59, "cubehelix"))
plt.axis('off')
plt.show()
"""
label_rule = pd.read_csv("People_Clothing_Segmentation/labels.csv")
print(label_rule)
mapping_rule = label_rule.to_dict()["label_list"]
mapping_rule[0] = "background"
from tqdm import tqdm
IMG_Path = []
for img_name in tqdm(os.listdir(r"C:\CNN\People_Clothing_Segmentation\IMAGES")):
    IMG_Path.append(os.path.join(r"C:\CNN\People_Clothing_Segmentation\IMAGES", img_name))


Mask_Path = []
for mask_name in tqdm(os.listdir(r"C:\CNN\People_Clothing_Segmentation\MASKS")):
    Mask_Path.append(os.path.join(r"C:\CNN\People_Clothing_Segmentation\MASKS", mask_name))


# IMGS = []
# for path in tqdm(IMG_Path):
#     img = cv.imread(path)
#     if img is None:
#         print(f"⚠️ 圖片讀取失敗：{path}")
#         continue

# 讀取 IMAGES 和 MASKS 的圖片
IMGS = []
for path in tqdm(IMG_Path):
    img = cv.imread(path)
    img = cv.resize(img, (256, 256))
    img = img/255
    IMGS.append(img)
    img_flip = cv.flip(img, flipCode=1)
    IMGS.append(img_flip)
IMGS = np.array(IMGS)

# 讀取 MASKS 的圖片
MASKS = []
for path in tqdm(Mask_Path):
    mask = cv.imread(path)
    mask = mask[:, :, 0]
    mask = cv.resize(mask, (256, 256), interpolation=cv.INTER_NEAREST)
    MASKS.append(mask)
    mask_flip = cv.flip(mask, flipCode=1)
    MASKS.append(mask_flip)
MASKS = np.array(MASKS).astype("int32")
# print(IMGS.shape)
# print(MASKS.shape)

# 分割訓練集和測試集
Train_IMGS = IMGS[100:900]  # 中間1800張作為訓練集
Test_IMGS = np.concatenate([IMGS[:100], IMGS[900:]])  # 前100張和剩餘的作為測試集
Train_MASKS = MASKS[100:900]  # 中間1800張遮罩作為訓練集
Test_MASKS = np.concatenate([MASKS[:100], MASKS[900:]])  # 前100張和剩餘的遮罩作為測試集
print("最小類別：", np.min(Train_MASKS))
print("最大類別：", np.max(Train_MASKS))

Train_MASKS[Train_MASKS >= 59] = 0
Test_MASKS[Test_MASKS >= 59] = 0
print("最小類別：", np.min(Train_MASKS))
print("最大類別：", np.max(Train_MASKS))


# print(Train_IMGS.shape)
# print(Test_IMGS.shape)
# print(Train_MASKS.shape, Train_MASKS.dtype)
# print(Test_MASKS.shape)
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
# Encoder
inputs = Input((256, 256, 3))
conv1 = Conv2D(64, 3, activation = "relu", padding = "same", kernel_regularizer = l2(0.01), kernel_initializer = "he_normal")(inputs)
conv1 = Conv2D(64, 3, activation = "relu", padding = "same", kernel_regularizer = l2(0.01), kernel_initializer = "he_normal")(conv1)
conv1 = Dropout(0.5)(conv1)
pool1 = MaxPooling2D(pool_size = (2, 2))(conv1)
conv2 = Conv2D(128, 3, activation = "relu", padding = "same", kernel_regularizer = l2(0.01), kernel_initializer = "he_normal")(pool1)
conv2 = Conv2D(128, 3, activation = "relu", padding = "same", kernel_regularizer = l2(0.01), kernel_initializer = "he_normal")(conv2)
conv2 = Dropout(0.5)(conv2)
pool2 = MaxPooling2D(pool_size = (2, 2))(conv2)
conv3 = Conv2D(256, 3, activation = "relu", padding = "same", kernel_regularizer = l2(0.01), kernel_initializer = "he_normal")(pool2)
conv3 = Conv2D(256, 3, activation = "relu", padding = "same", kernel_regularizer = l2(0.01), kernel_initializer = "he_normal")(conv3)
conv3 = Dropout(0.5)(conv3)
pool3 = MaxPooling2D(pool_size = (2, 2))(conv3)
conv4 = Conv2D(512, 3, activation = "relu", padding = "same", kernel_regularizer = l2(0.01), kernel_initializer = "he_normal")(pool3)
conv4 = Conv2D(512, 3, activation = "relu", padding = "same", kernel_regularizer = l2(0.01), kernel_initializer = "he_normal")(conv4)
conv4 = Dropout(0.5)(conv4)
pool4 = MaxPooling2D(pool_size = (2, 2))(conv4)
conv5 = Conv2D(1024, 3, activation = "relu", padding = "same", kernel_regularizer = l2(0.01), kernel_initializer = "he_normal")(pool4)
conv5 = Conv2D(1024, 3, activation = "relu", padding = "same", kernel_regularizer = l2(0.01), kernel_initializer = "he_normal")(conv5)
conv5 = Dropout(0.5)(conv5)

# Decoder
up6 = Conv2D(512, 2, activation = "relu", padding = "same", kernel_regularizer = l2(0.01), kernel_initializer = "he_normal")(UpSampling2D(size = (2, 2))(conv5))
merge6 = concatenate([conv4, up6])
conv6 = Conv2D(512, 3, activation = "relu", padding = "same", kernel_regularizer = l2(0.01), kernel_initializer = "he_normal")(merge6)
conv6 = Conv2D(512, 3, activation = "relu", padding = "same", kernel_regularizer = l2(0.01), kernel_initializer = "he_normal")(conv6)

up7 = Conv2D(256, 2, activation = "relu", padding = "same", kernel_regularizer = l2(0.01), kernel_initializer = "he_normal")(UpSampling2D(size = (2, 2))(conv6))
merge7 = concatenate([conv3, up7])
conv7 = Conv2D(256, 3, activation = "relu", padding = "same", kernel_regularizer = l2(0.01), kernel_initializer = "he_normal")(merge7)
conv7 = Conv2D(256, 3, activation = "relu", padding = "same", kernel_regularizer = l2(0.01), kernel_initializer = "he_normal")(conv7)

up8 = Conv2D(128, 2, activation = "relu", padding = "same", kernel_regularizer = l2(0.01), kernel_initializer = "he_normal")(UpSampling2D(size = (2, 2))(conv7))
merge8 = concatenate([conv2, up8])
conv8 = Conv2D(128, 3, activation = "relu", padding = "same", kernel_regularizer = l2(0.01), kernel_initializer = "he_normal")(merge8)
conv8 = Conv2D(128, 3, activation = "relu", padding = "same", kernel_regularizer = l2(0.01), kernel_initializer = "he_normal")(conv8)

up9 = Conv2D(64, 2, activation = "relu", padding = "same", kernel_regularizer = l2(0.01), kernel_initializer = "he_normal")(UpSampling2D(size = (2, 2))(conv8))
merge9 = concatenate([conv1, up9])
conv9 = Conv2D(64, 3, activation = "relu", padding = "same", kernel_regularizer = l2(0.01), kernel_initializer = "he_normal")(merge9)
conv9 = Conv2D(64, 3, activation = "relu", padding = "same", kernel_regularizer = l2(0.01), kernel_initializer = "he_normal")(conv9)

conv9 = Conv2D(59, 1, activation = "softmax", padding = "same", kernel_initializer = "he_normal")(conv9)

UNET = Model(inputs = inputs, outputs = conv9)
UNET.summary()

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler

def scheduler(epoch, lr):
    if epoch < 20:
        return lr
    else:
        return lr * tf.math.exp(-0.1).numpy()
optimizer = Adam(learning_rate = 0.0001)
lr_scheduler = LearningRateScheduler(scheduler)

UNET.compile(optimizer = optimizer, loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

History = UNET.fit(
    x = Train_IMGS,
    y = Train_MASKS,
    batch_size = 16,
    epochs = 35,
    callbacks = [lr_scheduler],
    validation_data = (Test_IMGS, Test_MASKS)
)

import matplotlib.pyplot as plt
plt.figure(figsize = (10, 5))
plt.plot(History.history["loss"], label = "train_loss")
plt.plot(History.history["val_loss"], label = "validation_loss")
plt.title("Loss", fontsize = 20)
plt.xlabel("Epoch", fontsize = 14)
plt.ylabel("Loss", fontsize = 14)
plt.legend()
plt.show()

plt.figure(figsize = (10, 5))
plt.plot(History.history["accuracy"], label = "train_accuracy")
plt.plot(History.history["val_accuracy"], label = "validation_accuracy")
plt.title("Accuracy", fontsize = 20)
plt.xlabel("Epoch", fontsize = 14)
plt.ylabel("Accuracy", fontsize = 14)
plt.legend()
plt.show()