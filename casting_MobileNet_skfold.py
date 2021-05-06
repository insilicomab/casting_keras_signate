# -*- coding: utf-8 -*-

'''
データの読み込みと確認
'''

# ライブラリのインポート
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io
import os
import zipfile
import io

from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
import keras
from keras.applications.mobilenet import MobileNet
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

# ランダムシードの設定
import random
np.random.seed(1234)
random.seed(1234)
tf.random.set_seed(1234)

# 入力と出力を指定
photo_size = 300
im_rows = 300 # 画像の縦ピクセルサイズ
im_cols = 300 # 画像の横ピクセルサイズ
im_color = 3 # 画像の色空間
num_classes = 2 # クラス数の定義

# ラベルデータの読み込みと確認
train_labels = pd.read_csv('./data/train.csv', sep=',')
print(train_labels.head())

# zipファイルのパス
zip_path = './data/train_data.zip'

# 画像データの格納リスト
X_train = [] # 画像のピクセル値とラベルを格納するリストを生成（説明変数）

# zipの読み込み
with zipfile.ZipFile(zip_path, 'r') as zip_file:
    for i in train_labels['id']:
        with zip_file.open('train_data/'+i) as img_file:
            # 画像のバイナリデータを読み込む
            img_bin = io.BytesIO(img_file.read())
            # バイナリデータをpillowで開く
            img = Image.open(img_bin)
            # 画像データを配列化
            img_array = np.array(img)
            # 格納リストに追加
            X_train.append(img_array)

# 画像の確認
for i in range(10):
    plt.imshow(X_train[i])
    plt.title(train_labels.iloc[i, 1])
    plt.show()
    
# np.arrayに変換
X_train = np.array(X_train)

'''
特徴量エンジニアリング
'''

# 読み込んだデータを四次元配列に変換後、正規化
train_filenumber = len(X_train)

X_train = X_train.reshape(train_filenumber, im_rows, im_cols, im_color)
X_train = X_train.astype('float32') / 255

# 画像の前処理としての正規化
def normalization(X_train):
    X_train = X_train.reshape(train_filenumber, im_color*im_rows*im_cols)
    
    for filenum in range(0, train_filenumber):
        X_train[filenum] -= np.mean(X_train[filenum])

    X_train = X_train.reshape(train_filenumber, im_rows, im_cols, im_color)

    return X_train

normalization(X_train)

# ラベルデータをNumpy配列に変換
Y_train = np.array(train_labels['target'])

'''
Fine-tuning
'''

# K分割する
folds = 5
skf = StratifiedKFold(n_splits=folds, shuffle=True)
index = 0

for train_index, val_index in skf.split(X_train, Y_train):
    x_train = X_train[train_index]
    x_valid = X_train[val_index]
    y_train = Y_train[train_index]
    y_valid = Y_train[val_index]

    # ラベルデータのOne-hot encoding
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_valid = keras.utils.to_categorical(y_valid, num_classes)

    # モデルを保存するファイルパス
    filepath = './model/casting_MobileNet_finetuning_skfold[%d].h5' % index
    index += 1
    
    # データの水増し
    datagen = ImageDataGenerator(rotation_range=180,                        # 整数．画像をランダムに回転する回転範囲
                                 horizontal_flip=True,                      # 真理値．水平方向に入力をランダムに反転
                                 vertical_flip=True,                        # 真理値．垂直方向に入力をランダムに反転
                                 )
    
    # モデルと学習済みの重みをロード（全結合層は除く）
    input_tensor = Input(shape=(im_cols, im_rows, im_color))
    mobileNet_model = MobileNet(include_top=False, weights='imagenet', input_tensor=input_tensor)
    
    # 全結合層の構築
    top_model = Sequential()
    top_model.add(Flatten(input_shape=mobileNet_model.output_shape[1:]))
    top_model.add(Dense(256))
    top_model.add(Activation("relu"))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(num_classes))
    top_model.add(Activation("softmax"))
    
    # 全結合層を削除したモデルと上で自前で構築した全結合層を結合
    model = Model(mobileNet_model.input, top_model(mobileNet_model.output))
    
    # 水増し画像を訓練用画像の形式に合わせる
    datagen.fit(x_train)
    
    # モデルをコンパイル
    model.compile(loss='binary_crossentropy',
                  optimizer=SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])
    
    # 過学習の抑制
    early_stopping = EarlyStopping(monitor='val_loss', patience=10 , verbose=1)
    
    # 評価に用いるモデル重みデータの保存
    checkpointer = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
    
    # 学習を実行
    hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=4),
                               epochs=50,
                               verbose=1,
                               validation_data=(x_valid, y_valid),
                               callbacks=[early_stopping, checkpointer])  # CallBacksに設定
    
    # モデルを評価
    score = model.evaluate(x_valid, y_valid, verbose=1)
    print('正解率=', score[1], 'loss=', score[0])
    
    '''
    学習過程のグラフ化
    '''
    
    # 正解率の推移をプロット
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('Accuracy')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # ロスの推移をプロット
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Loss')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# 評価に用いるモデル構造の保存
def save_model(model):
    json_string = model.to_json()
    if not os.path.isdir("mobilenet_model"):
        os.mkdir("mobilenet_model")
    json_name = "architecture_MobileNet_skfold.json"
    open(os.path.join("mobilenet_model", json_name),"w").write(json_string)

save_model(model)

'''
テストデータの予測
'''

# ライブラリのインポート
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import zipfile
import io
from keras.models import model_from_json
from scipy import stats

# 入力と出力を指定
photo_size = 300
im_rows = 300 # 画像の縦ピクセルサイズ
im_cols = 300 # 画像の横ピクセルサイズ
im_color = 3 # 画像の色空間
num_classes = 2 # クラス数の定義

# ラベルデータの読み込み
test_labels = pd.read_csv('./data/sample_submission.csv', sep=',', header=None)
print(test_labels.head())

# zipファイルのパス
zip_path = './data/test_data.zip'

# 画像データの格納リスト
X_test = [] # 画像のピクセル値とラベルを格納するリストを生成（説明変数）

# zipの読み込み
with zipfile.ZipFile(zip_path, 'r') as zip_file:
    for i in test_labels[0]:
        with zip_file.open('test_data/'+i) as img_file:
            # 画像のバイナリデータを読み込む
            img_bin = io.BytesIO(img_file.read())
            # バイナリデータをpillowで開く
            img = Image.open(img_bin)
            # 画像データを配列化
            img_array = np.array(img)
            # 格納リストに追加
            X_test.append(img_array)
            
# 画像の確認
for i in range(10):
    plt.imshow(X_test[i])
    plt.title(test_labels.iloc[i, 1])
    plt.show()
    
# np.arrayに変換
X_test = np.array(X_test)

# 読み込んだデータを四次元配列に変換後、正規化
test_filenumber = len(X_test)

X_test = X_test.reshape(test_filenumber, im_rows, im_cols, im_color)
X_test = X_test.astype('float32') / 255

# 画像の前処理としての正規化
def normalization(X_test):
    X_test = X_test.reshape(test_filenumber, im_color*im_rows*im_cols)
    
    for filenum in range(0, test_filenumber):
        X_test[filenum] -= np.mean(X_test[filenum])

    X_test = X_test.reshape(test_filenumber, im_rows, im_cols, im_color)

    return X_test

normalization(X_test)

# 予測データの格納リスト
preds = []

# 保存したモデル重みデータとモデル構造の読み込み
for index in range(0,5):
    filepath = './model/casting_MobileNet_finetuning_skfold[%d].h5' % index
    json_name = 'architecture_MobileNet_skfold.json'
    model = model_from_json(open(os.path.join("mobilenet_model", json_name)).read())
    model.load_weights(filepath)

    # 各モデルにおける推測確率の計算　
    pred = model.predict(X_test)
    pred_max = np.argmax(pred, axis=1)
    preds.append(pred_max)

# アンサンブル学習
preds_array = np.array(preds)
pred = stats.mode(preds_array)[0].T # 予測データリストのうち最頻値を算出し、行と列を入れ替え
print(pred)

'''
提出
'''

# 提出用サンプルの読み込み
sub = pd.read_csv('./data/sample_submission.csv', sep=',', header=None)
print(sub.head())

# 目的変数カラムの置き換え
sub[1] = pred

# ファイルのエクスポート
sub.to_csv('./submit/casting_MobileNet_finetuning_skfold.csv', sep=',', header=None, index=None)