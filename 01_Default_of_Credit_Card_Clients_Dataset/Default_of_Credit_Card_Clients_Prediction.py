import numpy as np
import pandas as pd
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# 讀取檔案
df = pd.read_csv('UCI_Credit_Card.csv', sep=',')
df.head()

# 計算是否欠繳的比數及比例
df['default.payment.next.month'].value_counts()

weights = compute_class_weight('balanced', classes=np.unique(df['default.payment.next.month']), y=df['default.payment.next.month'])
print(weights)

# 將資料分成training, validation and test set
train_data, temp_data = train_test_split(df.drop(['ID'], axis=1), test_size=0.3, random_state=42, stratify=df.iloc[:, -1]) # 把第一欄ID給drop掉
val_data, test_data = train_test_split(temp_data, test_size=0.3, random_state=42, stratify=temp_data.iloc[:, -1])

train_data.head()
val_data.head()
test_data.head()

# 將資料進行前處理與正規化
x_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values    # 最後一欄 (標籤)

x_val = val_data.iloc[:, :-1].values
y_val = val_data.iloc[:, -1].values

x_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_val shape:", x_val.shape)
print("y_val shape:", y_val.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

# 建立nueral neatwork model
inputs = layers.Input(shape=(x_train.shape[1], 1))

FC = layers.Flatten()(inputs)
FC = layers.Dense(units=23, activation='relu')(FC)

FC = layers.Dropout(0.4)(FC)
FC = layers.Dense(units=8, activation='relu')(FC)
FC = layers.Dropout(0.4)(FC)
outputs = layers.Dense(units=2)(FC)

model = keras.Model(inputs=inputs, outputs=outputs)

# 設定模型訓練條件與策略，並訓練模型，再將模型儲存
model.compile(optimizer=tf.keras.optimizers.Lion(), loss=tf.keras.losses.CategoricalFocalCrossentropy(alpha=weights, gamma=2.0, from_logits=True), metrics=['accuracy'])
model.summary()

earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00000001)
history = model.fit(x_train, y_train,
           epochs=1000,
           batch_size=128,
           validation_data=(x_val, y_val),
           callbacks=[earlystopping, learning_rate_reduction],
           verbose=1,
           )

model.save('UCI_Credit_Card_model.keras')
model.save('UCI_Credit_Card_model.h5')

# 繪製trainig/validation的acc/loss，來判斷模型訓練的狀況
# Plot training and validation accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.save('UCI_Credit_Card_model_Acc_Loss.png')
plt.show()

# 用test set來評估訓練好的模型的預測效能
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

y_pred_prob = model.predict(x_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)


cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(2), yticklabels=range(2))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.save('UCI_Credit_Card_model_CM.png')
plt.show()


from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

# Calculate metrics for each class
for i in range(2):
  TP = cm[i, i]
  FP = np.sum(cm[:, i]) - TP
  FN = np.sum(cm[i, :]) - TP
  TN = np.sum(cm) - TP - FP - FN

  recall = TP / (TP + FN) if (TP + FN) > 0 else 0
  specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
  precision = TP / (TP + FP) if (TP + FP) > 0 else 0
  f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

  print(f"Class {i}:")
  print(f"  Recall: {recall:.4f}")
  print(f"  Specificity: {specificity:.4f}")
  print(f"  Precision: {precision:.4f}")
  print(f"  F1 Score: {f1:.4f}")

# Calculate overall accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"\nOverall Accuracy: {accuracy:.4f}")
