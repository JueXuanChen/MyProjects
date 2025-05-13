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

# 導入資料
df = pd.read_csv('Going_concern_prediction.csv', sep=',')
df.head()

# 計算各標籤數量和比例權重
df['Going concern_CIK'].value_counts()

weights = compute_class_weight('balanced', classes=np.unique(df['Going concern_CIK']), y=df['Going concern_CIK'])
print(weights)

# 將資料分為trainig, validation and test set
train_data, temp_data = train_test_split(df.drop(['Data Year Fiscal'], axis=1), test_size=0.3, random_state=42, stratify=df.iloc[:, 0]) # 把第一欄Data Year Fiscal給drop掉
val_data, test_data = train_test_split(temp_data, test_size=0.3, random_state=42, stratify=temp_data.iloc[:, 0])

# 對資料做前處理與正規化
x_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values    # 第一欄 (標籤)

x_val = val_data.iloc[:, 1:].values
y_val = val_data.iloc[:, 0].values

x_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

print(x_train[0])
print(y_train[0])

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

y_train = encoder.fit_transform(y_train)
y_val = encoder.transform(y_val)
y_test = encoder.transform(y_test)

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

# 建立neural network model
inputs = layers.Input(shape=(x_train.shape[1], 1))

FC = layers.Flatten()(inputs)
FC = layers.Dense(units=15, activation='relu')(FC)
FC = layers.Dense(units=15, activation='relu')(FC)

FC = layers.Dropout(0.4)(FC)
FC = layers.Dense(units=4, activation='relu')(FC)
FC = layers.Dropout(0.4)(FC)
outputs = layers.Dense(units=2)(FC)

model = keras.Model(inputs=inputs, outputs=outputs)

# 設定模型訓練條件與策略，進行模型訓練並儲存訓練好的模型
model.compile(optimizer=tf.keras.optimizers.Lion(), loss=tf.keras.losses.CategoricalFocalCrossentropy(alpha=weights, gamma=2.0, from_logits=True), metrics=['accuracy'])
model.summary()

earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=1, verbose=1, factor=0.5, min_lr=0.00000001)
checkpoint = keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_accuracy',mode='max', save_best_only=True)
history = model.fit(x_train, y_train,
           epochs=500,
           batch_size=128,
           validation_data=(x_val, y_val),
           callbacks=[earlystopping, learning_rate_reduction, checkpoint],
           verbose=1,
           )

model.save('Going_concern_prediction_model.keras')
model.save('Going_concern_prediction_model.h5')


# 繪製training/validation的Acc/Loss，來觀察模型的訓練狀況
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
plt.savefig('Going_concern_prediction_Acc_Loss.png')
plt.show()

# 將訓練好的模型對test set進行預測，並評估模型的預測效能
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
plt.savefig('Going_concern_prediction_CM.png')
plt.show()


from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, classification_report

print(classification_report(y_true, y_pred))

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
  print(f"  Precision: {precision:.4f}")
  print(f"  Recall: {recall:.4f}")
  print(f"  F1 Score: {f1:.4f}")
  print(f"  Specificity: {specificity:.4f}")

# Calculate overall accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"\nOverall Accuracy: {accuracy:.4f}")
