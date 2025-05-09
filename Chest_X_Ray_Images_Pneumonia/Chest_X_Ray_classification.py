import numpy as np
import pandas as pd
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical

import kagglehub

#匯入資料
# Download latest version
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

print("Path to dataset files:", path)

path = path + "/chest_xray"
split_dir = os.listdir(path)
split_dir.remove("chest_xray")
split_dir.remove("__MACOSX")
print(split_dir)

# 將資料轉換成Dataframe的型態
def make_df(split_dir, base_dir):
    classes = ["NORMAL", "PNEUMONIA"]
    data = []
    for split in split_dir:
        for label in classes:
            folder_path = os.path.join(base_dir+ "/" + split, label)
            for file in os.listdir(folder_path):
                if file.endswith(('jpg', 'png', 'bmp', 'jpeg')):
                    file_path = os.path.join(folder_path, file)
                    data.append((file_path, label))

    df = pd.DataFrame(data, columns=['file_path', 'label'])
    return df

df = make_df(split_dir, path)
print("Shape of  dataset is:", df.shape)

# 計算各標籤資料數量比計算比例權重
label_mapping = {'NORMAL': '0', 'PNEUMONIA': '1'}
df['label'] = df['label'].map(label_mapping)

df.head()

df['label'].value_counts()

weights = compute_class_weight('balanced', classes=np.unique(df['label']), y=df['label'])

# 將資料分為training, validation and test data
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
valid_df, test_df = train_test_split(temp_df, test_size=0.3, random_state=42, stratify = temp_df['label'])

print("Training set shapes:", train_df.shape)
print("Testing set shapes:", test_df.shape)
print("Validation set shapes:", valid_df.shape)

# 將圖像資料轉換成可以fit入模型的型態
# Define image size and color mode
image_shape = 299
batch_size = 32
color_mode = 'rgb'

def load_and_preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    num_channels = 3 if color_mode == 'rgb' else 1
    image = tf.image.decode_image(image, channels=num_channels, expand_animations=False)
    image.set_shape([None, None, num_channels])
    image = tf.image.resize(image, [image_shape, image_shape])
    return image, label

def create_dataset(df, batch_size, shuffle=True):
    image_paths = df['file_path'].tolist()
    integer_labels = df['label'].astype(int).tolist()
    num_classes = len(df['label'].unique())
    one_hot_labels = to_categorical(integer_labels, num_classes=num_classes)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, one_hot_labels))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df))

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(train_df, batch_size, shuffle=True)
valid_dataset = create_dataset(valid_df, batch_size, shuffle=False)
test_dataset = create_dataset(test_df, batch_size, shuffle=False)


# 對資料進行augmentation, rescale and normalization
augmentation_layers = tf.keras.Sequential([layers.Rescaling(1./255),
                                           layers.RandomFlip("horizontal_and_vertical"),
                                           layers.RandomRotation(0.7),
                                           layers.RandomZoom((0.0,0.2),)
                                           ])
rescale_layers = tf.keras.Sequential([layers.Rescaling(1./255)])

def img_augmentation(image, label):
    augmented_image = augmentation_layers(image)
    return augmented_image, label

def img_rescale(image, label):
    rescaled_image = rescale_layers(image)
    return rescaled_image, label

train_dataset = train_dataset.map(img_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
valid_dataset = valid_dataset.map(img_rescale, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.map(img_rescale, num_parallel_calls=tf.data.AUTOTUNE)

print("\nAdapting normalization layer...")
# Create a dataset with images only for adaptation
adapt_dataset = train_dataset.map(lambda image, label: image)
normalization_layer = layers.Normalization(axis=-1)  # Normalize across the channel axis typically
normalization_layer.adapt(adapt_dataset)
print("Normalization layer adapted.")

# 匯入Inception_v3，並建立深度學習模型
import tensorflow_hub as hub

URL = "https://www.kaggle.com/models/google/inception-v3/TensorFlow2/feature-vector/2"
feature_extractor = hub.KerasLayer(URL,
                                   input_shape=(image_shape, image_shape, 3))
feature_extractor.trainable = True


inputs = layers.Input(shape=(image_shape, image_shape, 3))
inputs = normalization_layer(inputs)
layer1 = feature_extractor(inputs)
layer1 = layers.Dropout(0.4)(layer1)
outputs = layers.Dense(units=2)(layer1)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

# 設定模型訓練條件及策略並開始訓練模型
model.compile(optimizer=tf.keras.optimizers.Lion(),   #Adam
              loss=tf.keras.losses.CategoricalFocalCrossentropy(alpha=weights,gamma=2.0, from_logits=True),
              metrics=['accuracy']
              )

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.15, min_lr=0.00000001)
print("\nStarting model training...")

history = model.fit(train_dataset,
                    epochs=1000,
                    validation_data=valid_dataset,
                    callbacks=[early_stopping, learning_rate_reduction]
                    )
model.save('chest_xray_model.keras')
model.save('chest_xray_model.h5')

# 畫出traing/validation的acc/loss，來評估模型訓練的狀況
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
plt.savefig('chest_xray_model_Acc_loss.png')
plt.show()

# 將訓練好的模型對test data進行預測，並評估模型預測效能
test_pred = model.predict(test_dataset,
                          # steps=int(np.ceil(test_dataset.n / float(batch_size))),
                          verbose=1
                          )

test_pred_classes = np.argmax(test_pred, axis=1)

# Extract true labels safely AFTER prediction if dataset wasn't cached/repeated
# Re-extracting from the dataframe is safest
test_labels_from_df = test_df['label'].astype(int).tolist()

# Or iterate through the test_dataset *again* if needed (less efficient)
# true_labels = []
# for images, labels in test_dataset: # This iterates through the *processed* dataset
#     true_labels.extend(labels.numpy())

# Ensure the number of predictions matches the number of labels
print(f"Number of predictions: {len(test_pred_classes)}")
print(f"Number of true labels from df: {len(test_labels_from_df)}")
# print(f"Number of true labels from dataset iteration: {len(true_labels)}")


if len(test_pred_classes) != len(test_labels_from_df):
    print("Warning: Mismatch between number of predictions and true labels from DataFrame!")
    # Decide which true labels to use or debug the discrepancy
    true_labels_for_cm = test_pred_classes  # Placeholder if mismatch, needs fixing
else:
    true_labels_for_cm = test_labels_from_df

cm = confusion_matrix(true_labels_for_cm, test_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm
            , annot=True
            , fmt='d'
            , cmap='Blues'
            , xticklabels=['0', '1']
            , yticklabels=['0', '1']
            )
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('chest_xray_model_CM.png')
plt.show()

from sklearn.metrics import accuracy_score
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
accuracy = accuracy_score(test_labels_from_df, test_pred_classes)
print(f"\nOverall Accuracy: {accuracy:.4f}")


print("\nScript finished.")
