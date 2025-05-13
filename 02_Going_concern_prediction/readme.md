# Going Concern Predction Dataset

## Description
* Using Going Concern Predction Dataset to trainig neural network model for going concern predction
* **Overall accuracy for testing up to 86.36%**

## Data
  
* 2 classes:
    - Going concern_CIK: Y=yes, N=no


* 70% of data for training, 20% for validation and 10% for testing


## Enviroment
Google Colab: python 3/ T4-GPU

## Output
```

Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer (InputLayer)        │ (None, 15, 1)          │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 15)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 15)             │           240 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 15)             │           240 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 15)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 4)              │            64 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (Dropout)             │ (None, 4)              │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (Dense)                 │ (None, 2)              │            10 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 554 (2.16 KB)
 Trainable params: 554 (2.16 KB)
 Non-trainable params: 0 (0.00 B)

Starting model training...
Epoch 1/500
155/155 ━━━━━━━━━━━━━━━━━━━━ 8s 26ms/step - accuracy: 0.5797 - loss: 0.1747 - val_accuracy: 0.8360 - val_loss: 0.1242 - learning_rate: 0.0010
Epoch 2/500
155/155 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.8032 - loss: 0.1415 - val_accuracy: 0.8536 - val_loss: 0.1145 - learning_rate: 0.0010
Epoch 3/500
155/155 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.7287 - loss: 0.1300 - val_accuracy: 0.8679 - val_loss: 0.1120 - learning_rate: 0.0010
Epoch 4/500
145/155 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7543 - loss: 0.1233
Epoch 4: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
155/155 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.7539 - loss: 0.1233 - val_accuracy: 0.8646 - val_loss: 0.0992 - learning_rate: 0.0010
Epoch 5/500
138/155 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7625 - loss: 0.1149
Epoch 5: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.
155/155 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.7620 - loss: 0.1151 - val_accuracy: 0.8622 - val_loss: 0.0959 - learning_rate: 5.0000e-04
Epoch 6/500
140/155 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7593 - loss: 0.1154
Epoch 6: ReduceLROnPlateau reducing learning rate to 0.0001250000059371814.
155/155 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.7599 - loss: 0.1153 - val_accuracy: 0.8599 - val_loss: 0.0944 - learning_rate: 2.5000e-04
Epoch 7/500
155/155 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.7673 - loss: 0.1120 - val_accuracy: 0.8691 - val_loss: 0.0952 - learning_rate: 1.2500e-04
Epoch 8/500
155/155 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.7817 - loss: 0.1102 - val_accuracy: 0.8698 - val_loss: 0.0930 - learning_rate: 1.2500e-04
Epoch 9/500
135/155 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7677 - loss: 0.1148
Epoch 9: ReduceLROnPlateau reducing learning rate to 6.25000029685907e-05.
155/155 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.7677 - loss: 0.1146 - val_accuracy: 0.8651 - val_loss: 0.0936 - learning_rate: 1.2500e-04
Epoch 10/500
141/155 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7710 - loss: 0.1084
Epoch 10: ReduceLROnPlateau reducing learning rate to 3.125000148429535e-05.
155/155 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.7712 - loss: 0.1085 - val_accuracy: 0.8684 - val_loss: 0.0931 - learning_rate: 6.2500e-05
Epoch 11/500
141/155 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7673 - loss: 0.1128
Epoch 11: ReduceLROnPlateau reducing learning rate to 1.5625000742147677e-05.
155/155 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.7671 - loss: 0.1127 - val_accuracy: 0.8607 - val_loss: 0.0933 - learning_rate: 3.1250e-05
Epoch 12/500
137/155 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7690 - loss: 0.1117
Epoch 12: ReduceLROnPlateau reducing learning rate to 7.812500371073838e-06.
155/155 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.7687 - loss: 0.1116 - val_accuracy: 0.8683 - val_loss: 0.0930 - learning_rate: 1.5625e-05
Epoch 13/500
154/155 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7719 - loss: 0.1107
Epoch 13: ReduceLROnPlateau reducing learning rate to 3.906250185536919e-06.
155/155 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.7718 - loss: 0.1108 - val_accuracy: 0.8683 - val_loss: 0.0930 - learning_rate: 7.8125e-06
Epoch 14/500
137/155 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7711 - loss: 0.1066
Epoch 14: ReduceLROnPlateau reducing learning rate to 1.9531250927684596e-06.
155/155 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.7711 - loss: 0.1070 - val_accuracy: 0.8662 - val_loss: 0.0930 - learning_rate: 3.9063e-06
Epoch 15/500
144/155 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.7721 - loss: 0.1113
Epoch 15: ReduceLROnPlateau reducing learning rate to 9.765625463842298e-07.
155/155 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.7723 - loss: 0.1114 - val_accuracy: 0.8654 - val_loss: 0.0931 - learning_rate: 1.9531e-06
Epoch 16/500
155/155 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.7708 - loss: 0.1097
Epoch 16: ReduceLROnPlateau reducing learning rate to 4.882812731921149e-07.
155/155 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - accuracy: 0.7708 - loss: 0.1097 - val_accuracy: 0.8654 - val_loss: 0.0931 - learning_rate: 9.7656e-07
Epoch 17/500
146/155 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.7714 - loss: 0.1113
Epoch 17: ReduceLROnPlateau reducing learning rate to 2.4414063659605745e-07.
155/155 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.7713 - loss: 0.1114 - val_accuracy: 0.8654 - val_loss: 0.0931 - learning_rate: 4.8828e-07
Epoch 18/500
146/155 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7697 - loss: 0.1102
Epoch 18: ReduceLROnPlateau reducing learning rate to 1.2207031829802872e-07.
155/155 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.7697 - loss: 0.1102 - val_accuracy: 0.8654 - val_loss: 0.0931 - learning_rate: 2.4414e-07

Saving the model.
```
\
![](https://github.com/JueXuanChen/MyProjects/blob/main/02_Going_concern_prediction/Going_concern_prediction_Acc_Loss.png)

```
Model testing...
80/80 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step

              precision    recall  f1-score   support

           0       0.98      0.85      0.91      2142
           1       0.54      0.91      0.68       409

    accuracy                           0.86      2551
   macro avg       0.76      0.88      0.80      2551
weighted avg       0.91      0.86      0.88      2551

Class 0:
  Precision: 0.9802
  Recall: 0.8548
  F1 Score: 0.9132
  Specificity: 0.9095
Class 1:
  Precision: 0.5447
  Recall: 0.9095
  F1 Score: 0.6813
  Specificity: 0.8548

Overall Accuracy: 0.8636
```
\
![](https://github.com/JueXuanChen/MyProjects/blob/main/02_Going_concern_prediction/Going_concern_prediction_CM.png)
