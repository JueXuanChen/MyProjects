# Default of Credit Card Clients Dataset

## Description
* Using Default of Credit Card Clients Dataset to train neural network model for Default payment prediction
* **Overall accuracy for testing up to 79.81%**

## Data
* Data link: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
  
* 2 classes:
    - default.payment.next.month: Default payment (1=yes, 0=no)


* 70% of data for training, 20% for validation and 10% for testing


## Enviroment
Google Colab: python 3/ T4-GPU

## Output
```

Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_4 (InputLayer)      │ (None, 23, 1)          │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten_4 (Flatten)             │ (None, 23)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_13 (Dense)                │ (None, 23)             │           552 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_3 (Dropout)             │ (None, 23)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_14 (Dense)                │ (None, 8)              │           192 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_4 (Dropout)             │ (None, 8)              │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_15 (Dense)                │ (None, 2)              │            18 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 762 (2.98 KB)
 Trainable params: 762 (2.98 KB)
 Non-trainable params: 0 (0.00 B)

Starting model training...
Epoch 1/1000
165/165 ━━━━━━━━━━━━━━━━━━━━ 6s 21ms/step - accuracy: 0.4818 - loss: 0.2447 - val_accuracy: 0.7792 - val_loss: 0.1744 - learning_rate: 0.0010
Epoch 2/1000
165/165 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.7482 - loss: 0.1715 - val_accuracy: 0.8032 - val_loss: 0.1632 - learning_rate: 0.0010
Epoch 3/1000
165/165 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.7810 - loss: 0.1618 - val_accuracy: 0.7871 - val_loss: 0.1553 - learning_rate: 0.0010
Epoch 4/1000
160/165 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7824 - loss: 0.1608
Epoch 4: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
165/165 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.7825 - loss: 0.1608 - val_accuracy: 0.8003 - val_loss: 0.1543 - learning_rate: 0.0010
Epoch 5/1000
165/165 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.7841 - loss: 0.1587 - val_accuracy: 0.8014 - val_loss: 0.1536 - learning_rate: 5.0000e-04
Epoch 6/1000
163/165 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7853 - loss: 0.1594
Epoch 6: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.
165/165 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.7853 - loss: 0.1594 - val_accuracy: 0.7983 - val_loss: 0.1528 - learning_rate: 5.0000e-04
Epoch 7/1000
165/165 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.7866 - loss: 0.1580 - val_accuracy: 0.7990 - val_loss: 0.1525 - learning_rate: 2.5000e-04
Epoch 8/1000
152/165 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7914 - loss: 0.1558
Epoch 8: ReduceLROnPlateau reducing learning rate to 0.0001250000059371814.
165/165 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.7916 - loss: 0.1559 - val_accuracy: 0.7990 - val_loss: 0.1526 - learning_rate: 2.5000e-04
Epoch 9/1000
165/165 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.7858 - loss: 0.1584 - val_accuracy: 0.7987 - val_loss: 0.1523 - learning_rate: 1.2500e-04
Epoch 10/1000
165/165 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7911 - loss: 0.1560
Epoch 10: ReduceLROnPlateau reducing learning rate to 6.25000029685907e-05.
165/165 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.7911 - loss: 0.1560 - val_accuracy: 0.7994 - val_loss: 0.1523 - learning_rate: 1.2500e-04
Epoch 11/1000
165/165 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.7912 - loss: 0.1560 - val_accuracy: 0.7987 - val_loss: 0.1523 - learning_rate: 6.2500e-05
Epoch 12/1000
164/165 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7922 - loss: 0.1559
Epoch 12: ReduceLROnPlateau reducing learning rate to 3.125000148429535e-05.
165/165 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.7922 - loss: 0.1559 - val_accuracy: 0.8002 - val_loss: 0.1528 - learning_rate: 6.2500e-05
Epoch 13/1000
165/165 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.7884 - loss: 0.1589 - val_accuracy: 0.7994 - val_loss: 0.1523 - learning_rate: 3.1250e-05
Epoch 14/1000
162/165 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7964 - loss: 0.1562
Epoch 14: ReduceLROnPlateau reducing learning rate to 1.5625000742147677e-05.
165/165 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.7963 - loss: 0.1562 - val_accuracy: 0.7989 - val_loss: 0.1525 - learning_rate: 3.1250e-05
Epoch 15/1000
165/165 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.7894 - loss: 0.1573 - val_accuracy: 0.7990 - val_loss: 0.1524 - learning_rate: 1.5625e-05
Epoch 16/1000
165/165 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7930 - loss: 0.1563
Epoch 16: ReduceLROnPlateau reducing learning rate to 7.812500371073838e-06.
165/165 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.7930 - loss: 0.1563 - val_accuracy: 0.7994 - val_loss: 0.1523 - learning_rate: 1.5625e-05
Epoch 17/1000
165/165 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.7928 - loss: 0.1554 - val_accuracy: 0.7994 - val_loss: 0.1522 - learning_rate: 7.8125e-06
Epoch 18/1000
163/165 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - accuracy: 0.7924 - loss: 0.1563
Epoch 18: ReduceLROnPlateau reducing learning rate to 3.906250185536919e-06.
165/165 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.7924 - loss: 0.1563 - val_accuracy: 0.7992 - val_loss: 0.1523 - learning_rate: 7.8125e-06
Epoch 19/1000
165/165 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.7886 - loss: 0.1583 - val_accuracy: 0.7990 - val_loss: 0.1523 - learning_rate: 3.9063e-06
Epoch 20/1000
163/165 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7965 - loss: 0.1565
Epoch 20: ReduceLROnPlateau reducing learning rate to 1.9531250927684596e-06.
165/165 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.7965 - loss: 0.1565 - val_accuracy: 0.7992 - val_loss: 0.1523 - learning_rate: 3.9063e-06
Epoch 21/1000
165/165 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.7948 - loss: 0.1560 - val_accuracy: 0.7992 - val_loss: 0.1523 - learning_rate: 1.9531e-06
Epoch 22/1000
160/165 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7982 - loss: 0.1534
Epoch 22: ReduceLROnPlateau reducing learning rate to 9.765625463842298e-07.
165/165 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.7980 - loss: 0.1535 - val_accuracy: 0.7994 - val_loss: 0.1523 - learning_rate: 1.9531e-06
Epoch 23/1000
165/165 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.7889 - loss: 0.1583 - val_accuracy: 0.7994 - val_loss: 0.1523 - learning_rate: 9.7656e-07
Epoch 24/1000
148/165 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7927 - loss: 0.1554
Epoch 24: ReduceLROnPlateau reducing learning rate to 4.882812731921149e-07.
165/165 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.7925 - loss: 0.1555 - val_accuracy: 0.7994 - val_loss: 0.1523 - learning_rate: 9.7656e-07
Epoch 25/1000
165/165 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.7948 - loss: 0.1567 - val_accuracy: 0.7994 - val_loss: 0.1523 - learning_rate: 4.8828e-07
Epoch 26/1000
165/165 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7923 - loss: 0.1554
Epoch 26: ReduceLROnPlateau reducing learning rate to 2.4414063659605745e-07.
165/165 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.7923 - loss: 0.1554 - val_accuracy: 0.7994 - val_loss: 0.1523 - learning_rate: 4.8828e-07
Epoch 27/1000
165/165 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.7925 - loss: 0.1549 - val_accuracy: 0.7994 - val_loss: 0.1523 - learning_rate: 2.4414e-07

Saving the model.
```
\
![](https://github.com/JueXuanChen/MyProjects/blob/main/01_Default_of_Credit_Card_Clients_Dataset/UCI_Credit_Card_model_Acc_Loss.png)

```
Model testing...
85/85 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step

Class 0:
  Recall: 0.8887
  Specificity: 0.4791
  Precision: 0.8573
  F1 Score: 0.8728
Class 1:
  Recall: 0.4791
  Specificity: 0.8887
  Precision: 0.5500
  F1 Score: 0.5121

Overall Accuracy: 0.7981
```
\
![](https://github.com/JueXuanChen/MyProjects/blob/main/01_Default_of_Credit_Card_Clients_Dataset/UCI_Credit_Card_model_CM.png)
