# Chest X-Ray Images (Pneumonia)

## Description
* Using Chest X-Ray Images Data Set to fine tuning Inception_v3 model for classifying normal and pneumonia X-Ray images
* **Overall accuracy for testing up to 89.58%**

## Data
* Data link: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data

* 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal)

* 70% of data for training, 20% for validation and 10% for testing

## Model
* Inception_v3: https://www.kaggle.com/models/google/inception-v3/TensorFlow2/feature-vector/2

## Enviroment
* i5-13500/rtx4070ti with 32GB ram
* cuda 11.8
* cudnn 8.9.7
* TensorRT 8.6.1
* python 3.10
* tensorflow 2.13.1
* tensorflow-hub 0.15.0
* numpy 1.24.3
* pandas 2.1.4
* scikit-learn 1.3.2

## Output
```
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_2 (InputLayer)        [(None, 299, 299, 3)]     0         
                                                                 
 keras_layer (KerasLayer)    (None, 2048)              21802784  
                                                                 
 dropout (Dropout)           (None, 2048)              0         
                                                                 
 dense (Dense)               (None, 2)                 4098      
                                                                 
=================================================================
Total params: 21806882 (83.19 MB)
Trainable params: 21772450 (83.06 MB)
Non-trainable params: 34432 (134.50 KB)
_________________________________________________________________

Starting model training...
Epoch 1/1000
129/129 [==============================] - 43s 181ms/step - loss: 0.3210 - accuracy: 0.8819 - val_loss: 7.8922 - val_accuracy: 0.7144 - lr: 1.0000e-04
Epoch 2/1000
129/129 [==============================] - 23s 157ms/step - loss: 0.3660 - accuracy: 0.9244 - val_loss: 8.4217 - val_accuracy: 0.7299 - lr: 1.0000e-04
Epoch 3/1000
129/129 [==============================] - 23s 152ms/step - loss: 0.4613 - accuracy: 0.9217 - val_loss: 1.6573 - val_accuracy: 0.5606 - lr: 1.0000e-04
Epoch 4/1000
129/129 [==============================] - 23s 149ms/step - loss: 0.4499 - accuracy: 0.9307 - val_loss: 1.1282 - val_accuracy: 0.6395 - lr: 1.0000e-04
Epoch 5/1000
128/129 [============================>.] - ETA: 0s - loss: 0.3454 - accuracy: 0.9353  
Epoch 5: ReduceLROnPlateau reducing learning rate to 1.4999999621068127e-05.
129/129 [==============================] - 22s 144ms/step - loss: 0.3457 - accuracy: 0.9351 - val_loss: 0.4530 - val_accuracy: 0.6656 - lr: 1.0000e-04
Epoch 6/1000
129/129 [==============================] - 22s 143ms/step - loss: 0.2763 - accuracy: 0.9444 - val_loss: 0.3301 - val_accuracy: 0.7754 - lr: 1.5000e-05
Epoch 7/1000
129/129 [==============================] - 21s 141ms/step - loss: 0.2636 - accuracy: 0.9451 - val_loss: 0.3489 - val_accuracy: 0.7258 - lr: 1.5000e-05
Epoch 8/1000
129/129 [==============================] - 22s 145ms/step - loss: 0.2537 - accuracy: 0.9436 - val_loss: 0.5036 - val_accuracy: 0.6444 - lr: 1.5000e-05
Epoch 9/1000
129/129 [==============================] - 22s 143ms/step - loss: 0.2446 - accuracy: 0.9441 - val_loss: 0.3031 - val_accuracy: 0.7836 - lr: 1.5000e-05
Epoch 10/1000
129/129 [==============================] - 22s 143ms/step - loss: 0.2261 - accuracy: 0.9507 - val_loss: 0.2222 - val_accuracy: 0.9439 - lr: 1.5000e-05
Epoch 11/1000
129/129 [==============================] - 21s 142ms/step - loss: 0.2103 - accuracy: 0.9597 - val_loss: 0.2402 - val_accuracy: 0.8771 - lr: 1.5000e-05
Epoch 12/1000
129/129 [==============================] - 22s 146ms/step - loss: 0.1999 - accuracy: 0.9563 - val_loss: 0.3360 - val_accuracy: 0.7209 - lr: 1.5000e-05
Epoch 13/1000
128/129 [============================>.] - ETA: 0s - loss: 0.1907 - accuracy: 0.9612  
Epoch 13: ReduceLROnPlateau reducing learning rate to 2.249999943160219e-06.
129/129 [==============================] - 22s 146ms/step - loss: 0.1909 - accuracy: 0.9610 - val_loss: 0.3293 - val_accuracy: 0.7413 - lr: 1.5000e-05
Epoch 14/1000
129/129 [==============================] - 22s 143ms/step - loss: 0.1802 - accuracy: 0.9656 - val_loss: 0.2312 - val_accuracy: 0.8682 - lr: 2.2500e-06
Epoch 15/1000
129/129 [==============================] - 21s 142ms/step - loss: 0.1795 - accuracy: 0.9661 - val_loss: 0.2008 - val_accuracy: 0.9170 - lr: 2.2500e-06
Epoch 16/1000
128/129 [============================>.] - ETA: 0s - loss: 0.1769 - accuracy: 0.9670  
Epoch 16: ReduceLROnPlateau reducing learning rate to 3.374999778316123e-07.
129/129 [==============================] - 22s 145ms/step - loss: 0.1769 - accuracy: 0.9666 - val_loss: 0.2102 - val_accuracy: 0.8910 - lr: 2.2500e-06
Epoch 17/1000
129/129 [==============================] - 21s 142ms/step - loss: 0.1767 - accuracy: 0.9654 - val_loss: 0.2169 - val_accuracy: 0.8836 - lr: 3.3750e-07
Epoch 18/1000
129/129 [==============================] - 23s 150ms/step - loss: 0.1747 - accuracy: 0.9685 - val_loss: 0.2160 - val_accuracy: 0.8853 - lr: 3.3750e-07
Epoch 19/1000
129/129 [==============================] - ETA: 0s - loss: 0.1744 - accuracy: 0.9695  
Epoch 19: ReduceLROnPlateau reducing learning rate to 5.0624997527393134e-08.
129/129 [==============================] - 21s 141ms/step - loss: 0.1744 - accuracy: 0.9695 - val_loss: 0.2126 - val_accuracy: 0.8885 - lr: 3.3750e-07
Epoch 20/1000
129/129 [==============================] - 21s 140ms/step - loss: 0.1744 - accuracy: 0.9683 - val_loss: 0.2156 - val_accuracy: 0.8836 - lr: 5.0625e-08
Epoch 21/1000
129/129 [==============================] - 22s 150ms/step - loss: 0.1758 - accuracy: 0.9663 - val_loss: 0.2158 - val_accuracy: 0.8836 - lr: 5.0625e-08
Epoch 22/1000
128/129 [============================>.] - ETA: 0s - loss: 0.1746 - accuracy: 0.9678  
Epoch 22: ReduceLROnPlateau reducing learning rate to 1e-08.
129/129 [==============================] - 21s 141ms/step - loss: 0.1746 - accuracy: 0.9678 - val_loss: 0.2130 - val_accuracy: 0.8869 - lr: 5.0625e-08
Epoch 23/1000
129/129 [==============================] - 21s 140ms/step - loss: 0.1760 - accuracy: 0.9671 - val_loss: 0.2136 - val_accuracy: 0.8869 - lr: 1.0000e-08
Epoch 24/1000
129/129 [==============================] - 21s 142ms/step - loss: 0.1743 - accuracy: 0.9685 - val_loss: 0.2147 - val_accuracy: 0.8861 - lr: 1.0000e-08
Epoch 25/1000
129/129 [==============================] - 21s 142ms/step - loss: 0.1737 - accuracy: 0.9695 - val_loss: 0.2151 - val_accuracy: 0.8845 - lr: 1.0000e-08

Saving the model.
```
\
![](https://github.com/JueXuanChen/MyProjects/blob/main/Chest_X_Ray_Images_Pneumonia/chest_xray_model_Acc_loss.png)

```
Model testing...
17/17 [==============================] - 2s 96ms/step
Number of predictions: 528
Number of true labels from df: 528

Class 0:
  Recall: 0.9720
  Specificity: 0.8675
  Precision: 0.7316
  F1 Score: 0.8348
Class 1:
  Recall: 0.8675
  Specificity: 0.9720
  Precision: 0.9882
  F1 Score: 0.9239

Overall Accuracy: 0.8958
```
\
![](https://github.com/JueXuanChen/MyProjects/blob/main/Chest_X_Ray_Images_Pneumonia/chest_xray_model_CM.png)
