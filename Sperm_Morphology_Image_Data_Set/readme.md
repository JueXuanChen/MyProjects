# Sperm Morphology Image Data Set (SMIDS)
 
## Description
* Using Sperm Morphology Image Data Set to fine tuning Inception_v3 model for identifying Sperm Morphology
* **Overall accuracy for testing up to 91.85%**

## Data
* Data link: https://www.kaggle.com/datasets/orvile/sperm-morphology-image-data-set-smids/data

* 3 classes:
    - Normal sperm: 1021 images
    - Abnormal sperm: 1005 images
    - Non-sperm: 974 images

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
                                                                 
 dense (Dense)               (None, 3)                 6147      
                                                                 
=================================================================
Total params: 21808931 (83.19 MB)
Trainable params: 21774499 (83.06 MB)
Non-trainable params: 34432 (134.50 KB)
_________________________________________________________________


Starting model training...
Epoch 1/1000
66/66 [==============================] - 33s 239ms/step - loss: 0.5001 - accuracy: 0.7095 - val_loss: 2.2271 - val_accuracy: 0.3381 - lr: 1.0000e-04
Epoch 2/1000
66/66 [==============================] - 11s 148ms/step - loss: 0.4298 - accuracy: 0.7852 - val_loss: 7.4036 - val_accuracy: 0.3365 - lr: 1.0000e-04
Epoch 3/1000
66/66 [==============================] - 11s 146ms/step - loss: 0.4534 - accuracy: 0.7990 - val_loss: 8.7355 - val_accuracy: 0.3413 - lr: 1.0000e-04
Epoch 4/1000
66/66 [==============================] - 11s 146ms/step - loss: 0.5061 - accuracy: 0.8105 - val_loss: 7.1388 - val_accuracy: 0.3032 - lr: 1.0000e-04
Epoch 5/1000
66/66 [==============================] - ETA: 0s - loss: 0.5592 - accuracy: 0.8357  
Epoch 5: ReduceLROnPlateau reducing learning rate to 6.999999823165126e-05.
66/66 [==============================] - 10s 140ms/step - loss: 0.5592 - accuracy: 0.8357 - val_loss: 10.4789 - val_accuracy: 0.3159 - lr: 1.0000e-04
Epoch 6/1000
66/66 [==============================] - 10s 139ms/step - loss: 0.5962 - accuracy: 0.8429 - val_loss: 2.5340 - val_accuracy: 0.5619 - lr: 7.0000e-05
Epoch 7/1000
66/66 [==============================] - 10s 142ms/step - loss: 0.5921 - accuracy: 0.8538 - val_loss: 0.9039 - val_accuracy: 0.7270 - lr: 7.0000e-05
Epoch 8/1000
66/66 [==============================] - 10s 140ms/step - loss: 0.5660 - accuracy: 0.8643 - val_loss: 1.3367 - val_accuracy: 0.6143 - lr: 7.0000e-05
Epoch 9/1000
66/66 [==============================] - ETA: 0s - loss: 0.5194 - accuracy: 0.8714  
Epoch 9: ReduceLROnPlateau reducing learning rate to 4.899999621557071e-05.
66/66 [==============================] - 10s 143ms/step - loss: 0.5194 - accuracy: 0.8714 - val_loss: 0.7350 - val_accuracy: 0.6984 - lr: 7.0000e-05
Epoch 10/1000
66/66 [==============================] - 10s 141ms/step - loss: 0.4787 - accuracy: 0.8595 - val_loss: 1.0070 - val_accuracy: 0.7175 - lr: 4.9000e-05
Epoch 11/1000
66/66 [==============================] - 10s 141ms/step - loss: 0.4485 - accuracy: 0.8719 - val_loss: 0.7615 - val_accuracy: 0.7302 - lr: 4.9000e-05
Epoch 12/1000
66/66 [==============================] - 10s 142ms/step - loss: 0.4035 - accuracy: 0.8767 - val_loss: 0.4658 - val_accuracy: 0.8413 - lr: 4.9000e-05
Epoch 13/1000
66/66 [==============================] - 10s 142ms/step - loss: 0.3815 - accuracy: 0.8690 - val_loss: 0.4193 - val_accuracy: 0.8492 - lr: 4.9000e-05
Epoch 14/1000
66/66 [==============================] - 10s 142ms/step - loss: 0.3486 - accuracy: 0.8719 - val_loss: 0.3594 - val_accuracy: 0.8587 - lr: 4.9000e-05
Epoch 15/1000
66/66 [==============================] - 10s 141ms/step - loss: 0.3181 - accuracy: 0.8919 - val_loss: 0.3649 - val_accuracy: 0.8508 - lr: 4.9000e-05
Epoch 16/1000
66/66 [==============================] - ETA: 0s - loss: 0.2986 - accuracy: 0.8800  
Epoch 16: ReduceLROnPlateau reducing learning rate to 3.4299996332265434e-05.
66/66 [==============================] - 10s 141ms/step - loss: 0.2986 - accuracy: 0.8800 - val_loss: 0.4152 - val_accuracy: 0.8302 - lr: 4.9000e-05
Epoch 17/1000
66/66 [==============================] - 10s 142ms/step - loss: 0.2824 - accuracy: 0.8929 - val_loss: 0.3061 - val_accuracy: 0.8778 - lr: 3.4300e-05
Epoch 18/1000
66/66 [==============================] - 10s 140ms/step - loss: 0.2712 - accuracy: 0.9057 - val_loss: 0.3437 - val_accuracy: 0.8397 - lr: 3.4300e-05
Epoch 19/1000
66/66 [==============================] - ETA: 0s - loss: 0.2620 - accuracy: 0.8962  
Epoch 19: ReduceLROnPlateau reducing learning rate to 2.400999692326877e-05.
66/66 [==============================] - 10s 142ms/step - loss: 0.2620 - accuracy: 0.8962 - val_loss: 0.2930 - val_accuracy: 0.8714 - lr: 3.4300e-05
Epoch 20/1000
66/66 [==============================] - 10s 142ms/step - loss: 0.2440 - accuracy: 0.9086 - val_loss: 0.2849 - val_accuracy: 0.8794 - lr: 2.4010e-05
Epoch 21/1000
66/66 [==============================] - 10s 142ms/step - loss: 0.2378 - accuracy: 0.9033 - val_loss: 0.2667 - val_accuracy: 0.8778 - lr: 2.4010e-05
Epoch 22/1000
66/66 [==============================] - ETA: 0s - loss: 0.2279 - accuracy: 0.9124  
Epoch 22: ReduceLROnPlateau reducing learning rate to 1.6806997336971108e-05.
66/66 [==============================] - 10s 141ms/step - loss: 0.2279 - accuracy: 0.9124 - val_loss: 0.2938 - val_accuracy: 0.8587 - lr: 2.4010e-05
Epoch 23/1000
66/66 [==============================] - 10s 142ms/step - loss: 0.2232 - accuracy: 0.9152 - val_loss: 0.2540 - val_accuracy: 0.8857 - lr: 1.6807e-05
Epoch 24/1000
66/66 [==============================] - 10s 142ms/step - loss: 0.2192 - accuracy: 0.9124 - val_loss: 0.2515 - val_accuracy: 0.8810 - lr: 1.6807e-05
Epoch 25/1000
66/66 [==============================] - ETA: 0s - loss: 0.2130 - accuracy: 0.9119  
Epoch 25: ReduceLROnPlateau reducing learning rate to 1.1764897499233484e-05.
66/66 [==============================] - 10s 141ms/step - loss: 0.2130 - accuracy: 0.9119 - val_loss: 0.2585 - val_accuracy: 0.8810 - lr: 1.6807e-05
Epoch 26/1000
66/66 [==============================] - 10s 141ms/step - loss: 0.2075 - accuracy: 0.9171 - val_loss: 0.2669 - val_accuracy: 0.8810 - lr: 1.1765e-05
Epoch 27/1000
66/66 [==============================] - 10s 143ms/step - loss: 0.2025 - accuracy: 0.9229 - val_loss: 0.2494 - val_accuracy: 0.8984 - lr: 1.1765e-05
Epoch 28/1000
66/66 [==============================] - 10s 142ms/step - loss: 0.2024 - accuracy: 0.9243 - val_loss: 0.2492 - val_accuracy: 0.8952 - lr: 1.1765e-05
Epoch 29/1000
66/66 [==============================] - ETA: 0s - loss: 0.2000 - accuracy: 0.9214  
Epoch 29: ReduceLROnPlateau reducing learning rate to 8.235428504121954e-06.
66/66 [==============================] - 10s 141ms/step - loss: 0.2000 - accuracy: 0.9214 - val_loss: 0.2559 - val_accuracy: 0.8921 - lr: 1.1765e-05
Epoch 30/1000
66/66 [==============================] - 11s 149ms/step - loss: 0.1958 - accuracy: 0.9290 - val_loss: 0.2488 - val_accuracy: 0.9079 - lr: 8.2354e-06
Epoch 31/1000
66/66 [==============================] - 11s 154ms/step - loss: 0.1933 - accuracy: 0.9238 - val_loss: 0.2400 - val_accuracy: 0.9032 - lr: 8.2354e-06
Epoch 32/1000
66/66 [==============================] - ETA: 0s - loss: 0.1892 - accuracy: 0.9290  
Epoch 32: ReduceLROnPlateau reducing learning rate to 5.764799698226852e-06.
66/66 [==============================] - 11s 149ms/step - loss: 0.1892 - accuracy: 0.9290 - val_loss: 0.2423 - val_accuracy: 0.8905 - lr: 8.2354e-06
Epoch 33/1000
66/66 [==============================] - 10s 140ms/step - loss: 0.1864 - accuracy: 0.9343 - val_loss: 0.2481 - val_accuracy: 0.9016 - lr: 5.7648e-06
Epoch 34/1000
66/66 [==============================] - ETA: 0s - loss: 0.1858 - accuracy: 0.9267  
Epoch 34: ReduceLROnPlateau reducing learning rate to 4.035359916088054e-06.
66/66 [==============================] - 10s 140ms/step - loss: 0.1858 - accuracy: 0.9267 - val_loss: 0.2496 - val_accuracy: 0.8984 - lr: 5.7648e-06
Epoch 35/1000
66/66 [==============================] - 10s 140ms/step - loss: 0.1833 - accuracy: 0.9376 - val_loss: 0.2466 - val_accuracy: 0.9032 - lr: 4.0354e-06
Epoch 36/1000
66/66 [==============================] - ETA: 0s - loss: 0.1862 - accuracy: 0.9281  
Epoch 36: ReduceLROnPlateau reducing learning rate to 2.8247518457646947e-06.
66/66 [==============================] - 10s 144ms/step - loss: 0.1862 - accuracy: 0.9281 - val_loss: 0.2435 - val_accuracy: 0.9063 - lr: 4.0354e-06
Epoch 37/1000
66/66 [==============================] - 10s 141ms/step - loss: 0.1810 - accuracy: 0.9295 - val_loss: 0.2435 - val_accuracy: 0.9095 - lr: 2.8248e-06
Epoch 38/1000
66/66 [==============================] - 10s 141ms/step - loss: 0.1775 - accuracy: 0.9400 - val_loss: 0.2419 - val_accuracy: 0.9032 - lr: 2.8248e-06
Epoch 39/1000
66/66 [==============================] - 10s 142ms/step - loss: 0.1782 - accuracy: 0.9367 - val_loss: 0.2412 - val_accuracy: 0.9127 - lr: 2.8248e-06
Epoch 40/1000
66/66 [==============================] - 10s 142ms/step - loss: 0.1762 - accuracy: 0.9371 - val_loss: 0.2392 - val_accuracy: 0.9143 - lr: 2.8248e-06
Epoch 41/1000
66/66 [==============================] - 10s 141ms/step - loss: 0.1753 - accuracy: 0.9429 - val_loss: 0.2401 - val_accuracy: 0.9175 - lr: 2.8248e-06
Epoch 42/1000
66/66 [==============================] - 10s 141ms/step - loss: 0.1745 - accuracy: 0.9457 - val_loss: 0.2406 - val_accuracy: 0.9159 - lr: 2.8248e-06
Epoch 43/1000
66/66 [==============================] - ETA: 0s - loss: 0.1769 - accuracy: 0.9343  
Epoch 43: ReduceLROnPlateau reducing learning rate to 1.9773262920352863e-06.
66/66 [==============================] - 10s 142ms/step - loss: 0.1769 - accuracy: 0.9343 - val_loss: 0.2385 - val_accuracy: 0.9143 - lr: 2.8248e-06
Epoch 44/1000
66/66 [==============================] - 10s 142ms/step - loss: 0.1776 - accuracy: 0.9405 - val_loss: 0.2374 - val_accuracy: 0.9159 - lr: 1.9773e-06
Epoch 45/1000
66/66 [==============================] - ETA: 0s - loss: 0.1741 - accuracy: 0.9448  
Epoch 45: ReduceLROnPlateau reducing learning rate to 1.3841284044247003e-06.
66/66 [==============================] - 10s 142ms/step - loss: 0.1741 - accuracy: 0.9448 - val_loss: 0.2351 - val_accuracy: 0.9159 - lr: 1.9773e-06
Epoch 46/1000
66/66 [==============================] - 10s 140ms/step - loss: 0.1742 - accuracy: 0.9352 - val_loss: 0.2378 - val_accuracy: 0.9159 - lr: 1.3841e-06
Epoch 47/1000
66/66 [==============================] - ETA: 0s - loss: 0.1750 - accuracy: 0.9443  
Epoch 47: ReduceLROnPlateau reducing learning rate to 9.68889867181133e-07.
66/66 [==============================] - 10s 140ms/step - loss: 0.1750 - accuracy: 0.9443 - val_loss: 0.2410 - val_accuracy: 0.9143 - lr: 1.3841e-06
Epoch 48/1000
66/66 [==============================] - 10s 141ms/step - loss: 0.1706 - accuracy: 0.9476 - val_loss: 0.2431 - val_accuracy: 0.9159 - lr: 9.6889e-07
Epoch 49/1000
66/66 [==============================] - ETA: 0s - loss: 0.1744 - accuracy: 0.9400  
Epoch 49: ReduceLROnPlateau reducing learning rate to 6.782228751944785e-07.
66/66 [==============================] - 10s 140ms/step - loss: 0.1744 - accuracy: 0.9400 - val_loss: 0.2440 - val_accuracy: 0.9143 - lr: 9.6889e-07
Epoch 50/1000
66/66 [==============================] - 10s 141ms/step - loss: 0.1721 - accuracy: 0.9443 - val_loss: 0.2433 - val_accuracy: 0.9143 - lr: 6.7822e-07
Epoch 51/1000
66/66 [==============================] - ETA: 0s - loss: 0.1701 - accuracy: 0.9457  
Epoch 51: ReduceLROnPlateau reducing learning rate to 4.7475601263613496e-07.
66/66 [==============================] - 10s 141ms/step - loss: 0.1701 - accuracy: 0.9457 - val_loss: 0.2431 - val_accuracy: 0.9095 - lr: 6.7822e-07
Epoch 52/1000
66/66 [==============================] - 10s 141ms/step - loss: 0.1694 - accuracy: 0.9438 - val_loss: 0.2420 - val_accuracy: 0.9095 - lr: 4.7476e-07
Epoch 53/1000
66/66 [==============================] - ETA: 0s - loss: 0.1698 - accuracy: 0.9419  
Epoch 53: ReduceLROnPlateau reducing learning rate to 3.323292048662551e-07.
66/66 [==============================] - 10s 141ms/step - loss: 0.1698 - accuracy: 0.9419 - val_loss: 0.2414 - val_accuracy: 0.9143 - lr: 4.7476e-07
Epoch 54/1000
66/66 [==============================] - 10s 141ms/step - loss: 0.1721 - accuracy: 0.9414 - val_loss: 0.2408 - val_accuracy: 0.9127 - lr: 3.3233e-07
Epoch 55/1000
66/66 [==============================] - ETA: 0s - loss: 0.1742 - accuracy: 0.9390  
Epoch 55: ReduceLROnPlateau reducing learning rate to 2.3263043544829997e-07.
66/66 [==============================] - 10s 141ms/step - loss: 0.1742 - accuracy: 0.9390 - val_loss: 0.2406 - val_accuracy: 0.9079 - lr: 3.3233e-07
Epoch 56/1000
66/66 [==============================] - 10s 141ms/step - loss: 0.1708 - accuracy: 0.9452 - val_loss: 0.2404 - val_accuracy: 0.9079 - lr: 2.3263e-07
Epoch 57/1000
66/66 [==============================] - ETA: 0s - loss: 0.1727 - accuracy: 0.9414  
Epoch 57: ReduceLROnPlateau reducing learning rate to 1.6284130879284928e-07.
66/66 [==============================] - 10s 141ms/step - loss: 0.1727 - accuracy: 0.9414 - val_loss: 0.2405 - val_accuracy: 0.9079 - lr: 2.3263e-07
Epoch 58/1000
66/66 [==============================] - 10s 142ms/step - loss: 0.1704 - accuracy: 0.9457 - val_loss: 0.2410 - val_accuracy: 0.9111 - lr: 1.6284e-07
Epoch 59/1000
66/66 [==============================] - ETA: 0s - loss: 0.1719 - accuracy: 0.9386  
Epoch 59: ReduceLROnPlateau reducing learning rate to 1.1398891217595519e-07.
66/66 [==============================] - 10s 141ms/step - loss: 0.1719 - accuracy: 0.9386 - val_loss: 0.2409 - val_accuracy: 0.9095 - lr: 1.6284e-07
Epoch 60/1000
66/66 [==============================] - 10s 144ms/step - loss: 0.1733 - accuracy: 0.9410 - val_loss: 0.2407 - val_accuracy: 0.9111 - lr: 1.1399e-07
Saving the model.
```
\
![](https://github.com/JueXuanChen/MyProjects/blob/main/Sperm_Morphology_Image_Data_Set/Sperm_class_model_Acc_Loss.png)


```
Model testing...
9/9 [==============================] - 2s 159ms/step
Number of predictions: 270
Number of true labels from df: 270

Class 0:
  Recall: 0.9783
  Specificity: 0.9438
  Precision: 0.9000
  F1 Score: 0.9375
Class 1:
  Recall: 0.8333
  Specificity: 0.9667
  Precision: 0.9259
  F1 Score: 0.8772
Class 2:
  Recall: 0.9432
  Specificity: 0.9670
  Precision: 0.9326
  F1 Score: 0.9379

Overall Accuracy: 0.9185
```
\
![](https://github.com/JueXuanChen/MyProjects/blob/main/Sperm_Morphology_Image_Data_Set/Sperm_class_model_CM.png)
