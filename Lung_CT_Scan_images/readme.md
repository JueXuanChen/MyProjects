# Lung CT-Scan images
 
## Description
* Using Lung CT-Scan images dataset to fine tuning Inception_v3 model for classifying cancerous and non-cancerous images
* **Overall accuracy for testing up to 100.00%**

## Data
* Data link: https://www.kaggle.com/datasets/orvile/ct-scan-images/data
  
* 2 classes:
    - Cancerous: 238 images from lung cancer patients
    - Non-Cancerous: 126 images from patients with COVID-19 or other lung conditions

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
16/16 [==============================] - 22s 439ms/step - loss: 0.3032 - accuracy: 0.8701 - val_loss: 1.3835 - val_accuracy: 0.6623 - lr: 1.0000e-04
Epoch 2/1000
16/16 [==============================] - 2s 90ms/step - loss: 0.2495 - accuracy: 0.9724 - val_loss: 0.4285 - val_accuracy: 0.9481 - lr: 1.0000e-04
Epoch 3/1000
16/16 [==============================] - ETA: 0s - loss: 0.3605 - accuracy: 0.9173
Epoch 3: ReduceLROnPlateau reducing learning rate to 1.9999999494757503e-05.
16/16 [==============================] - 2s 85ms/step - loss: 0.3605 - accuracy: 0.9173 - val_loss: 4.0211 - val_accuracy: 0.6494 - lr: 1.0000e-04
Epoch 4/1000
16/16 [==============================] - ETA: 0s - loss: 0.2405 - accuracy: 0.9843
Epoch 4: ReduceLROnPlateau reducing learning rate to 3.999999898951501e-06.
16/16 [==============================] - 2s 83ms/step - loss: 0.2405 - accuracy: 0.9843 - val_loss: 1.8220 - val_accuracy: 0.6494 - lr: 2.0000e-05
Epoch 5/1000
16/16 [==============================] - ETA: 0s - loss: 0.2339 - accuracy: 1.0000
Epoch 5: ReduceLROnPlateau reducing learning rate to 7.999999979801942e-07.
16/16 [==============================] - 2s 80ms/step - loss: 0.2339 - accuracy: 1.0000 - val_loss: 0.7596 - val_accuracy: 0.7143 - lr: 4.0000e-06
Epoch 6/1000
16/16 [==============================] - ETA: 0s - loss: 0.2363 - accuracy: 0.9921
Epoch 6: ReduceLROnPlateau reducing learning rate to 1.600000018697756e-07.
16/16 [==============================] - 2s 107ms/step - loss: 0.2363 - accuracy: 0.9921 - val_loss: 0.3341 - val_accuracy: 0.9091 - lr: 8.0000e-07
Epoch 7/1000
16/16 [==============================] - 2s 83ms/step - loss: 0.2345 - accuracy: 0.9961 - val_loss: 0.2545 - val_accuracy: 0.9740 - lr: 1.6000e-07
Epoch 8/1000
16/16 [==============================] - 2s 86ms/step - loss: 0.2464 - accuracy: 0.9921 - val_loss: 0.2408 - val_accuracy: 0.9870 - lr: 1.6000e-07
Epoch 9/1000
16/16 [==============================] - ETA: 0s - loss: 0.2467 - accuracy: 0.9764
Epoch 9: ReduceLROnPlateau reducing learning rate to 3.199999980552093e-08.
16/16 [==============================] - 2s 83ms/step - loss: 0.2467 - accuracy: 0.9764 - val_loss: 0.2366 - val_accuracy: 0.9870 - lr: 1.6000e-07
Epoch 10/1000
16/16 [==============================] - ETA: 0s - loss: 0.2427 - accuracy: 0.9882
Epoch 10: ReduceLROnPlateau reducing learning rate to 1e-08.
16/16 [==============================] - 2s 84ms/step - loss: 0.2427 - accuracy: 0.9882 - val_loss: 0.2352 - val_accuracy: 0.9870 - lr: 3.2000e-08
Epoch 11/1000
16/16 [==============================] - 2s 87ms/step - loss: 0.2354 - accuracy: 0.9961 - val_loss: 0.2342 - val_accuracy: 1.0000 - lr: 1.0000e-08
Epoch 12/1000
16/16 [==============================] - 2s 84ms/step - loss: 0.2340 - accuracy: 0.9921 - val_loss: 0.2336 - val_accuracy: 1.0000 - lr: 1.0000e-08
Epoch 13/1000
16/16 [==============================] - 2s 85ms/step - loss: 0.2359 - accuracy: 0.9921 - val_loss: 0.2332 - val_accuracy: 1.0000 - lr: 1.0000e-08
Epoch 14/1000
16/16 [==============================] - 2s 85ms/step - loss: 0.2370 - accuracy: 0.9843 - val_loss: 0.2330 - val_accuracy: 1.0000 - lr: 1.0000e-08
Epoch 15/1000
16/16 [==============================] - 2s 85ms/step - loss: 0.2330 - accuracy: 1.0000 - val_loss: 0.2330 - val_accuracy: 1.0000 - lr: 1.0000e-08
Epoch 16/1000
16/16 [==============================] - 2s 85ms/step - loss: 0.2334 - accuracy: 0.9961 - val_loss: 0.2330 - val_accuracy: 1.0000 - lr: 1.0000e-08
Epoch 17/1000
16/16 [==============================] - 2s 82ms/step - loss: 0.2369 - accuracy: 0.9882 - val_loss: 0.2331 - val_accuracy: 1.0000 - lr: 1.0000e-08
Epoch 18/1000
16/16 [==============================] - 2s 80ms/step - loss: 0.2552 - accuracy: 0.9803 - val_loss: 0.2332 - val_accuracy: 1.0000 - lr: 1.0000e-08
Epoch 19/1000
16/16 [==============================] - 2s 84ms/step - loss: 0.2412 - accuracy: 0.9843 - val_loss: 0.2333 - val_accuracy: 1.0000 - lr: 1.0000e-08
Epoch 20/1000
16/16 [==============================] - 2s 81ms/step - loss: 0.2357 - accuracy: 0.9961 - val_loss: 0.2334 - val_accuracy: 1.0000 - lr: 1.0000e-08
Epoch 21/1000
16/16 [==============================] - 2s 81ms/step - loss: 0.2720 - accuracy: 0.9724 - val_loss: 0.2337 - val_accuracy: 1.0000 - lr: 1.0000e-08
Epoch 22/1000
16/16 [==============================] - 2s 83ms/step - loss: 0.2627 - accuracy: 0.9764 - val_loss: 0.2337 - val_accuracy: 1.0000 - lr: 1.0000e-08
Epoch 23/1000
16/16 [==============================] - 2s 82ms/step - loss: 0.2380 - accuracy: 0.9921 - val_loss: 0.2338 - val_accuracy: 1.0000 - lr: 1.0000e-08
Epoch 24/1000
16/16 [==============================] - 2s 78ms/step - loss: 0.2424 - accuracy: 0.9843 - val_loss: 0.2338 - val_accuracy: 1.0000 - lr: 1.0000e-08
Epoch 25/1000
16/16 [==============================] - 2s 80ms/step - loss: 0.2405 - accuracy: 0.9882 - val_loss: 0.2339 - val_accuracy: 1.0000 - lr: 1.0000e-08
Epoch 26/1000
16/16 [==============================] - 2s 87ms/step - loss: 0.2908 - accuracy: 0.9882 - val_loss: 0.2340 - val_accuracy: 1.0000 - lr: 1.0000e-08
Saving the model.
```
\
![](https://github.com/JueXuanChen/MyProjects/blob/main/Lung_CT_Scan_images/Lung_CT_scan_image_Acc_loss.png)

```
Model testing...
3/3 [==============================] - 1s 396ms/step
Number of predictions: 33
Number of true labels from df: 33

Class 0:
  Recall: 1.0000
  Specificity: 1.0000
  Precision: 1.0000
  F1 Score: 1.0000
Class 1:
  Recall: 1.0000
  Specificity: 1.0000
  Precision: 1.0000
  F1 Score: 1.0000

Overall Accuracy: 1.0000
```
\
![](https://github.com/JueXuanChen/MyProjects/blob/main/Lung_CT_Scan_images/Lung_CT_scan_image_CM.png)
