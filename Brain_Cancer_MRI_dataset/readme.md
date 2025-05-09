# Brain Cancer - MRI dataset

## Description
* Using Brain Cancer - MRI dataset to fine tuning Inception_v3 model for classifying brain glioma, menin and tumor MRI images
* **Overall accuracy for testing up to 97.62%**

## Data
* Data link: https://www.kaggle.com/datasets/orvile/brain-cancer-mri-dataset/data
  
* 3 classes:
    - Brain_Glioma: 2004 images
    - Brain_Menin: 2004 images
    - Brain_Tumor: 2048 images

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
133/133 [==============================] - 42s 187ms/step - loss: 0.3407 - accuracy: 0.8971 - val_loss: 1.9498 - val_accuracy: 0.4422 - lr: 1.0000e-04
Epoch 2/1000
133/133 [==============================] - 21s 146ms/step - loss: 0.3738 - accuracy: 0.9387 - val_loss: 10.4745 - val_accuracy: 0.3423 - lr: 1.0000e-04
Epoch 3/1000
133/133 [==============================] - 20s 144ms/step - loss: 0.4804 - accuracy: 0.9547 - val_loss: 2.2737 - val_accuracy: 0.6192 - lr: 1.0000e-04
Epoch 4/1000
133/133 [==============================] - 20s 144ms/step - loss: 0.4789 - accuracy: 0.9573 - val_loss: 1.4119 - val_accuracy: 0.7695 - lr: 1.0000e-04
Epoch 5/1000
133/133 [==============================] - 21s 146ms/step - loss: 0.3694 - accuracy: 0.9611 - val_loss: 0.6853 - val_accuracy: 0.8332 - lr: 1.0000e-04
Epoch 6/1000
133/133 [==============================] - 20s 140ms/step - loss: 0.2666 - accuracy: 0.9656 - val_loss: 0.8700 - val_accuracy: 0.7899 - lr: 1.0000e-04
Epoch 7/1000
133/133 [==============================] - 20s 144ms/step - loss: 0.2069 - accuracy: 0.9696 - val_loss: 3.3315 - val_accuracy: 0.6900 - lr: 1.0000e-04
Epoch 8/1000
133/133 [==============================] - 20s 143ms/step - loss: 0.1739 - accuracy: 0.9717 - val_loss: 0.2868 - val_accuracy: 0.9221 - lr: 1.0000e-04
Epoch 9/1000
133/133 [==============================] - 20s 139ms/step - loss: 0.1614 - accuracy: 0.9670 - val_loss: 1.2590 - val_accuracy: 0.7781 - lr: 1.0000e-04
Epoch 10/1000
133/133 [==============================] - 20s 140ms/step - loss: 0.1545 - accuracy: 0.9705 - val_loss: 0.1885 - val_accuracy: 0.9512 - lr: 1.0000e-04
Epoch 11/1000
133/133 [==============================] - 20s 140ms/step - loss: 0.1452 - accuracy: 0.9757 - val_loss: 0.3727 - val_accuracy: 0.7671 - lr: 1.0000e-04
Epoch 12/1000
133/133 [==============================] - 20s 139ms/step - loss: 0.1287 - accuracy: 0.9757 - val_loss: 1.1931 - val_accuracy: 0.5720 - lr: 1.0000e-04
Epoch 13/1000
133/133 [==============================] - ETA: 0s - loss: 0.1188 - accuracy: 0.9731  
Epoch 13: ReduceLROnPlateau reducing learning rate to 2.9999999242136255e-05.
133/133 [==============================] - 20s 140ms/step - loss: 0.1188 - accuracy: 0.9731 - val_loss: 0.5890 - val_accuracy: 0.7443 - lr: 1.0000e-04
Epoch 14/1000
133/133 [==============================] - 20s 140ms/step - loss: 0.1097 - accuracy: 0.9785 - val_loss: 0.1316 - val_accuracy: 0.9520 - lr: 3.0000e-05
Epoch 15/1000
133/133 [==============================] - 20s 141ms/step - loss: 0.0973 - accuracy: 0.9825 - val_loss: 0.1739 - val_accuracy: 0.9268 - lr: 3.0000e-05
Epoch 16/1000
133/133 [==============================] - 20s 140ms/step - loss: 0.0906 - accuracy: 0.9863 - val_loss: 0.1490 - val_accuracy: 0.9426 - lr: 3.0000e-05
Epoch 17/1000
133/133 [==============================] - 20s 140ms/step - loss: 0.0822 - accuracy: 0.9873 - val_loss: 0.1037 - val_accuracy: 0.9701 - lr: 3.0000e-05
Epoch 18/1000
133/133 [==============================] - 20s 140ms/step - loss: 0.0753 - accuracy: 0.9920 - val_loss: 0.1553 - val_accuracy: 0.9072 - lr: 3.0000e-05
Epoch 19/1000
133/133 [==============================] - 20s 140ms/step - loss: 0.0733 - accuracy: 0.9915 - val_loss: 0.1140 - val_accuracy: 0.9544 - lr: 3.0000e-05
Epoch 20/1000
133/133 [==============================] - ETA: 0s - loss: 0.0701 - accuracy: 0.9901  
Epoch 20: ReduceLROnPlateau reducing learning rate to 8.999999772640877e-06.
133/133 [==============================] - 20s 141ms/step - loss: 0.0701 - accuracy: 0.9901 - val_loss: 0.1948 - val_accuracy: 0.9024 - lr: 3.0000e-05
Epoch 21/1000
133/133 [==============================] - 20s 139ms/step - loss: 0.0658 - accuracy: 0.9943 - val_loss: 0.1319 - val_accuracy: 0.9434 - lr: 9.0000e-06
Epoch 22/1000
133/133 [==============================] - 20s 141ms/step - loss: 0.0629 - accuracy: 0.9943 - val_loss: 0.2002 - val_accuracy: 0.8875 - lr: 9.0000e-06
Epoch 23/1000
133/133 [==============================] - ETA: 0s - loss: 0.0606 - accuracy: 0.9950  
Epoch 23: ReduceLROnPlateau reducing learning rate to 2.6999998226528985e-06.
133/133 [==============================] - 20s 141ms/step - loss: 0.0606 - accuracy: 0.9950 - val_loss: 0.1384 - val_accuracy: 0.9441 - lr: 9.0000e-06
Epoch 24/1000
133/133 [==============================] - 20s 140ms/step - loss: 0.0576 - accuracy: 0.9983 - val_loss: 0.1454 - val_accuracy: 0.9410 - lr: 2.7000e-06
Epoch 25/1000
133/133 [==============================] - 20s 140ms/step - loss: 0.0617 - accuracy: 0.9972 - val_loss: 0.1390 - val_accuracy: 0.9457 - lr: 2.7000e-06
Epoch 26/1000
133/133 [==============================] - ETA: 0s - loss: 0.0579 - accuracy: 0.9972  
Epoch 26: ReduceLROnPlateau reducing learning rate to 8.099999604382901e-07.
133/133 [==============================] - 20s 142ms/step - loss: 0.0579 - accuracy: 0.9972 - val_loss: 0.1422 - val_accuracy: 0.9457 - lr: 2.7000e-06
Epoch 27/1000
133/133 [==============================] - 20s 144ms/step - loss: 0.0577 - accuracy: 0.9967 - val_loss: 0.1466 - val_accuracy: 0.9426 - lr: 8.1000e-07

Saving the model.
```
\
![](https://github.com/JueXuanChen/MyProjects/blob/main/Brain_Cancer_MRI_dataset/Brain_MRI_model_Acc_loss.png)

```
Model testing...
18/18 [==============================] - 2s 71ms/step
Number of predictions: 546
Number of true labels from df: 546

Class 0:
  Recall: 0.9834
  Specificity: 0.9890
  Precision: 0.9780
  F1 Score: 0.9807
Class 1:
  Recall: 0.9500
  Specificity: 0.9973
  Precision: 0.9942
  F1 Score: 0.9716
Class 2:
  Recall: 0.9946
  Specificity: 0.9778
  Precision: 0.9583
  F1 Score: 0.9761

Overall Accuracy: 0.9762
```
\
![](https://github.com/JueXuanChen/MyProjects/blob/main/Brain_Cancer_MRI_dataset/Brain_MRI_model_CM.png)
