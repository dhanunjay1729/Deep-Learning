# Modified CNN for MRI Classification

I completed this four-class MRI image classification experiment as part of my Deep Learning coursework.

## What I did

Used progressive filters 32, 64, 128, 256, and 256, global average pooling, and dropout between dense layers.

I trained on 128 × 128 RGB images with augmentation, batch size 8, and a 50-epoch budget. The initial Adam learning rate was 0.0001, with He initialization and learning-rate reduction when validation loss stalled. I selected the checkpoint using validation loss and reserved the provided test images for evaluation.

## Results

| Measurement | Result |
|---|---:|
| Test accuracy | 70.91% |
| Test macro F1 | 0.7699 |
| Test balanced accuracy | 78.43% |
| Test cross-entropy loss | 0.6712 |
| Training accuracy without augmentation | 87.82% |
| Validation-selected epoch | 50 |
| Parameters | 1,019,908 |

The model completed 50 epochs on a Tesla T4 GPU. Partitions contain 8,192 training, 2,048 validation, and 1,279 test images. The test set is imbalanced; macro F1 and balanced accuracy complement overall accuracy.

## What I learned

Global average pooling summarizes each feature channel and reduces the dense classifier size. A smaller model still needs suitable optimization settings.

I used learning curves, a confusion matrix, class metrics, and prediction examples to examine the model beyond overall accuracy. The partitions are at image level; patient-level independence is not established.

[Notebook](modified_cnn.ipynb) · [Comparison of the three CNNs](../README.md)

![Architecture](figures/modified_cnn_architecture.png)
