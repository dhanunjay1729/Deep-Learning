# Basic CNN for MRI Classification

I completed this four-class MRI image classification experiment as part of my Deep Learning coursework.

## What I did

Implemented five convolution and pooling blocks with filters 256, 64, 256, 128, and 256, followed by a flattened dense classifier.

I trained on 128 × 128 RGB images with augmentation, batch size 8, and a 50-epoch budget. The initial Adam learning rate was 0.0003, with He initialization and learning-rate reduction when validation loss stalled. I selected the checkpoint using validation loss and reserved the provided test images for evaluation.

## Results

| Measurement | Result |
|---|---:|
| Test accuracy | 70.84% |
| Test macro F1 | 0.7365 |
| Test balanced accuracy | 71.54% |
| Test cross-entropy loss | 0.6473 |
| Training accuracy without augmentation | 85.62% |
| Validation-selected epoch | 37 |
| Parameters | 1,425,540 |

The model completed 50 epochs on a Tesla T4 GPU. Partitions contain 8,192 training, 2,048 validation, and 1,279 test images. The test set is imbalanced; macro F1 and balanced accuracy complement overall accuracy.

## What I learned

A large flattened classifier increases the parameter count. Initialization and learning rate matter: the earlier one-class predictions were an optimization failure.

I used learning curves, a confusion matrix, class metrics, and prediction examples to examine the model beyond overall accuracy. The partitions are at image level; patient-level independence is not established.

[Notebook](basic_cnn.ipynb) · [Comparison of the three CNNs](../README.md)

![Architecture](figures/basic_cnn_architecture.png)
