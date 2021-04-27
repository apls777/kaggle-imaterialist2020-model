# iMaterialist 2020 - 1st Place Solution

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F853978%2F378129d2e9afb90bbd1f320858b9b73b%2Fimat2020_2.jpg?generation=1591441185889910&alt=media)

__Model architecture:__

- Mask R-CNN model
- SpineNet-143 + FPN backbone
- An extra head to classify attributes

__Training:__

- Pre-trained on the COCO dataset
- Image resolution: 1280x1280
- Focal loss for the attributes head
- Augmentations: random scaling (0.5x - 2.0x), v3 policy from the AutoAugment (modified to support masks)

All the changes were made on top of the [TPU Object Detection and Segmentation Framework](https://github.com/tensorflow/tpu/tree/master/models/official/detection).

Read more about the solution in the [Kaggle post](https://www.kaggle.com/c/imaterialist-fashion-2020-fgvc7/discussion/154306).

Download the model weights [here](https://drive.google.com/file/d/1bdC-LVj5_rJFfSiWWpeyQwKXYOLR-oNb/view?usp=sharing).

---

# Cloud TPUs #

This repository is a collection of reference models and tools used with
[Cloud TPUs](https://cloud.google.com/tpu/).

The fastest way to get started training a model on a Cloud TPU is by following
the tutorial. Click the button below to launch the tutorial using Google Cloud
Shell.

[![Open in Cloud Shell](http://gstatic.com/cloudssh/images/open-btn.svg)](https://console.cloud.google.com/cloudshell/open?git_repo=https%3A%2F%2Fgithub.com%2Ftensorflow%2Ftpu&page=shell&tutorial=tools%2Fctpu%2Ftutorial.md)

_Note:_ This repository is a public mirror, pull requests will not be accepted.
Please file an issue if you have a feature or bug request.

## Running Models

To run models in the `models` subdirectory, you may need to add the top-level
`/models` folder to the Python path with the command:

```
export PYTHONPATH="$PYTHONPATH:/path/to/models"
```
