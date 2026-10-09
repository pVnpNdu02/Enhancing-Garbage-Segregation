# EXTENSION_WORK

`Resnet50_modified_12class.ipynb`: ResNet-50 with a modified classification head on a 12-class garbage dataset (15,493 images), trained with a bilateral filter and flip/rotation augmentation on the training split only.

- Split: stratified 80/10/10 (12,394 train, 1,549 validation, 1,550 test).
- Result: 98.19% test accuracy, macro-F1 0.977 (single run).
- The notebook expects `garbage_dataset_12class.zip` at the root of Google Drive; change `ZIP_PATH` in the data-extraction cell if yours is elsewhere.

The dataset link: https://www.kaggle.com/datasets/mostafaabla/garbage-classification?resource=download
