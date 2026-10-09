# Enhancing Garbage Segregation

Waste image classification with transfer learning (PyTorch / TensorFlow, Google Colab).
This repository collects the experiments behind our IEEE ICCCNT 2025 paper, *Enhancing Garbage Segregation Efficiency Using a Deep Learning-Based Multi-Class Classification Framework*, and a later extension on a larger dataset. The paper work was done by a team; co-authors are listed in the paper.

## Contents

| Folder | Notebook | Dataset | Model | Test result |
|---|---|---|---|---|
| `PAPER_WORK` | `Resnet50_paperCode.ipynb` | 6 classes, 2,527 images (as reported in the paper) | ResNet-50 | 93.54% accuracy (758 test images) |
| `PAPER_WORK` | `VGG16_MODEL.ipynb` | 7 classes, 2,751 images | VGG-16 | 58.33% accuracy (564 test images), frozen base |
| `EXTENSION_WORK` | `Resnet50_modified_12class.ipynb` | 12 classes, 15,493 images | ResNet-50 | 98.19% accuracy, macro-F1 0.977 (1,550 test images) |

The three experiments use different datasets, class sets and splits. Their results are not comparable with each other.

## Datasets

Datasets are not included in this repository (size and licensing). Update the dataset path in each notebook's data-loading cell to match your own copy.

- `Resnet50_paperCode.ipynb`: 6 classes (cardboard, glass, metal, paper, plastic, trash); expects the zip uploaded to Colab as `/content/archive (5).zip`.
- `VGG16_MODEL.ipynb`: 7 classes (cardboard, compost, glass, metal, paper, plastic, trash), `train/` and `test/` folders; expects `CAPSTONE_CNN` in Google Drive (`/content/drive/MyDrive/CAPSTONE_CNN`).
- `Resnet50_modified_12class.ipynb`: 12 classes (battery, biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash, white-glass); expects `garbage_dataset_12class.zip` at the root of Google Drive.

## Running

Open a notebook in Google Colab, select a GPU runtime, set the dataset path, and run all cells.

## Notes and limitations

- All results are from a single run on a single split, with no seed averaging. With a few hundred to ~1,500 test images, differences of a point or two are within noise.
- `Resnet50_paperCode.ipynb` is the code as submitted. Its transforms are assigned through a dataset object shared by the train, validation and test subsets, so the bilateral filter and augmentation were most likely not applied during training. The final explainability cell raises an error. The extension notebook builds separate train and evaluation datasets to avoid this.
- `VGG16_MODEL.ipynb` rescales inputs by 1/255 instead of using VGG-16's own `preprocess_input`. The fine-tuning stage did not converge (about 23% validation accuracy) and was not evaluated on the test set.
- `Resnet50_modified_12class.ipynb` uses a stratified 80/10/10 split. A near-duplicate check (32x32 correlation of at least 0.95) found roughly 2-3% of test images with a near-identical image in the training set, so the reported accuracy is likely slightly optimistic. The effect of the bilateral filter and augmentation was not ablated.
