
# Dynamic Patch Selection for Vision Transformers

## Overview

This repository contains an implementation of **Dynamic Patch Selection** for Vision Transformers (ViTs), a novel mechanism that allows a ViT to dynamically focus on specific regions of an image. The goal is to enable more effective learning by selecting patches from varying locations on each input image while embedding these patches' spatial information into the model's positional encodings. This repository also includes a comparison with the standard Vision Transformer, where both models are tested on the CIFAR-10 dataset under identical conditions.

### Key Idea: Dynamic Patch Selection

In a standard Vision Transformer, an image is uniformly divided into patches, and each patch is treated equally with a fixed, learnable positional embedding. However, this approach can be suboptimal because it treats all regions of the image equally, even though different areas may have varying importance for a task.

**Dynamic Patch Selection** introduces a learnable crop function, implemented using a Spatial Transformer Network (STN), that selects patches dynamically from the image. This method involves the following steps:
- **Affine Transformation**: The STN generates translation parameters that define an affine transformation for each patch. The parameters are bounded by a tanh activation, ensuring the patches remain within the image.
- **Patch Sampling**: The transformation is applied to sample patches from different parts of the image. The selection varies across images and training steps, acting as built-in regularization.
- **Positional Encodings**: Unlike standard ViTs with fixed positional embeddings, here, the translation parameters are directly embedded into the positional encodings, ensuring the positional information aligns with the dynamic patches.

## Model Comparison

### Dynamic Patch Selection ViT

The dynamic patch selection model utilizes the STN for flexible patch selection and directly computes positional embeddings from the translation parameters. This flexibility allows the model to adapt to various image regions while maintaining spatial information.

### Standard Vision Transformer

For comparison, we also provide a standard Vision Transformer model that uses a fixed grid of patches with learned positional embeddings. The comparison focuses on accuracy, training convergence, and the model's ability to generalize using the CIFAR-10 dataset.

## Results

On the CIFAR-10 dataset (without any pretraining), **Dynamic Patch Selection** achieved a final accuracy of **77%**, whereas the standard Vision Transformer achieved **74%** under the same conditions and with identical hyperparameters and model size. The core difference is the patch selection mechanism and how positional embeddings are handled.

- **Dynamic Patch Selection**: 77% accuracy
- **Standard ViT**: 74% accuracy

The results suggest that dynamic patch selection improves accuracy by a **flat 3%**, likely due to the more adaptive patch selection and the built-in regularization that comes from sampling patches from different regions of the image at each step.

## How to Run

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- torchvision
- tqdm

You can install the required dependencies using:
```bash
pip install -r requirements.txt
```

### Training the Models

To train the dynamic patch selection ViT on CIFAR-10:
```bash
python dps_vit_cifar10.py
```

To train the standard Vision Transformer on CIFAR-10:
```bash
python std_vit_cifar10.py
```

Both scripts will train the respective models and save the best-performing weights.

### Evaluation

The evaluation functions are built into the training scripts and will report the test accuracy after every epoch. The models will save the best weights during training:
- `dps_vit_cifar10.pth` for the dynamic patch selection model.
- `std_vit_cifar10.pth` for the standard ViT.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Alec Fessler** © 2024
- Special thanks to the PyTorch community and the open-source deep learning ecosystem for tools and inspiration.
