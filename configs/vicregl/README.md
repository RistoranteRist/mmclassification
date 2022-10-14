# VICRegL

> [VICRegL: Self-Supervised Learning of Local Visual Features](https://arxiv.org/abs/2210.01571)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Most recent self-supervised methods for learning image representations focus on either producing a global feature with invariance properties, or producing a set of local features. The former works best for classification tasks while the latter is best for detection and segmentation tasks. This paper explores the fundamental trade-off between learning local and global features. A new method called VICRegL is proposed that learns good global and local features simultaneously, yielding excellent performance on detection and segmentation tasks while maintaining good performance on classification tasks. Concretely, two identical branches of a standard convolutional net architecture are fed two differently distorted versions of the same image. The VICReg criterion is applied to pairs of global feature vectors. Simultaneously, the VICReg criterion is applied to pairs of local feature vectors occurring before the last pooling layer. Two local feature vectors are attracted to each other if their l2-distance is below a threshold or if their relative locations are consistent with a known geometric transformation between the two input images. We demonstrate strong performance on linear classification and segmentation transfer tasks.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/24734142/195762250-8b8911a0-6963-4fa2-8527-b67c3bf9ff99.png" width="100%"/>
</div>

## Results and models

### Pre-trained Models

The pre-trained models on ImageNet-1k or ImageNet-21k are used to fine-tune on the downstream tasks, and therefore don't have evaluation results.

|     Model     | Training Data | alpha |                                                               Download                                                                |
| :-----------: | :-----------: | :-------: |:-----------------------------------------------------------------------------------------------------------------------------------: |
| ResNet-50  |  ImageNet-1k  |   0.9   | [model](https://dl.fbaipublicfiles.com/vicregl/resnet50_alpha0.9.pth) |
| ResNet-50  |  ImageNet-1k  |   0.75   | [model](https://dl.fbaipublicfiles.com/vicregl/resnet50_alpha0.75.pth) |
| ConvNeXt-S\*  |  ImageNet-1k  |   0.9   | [model]() |
| ConvNeXt-S\*  |  ImageNet-1k  |   0.75   | [model]() |
| ConvNeXt-B\*  |  ImageNet-1k  |   0.9   | [model]()  |
| ConvNeXt-B\*  |  ImageNet-1k  |   0.75   | [model]()  |
| ConvNeXt-XL\* | ImageNet-21k  |   0.75   |       [model]()       |

*Models with * are converted from the [official repo](https://github.com/facebookresearch/vicregl). Not full ckpt but backbone ckpt are converted.*

### CUB-200-2011

|   Model   |                           Pretrain                            | resolution | Params(M) | Flops(G) | Top-1 (%) |              Config              |                            Download                            |
| :-------: | :-----------------------------------------------------------: | :--------: | :-------: | :------: | :-------: | :------------------------------: | :------------------------------------------------------------: |
| ResNet-50 | [VICRegL-alpha0.9](https://dl.fbaipublicfiles.com/vicregl/resnet50_alpha0.9.pth) |  448x448   |   23.92   |  16.48   |      | [config](./vicregl_resnet50_8xb8_cub.py) | [model]() \| [log]() |

## Citation

```bibtex
@inproceedings{bardes2022vicregl,
  author  = {Adrien Bardes and Jean Ponce and Yann LeCun},
  title   = {VICRegL: Self-Supervised Learning of Local Visual Features},
  booktitle = {NeurIPS},
  year    = {2022},
}
```
