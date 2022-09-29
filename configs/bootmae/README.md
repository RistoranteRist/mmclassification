# BootMAE

> [Bootstrapped Masked Autoencoders for Vision BERT Pretraining](https://arxiv.org/abs/2207.07116v1)

<!-- [ALGORITHM] -->

## Abstract

We propose bootstrapped masked autoencoders (BootMAE), a new approach for vision BERT pretraining. BootMAE improves the original masked autoencoders (MAE) with two core designs: 1) momentum encoder that provides online feature as extra BERT prediction targets; 2) target-aware decoder that tries to reduce the pressure on the encoder to memorize target-specific information in BERT pretraining. The first design is motivated by the observation that using a pretrained MAE to extract the features as the BERT prediction target for masked tokens can achieve better pretraining performance. Therefore, we add a momentum encoder in parallel with the original MAE encoder, which bootstraps the pretraining performance by using its own representation as the BERT prediction target. In the second design, we introduce target-specific information (e.g., pixel values of unmasked patches) from the encoder directly to the decoder to reduce the pressure on the encoder of memorizing the target-specific information. Thus, the encoder focuses on semantic modeling, which is the goal of BERT pretraining, and does not need to waste its capacity in memorizing the information of unmasked tokens related to the prediction target. Through extensive experiments, our BootMAE achieves 84.2% Top-1 accuracy on ImageNet-1K with ViT-B backbone, outperforming MAE by +0.8% under the same pre-training epochs. BootMAE also gets +1.0 mIoU improvements on semantic segmentation on ADE20K and +1.3 box AP, +1.4 mask AP improvement on object detection and segmentation on COCO dataset. Code is released at this https URL.

<div align=center>
<img src="https://user-images.githubusercontent.com/24734142/192964480-46726469-21d9-4e45-a06a-87c6a57c3367.png" width="90%"/>
</div>

## Results and models

### ImageNet-1k

|  Model   |   Pretrain   | resolution | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                   Config                    |                                           Download                                           |
| :------: | :----------: | :--------: | :-------: | :------: | :-------: | :-------: | :-----------------------------------------: | :------------------------------------------------------------------------------------------: |
| ViT-B\* | BootMAE |  224x224   |   86.56   |   17.58   |   83.86   |   96.76   |    [config](./vit-base-p16-16xb256-bootmae_in1k-224.py)    | [model]() |
| ViT-L\* | BootMAE |  224x224   |   304   |  61.60   |   85.58   |   97.48   |    [config](./swin-base_16xb64_in1k.py)     | [model]() |

*Models with * are converted from the [official repo](https://github.com/LightDXY/BootMAE). The config files of these models are only for validation. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*

## Citation

```
@article{dong2022ict,
  title={Bootstrapped Masked Autoencoders for Vision BERT Pretraining},
  author={Xiaoyi Dong, Jianmin Bao, Ting Zhang, Dongdong Chen, Weiming Zhang, Lu Yuan, Dong Chen, Fang Wen, Nenghai Yu},
  journal={arXiv preprint arXiv:2207.07116},
  year={2022}
}
```
