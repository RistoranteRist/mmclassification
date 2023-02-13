# Google Universal Image Embedding

This project support models of Google Universal Image Embedding compeition in https://www.kaggle.com/competitions/google-universal-image-embedding/overview.

### Testing commands

```bash
mim test mmcls configs/vit-l-p14-336px-4th.py --checkpoint https://github.com/okotaku/clshub-weights/releases/download/v0.1.1guie/ViT-L-14-336-4th.pth
mim test mmcls configs/vit-b-p16-baseline.py --checkpoint https://github.com/okotaku/clshub-weights/releases/download/v0.1.1guie/clip-vit-base-p16_openai-pre_3rdparty_in1k.pth
```

## Results

### Zero-Shot inference for InShop Image Retrieval

|       Model        |                               URL                               | Recall@1 |                   Config                   |                               Download                                |
| :----------------: | :-------------------------------------------------------------: | :------: | :----------------------------------------: | :-------------------------------------------------------------------: |
| vit-b-p16-baseline |              [mmcls](../../configs/clip/README.md)              |  50.09   | [config](./configs/vit-b-p16-baseline.py)  | [model](https://github.com/okotaku/clshub-weights/releases/download/v0.1.1guie/clip-vit-base-p16_openai-pre_3rdparty_in1k.pth) \| [log](<>) |
|   vit-l-336-4th    | [4th repo](https://github.com/IvanAer/G-Universal-CLIP) [4th models](https://www.kaggle.com/datasets/ivanaerlic/guiemodels) |  67.37   | [config](./configs/vit-l-p14-336px-4th.py) | [model](https://github.com/okotaku/clshub-weights/releases/download/v0.1.1guie/ViT-L-14-336-4th.pth) \| [log](<>) |

*Models are converted. The config files of these models are only for inference. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*

## Checklist

Here is a checklist of this project's progress. And you can ignore this part if you don't plan to contribute
to MMClassification projects.

- [ ] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [ ] Finish the code

    <!-- The code's design shall follow existing interfaces and convention. For example, each model component should be registered into `mmcls.registry.MODELS` and configurable via a config file. -->

  - [ ] Basic docstrings & proper citation

    <!-- Each major class should contains a docstring, describing its functionality and arguments. If your code is copied or modified from other open-source projects, don't forget to cite the source project in docstring and make sure your behavior is not against its license. Typically, we do not accept any code snippet under GPL license. [A Short Guide to Open Source Licenses](https://medium.com/nationwide-technology/a-short-guide-to-open-source-licenses-cf5b1c329edd) -->

  - [ ] Converted checkpoint and results (Only for reproduction)

    <!-- If you are reproducing the result from a paper, make sure the model in the project can match that results. Also please provide checkpoint links or a checkpoint conversion script for others to get the pre-trained model. -->

- [ ] Milestone 2: Indicates a successful model implementation.

  - [ ] Training results

    <!-- If you are reproducing the result from a paper, train your model from scratch and verified that the final result can match the original result. Usually, ±0.1% is acceptable for the image classification task on ImageNet-1k. -->

- [ ] Milestone 3: Good to be a part of our core package!

  - [ ] Unit tests

    <!-- Unit tests for the major module are required. [Example](https://github.com/open-mmlab/mmclassification/blob/1.x/tests/test_models/test_backbones/test_vision_transformer.py) -->

  - [ ] Code style

    <!-- Refactor your code according to reviewer's comment. -->

  - [ ] `metafile.yml` and `README.md`

    <!-- It will used for MMClassification to acquire your models. [Example](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/mvit/metafile.yml). In particular, you may have to refactor this README into a standard one. [Example](https://github.com/open-mmlab/mmclassification/blob/1.x/configs/swin_transformer/README.md) -->
