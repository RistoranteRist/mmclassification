# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch

from mmcls.models.heads import ClsHead
from mmcls.registry import MODELS
from .text_encoder import CLIPTextEncoder


@MODELS.register_module()
class ZeroShotClsHead(ClsHead):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text_encoder = CLIPTextEncoder()

    def _build_class_vocabulary(self, class_vocabulary, prompt_prefix='a '):
        self.texts = [prompt_prefix + x for x in class_vocabulary]

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``ClsHead``, we just obtain the feature
        of the last stage.
        """
        # The ClsHead doesn't have other module, just return after unpacking.
        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        img_feats = self.pre_logits(feats)
        img_feats /= img_feats.norm(dim=-1, keepdim=True)

        text_feats = self.text_encoder(self.texts)
        text_feats /= text_feats.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * img_feats @ text_feats.T).softmax(dim=-1)

        return text_probs

    def loss(self, **kwargs) -> dict:
        raise NotImplementedError(
            'ZeroShotClsHead training hes not been supported yet.')
