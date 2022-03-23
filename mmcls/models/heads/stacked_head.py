# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.runner import BaseModule, ModuleList
from mmcv.utils import digit_version

from ..builder import HEADS
from .cls_head import ClsHead


class LinearBlock(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout_rate=0.,
                 norm_cfg=None,
                 act_cfg=None,
                 lazy_linear=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.lazy_linear = lazy_linear
        if self.lazy_linear:
            self.fc = nn.LazyLinear(out_channels)
        else:
            self.fc = nn.Linear(in_channels, out_channels)

        self.norm = None
        self.act = None
        self.dropout = None

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        if act_cfg is not None:
            self.act = build_activation_layer(act_cfg)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.fc(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


@HEADS.register_module()
class StackedLinearClsHead(ClsHead):
    """Classifier head with several hidden fc layer and a output fc layer.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        mid_channels (Sequence): Number of channels in the hidden fc layers.
        dropout_rate (float): Dropout rate after each hidden fc layer,
            except the last layer. Defaults to 0.
        norm_cfg (dict, optional): Config dict of normalization layer after
            each hidden fc layer, except the last layer. Defaults to None.
        act_cfg (dict, optional): Config dict of activation function after each
            hidden layer, except the last layer. Defaults to use "ReLU".
        lazy_linear (bool): If True, the fc layer use nn.LazyLinear.
            Defaults to False.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 mid_channels: Sequence,
                 dropout_rate: float = 0.,
                 norm_cfg: Dict = None,
                 act_cfg: Dict = dict(type='ReLU'),
                 lazy_linear=False,
                 **kwargs):
        super(StackedLinearClsHead, self).__init__(**kwargs)
        if lazy_linear:
            if digit_version(torch.__version__) < digit_version('1.8.0'):
                raise RuntimeError(
                    'torch.nn.LazyLinear is not available before 1.8.0')
            warnings.warn(
                'For StackedLinearClsHead with lazy_linear=True, '
                f'in_channels={in_channels} and init_cfg={self.init_cfg} '
                f'is ignored and calculated automatically.')

        assert num_classes > 0, \
            f'`num_classes` of StackedLinearClsHead must be a positive ' \
            f'integer, got {num_classes} instead.'
        self.num_classes = num_classes

        self.in_channels = in_channels

        assert isinstance(mid_channels, Sequence), \
            f'`mid_channels` of StackedLinearClsHead should be a sequence, ' \
            f'instead of {type(mid_channels)}'
        self.mid_channels = mid_channels

        self.dropout_rate = dropout_rate
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.lazy_linear = lazy_linear

        self._init_layers()

    def _init_layers(self):
        self.layers = ModuleList()
        in_channels = self.in_channels
        for hidden_channels in self.mid_channels:
            self.layers.append(
                LinearBlock(
                    in_channels,
                    hidden_channels,
                    dropout_rate=self.dropout_rate,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    lazy_linear=self.lazy_linear))
            in_channels = hidden_channels

        self.layers.append(
            LinearBlock(
                self.mid_channels[-1],
                self.num_classes,
                dropout_rate=0.,
                norm_cfg=None,
                act_cfg=None,
                lazy_linear=self.lazy_linear))

    def init_weights(self):
        if not self.lazy_linear:
            self.layers.init_weights()

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        for layer in self.layers[:-1]:
            x = layer(x)
        return x

    @property
    def fc(self):
        return self.layers[-1]

    def simple_test(self, x, softmax=True, post_process=True):
        """Inference without augmentation.

        Args:
            x (tuple[Tensor]): The input features.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. The shape of every item should be
                ``(num_samples, in_channels)``.
            softmax (bool): Whether to softmax the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
        x = self.pre_logits(x)
        cls_score = self.fc(x)

        if softmax:
            pred = (
                F.softmax(cls_score, dim=1) if cls_score is not None else None)
        else:
            pred = cls_score

        if post_process:
            return self.post_process(pred)
        else:
            return pred

    def forward_train(self, x, gt_label, **kwargs):
        x = self.pre_logits(x)
        cls_score = self.fc(x)
        losses = self.loss(cls_score, gt_label, **kwargs)
        return losses
