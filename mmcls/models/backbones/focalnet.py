# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Sequence

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (Conv2d, ConvModule, build_activation_layer,
                      build_conv_layer, build_norm_layer)
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import FFN, AdaptivePadding, PatchEmbed
from mmengine.logging import MMLogger
from mmengine.model import BaseModule, ModuleList
from mmengine.runner.checkpoint import CheckpointLoader
from mmengine.utils import to_2tuple
from torch.nn import functional as F

from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.registry import MODELS
from ..utils import LayerScale


class FocalModulation(BaseModule):
    """Focal Modulation.

    Args:
        embed_dims (int): The feature dimension
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at focal level 1
        focal_factor (int, default=2): Step to increase the focal window
        use_postln (bool, default=False): Whether use post-modulation layernorm
    """

    def __init__(self,
                 embed_dims,
                 drop_rate=0.,
                 focal_level=2,
                 focal_window=7,
                 focal_factor=2,
                 act_cfg=dict(type='GELU'),
                 normalize_modulator=False,
                 norm_cfg=None,
                 init_cfg=None):

        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.normalize_modulator = normalize_modulator

        # specific args for focalv3
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.focal_factor = focal_factor

        self.f = nn.Linear(
            embed_dims, 2 * embed_dims + (self.focal_level + 1), bias=True)
        self.h = Conv2d(
            embed_dims, embed_dims, 1, stride=1, padding=0, groups=1)

        self.act = build_activation_layer(act_cfg)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(drop_rate)
        self.focal_layers = ModuleList()

        if norm_cfg is not None:
            self.ln = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.ln = None

        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(
                ConvModule(
                    embed_dims,
                    embed_dims,
                    kernel_size,
                    stride=1,
                    groups=embed_dims,
                    padding=kernel_size // 2,
                    bias=False,
                    act_cfg=act_cfg))

    def forward(self, x):
        C = x.shape[-1]
        x = self.f(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        q, ctx, gates = torch.split(x, (C, C, self.focal_level + 1), 1)

        ctx_all = 0
        for level in range(self.focal_level):
            ctx = self.focal_layers[level](ctx)
            ctx_all = ctx_all + ctx * gates[:, level:level + 1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global * gates[:, self.focal_level:]

        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)

        x_out = q * self.h(ctx_all)
        x_out = x_out.permute(0, 2, 3, 1).contiguous()
        x_out = self.ln(x_out) if self.ln else x_out
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)

        return x_out


class FocalModulationBlock(BaseModule):
    """Focal Modulation Block.

    Args:
        embed_dims (int): The feature dimension
        ffn_ratio (float): Ratio of ffn hidden dim to embedding dim.
        drop_rate (float): Dropout rate. Defaults to 0.0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        focal_level (int): number of focal levels. Defaults to 2.
        focal_window (int): focal kernel size at level 1. Defaults to 9.
        act_cfg (dict): The activation config for FFNs.
            Defaluts to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 ffn_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 focal_level=2,
                 focal_window=9,
                 use_layer_scale=False,
                 use_postln=False,
                 normalize_modulator=False,
                 use_postln_in_modulation=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.ffn_ratio = ffn_ratio
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.with_cp = with_cp
        self.use_postln = use_postln

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        self.modulation = FocalModulation(
            embed_dims,
            focal_window=self.focal_window,
            focal_level=self.focal_level,
            drop_rate=drop_rate,
            normalize_modulator=normalize_modulator,
            norm_cfg=dict(type='LN') if use_postln_in_modulation else None)

        dropout_layer = dict(type='DropPath', drop_prob=drop_path_rate)
        self.drop_path = build_dropout(
            dropout_layer) if drop_path_rate > 0 else nn.Identity()

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        mlp_hidden_dim = int(embed_dims * ffn_ratio)
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=mlp_hidden_dim,
            act_cfg=act_cfg,
            ffn_drop=drop_rate,
            add_identity=False)

        if use_layer_scale:
            self.gamma1 = LayerScale(embed_dims)
            self.gamma2 = LayerScale(embed_dims)
        else:
            self.gamma1, self.gamma2 = nn.Identity(), nn.Identity()

    def forward(self, x, hw_shape):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        def _inner_forward(x):
            B, L, C = x.shape
            H, W = hw_shape
            assert L == H * W, 'input feature has wrong size'

            shortcut = x
            x = x if self.use_postln else self.ln1(x)
            x = x.view(B, H, W, C)

            # FM
            x = self.modulation(x).view(B, H * W, C)
            x = x if not self.use_postln else self.ln1(x)

            # FFN
            x = shortcut + self.drop_path(self.gamma1(x))
            if self.use_postln:
                x = x + self.drop_path(self.gamma2(self.ln2(self.ffn(x))))
            else:
                x = x + self.drop_path(self.gamma2(self.ffn(self.ln2(x))))

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


class BasicLayer(BaseModule):
    """A basic focal modulation layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        ffn_ratio (float): Ratio of ffn hidden dim to embedding dim.
            Default: 4.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate.
            Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer.
            Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end
            of the layer. Default: None
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at focal level 1
        use_conv_embed (bool): Use overlapped convolution for patch embedding
            or now. Default: False
        with_cp (bool): Whether to use checkpointing to save memory.
            Default: False.
    """

    def __init__(self,
                 embed_dims,
                 depth,
                 ffn_ratio=4.,
                 drop_rate=0.,
                 drop_paths=0.,
                 norm_cfg=dict(type='LN'),
                 downsample=False,
                 downsample_cfg=dict(),
                 focal_window=9,
                 focal_level=2,
                 use_conv_embed=False,
                 use_layer_scale=False,
                 use_postln=False,
                 normalize_modulator=False,
                 use_postln_in_modulation=False,
                 block_cfgs=dict(),
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg)

        if not isinstance(drop_paths, Sequence):
            drop_paths = [drop_paths] * depth

        if not isinstance(block_cfgs, Sequence):
            block_cfgs = [deepcopy(block_cfgs) for _ in range(depth)]

        self.depth = depth
        self.embed_dims = embed_dims
        self.blocks = ModuleList()
        for i in range(depth):
            _block_cfg = {
                'embed_dims': embed_dims,
                'ffn_ratio': ffn_ratio,
                'drop_rate': drop_rate,
                'drop_path_rate': drop_paths[i],
                'focal_window': focal_window,
                'focal_level': focal_level,
                'use_layer_scale': use_layer_scale,
                'norm_cfg': norm_cfg,
                'use_postln': use_postln,
                'with_cp': with_cp,
                'normalize_modulator': normalize_modulator,
                'use_postln_in_modulation': use_postln_in_modulation,
                **block_cfgs[i]
            }
            block = FocalModulationBlock(**_block_cfg)
            self.blocks.append(block)

        if downsample:
            if use_conv_embed:
                _downsample_cfg = dict(
                    in_channels=embed_dims,
                    input_size=None,
                    embed_dims=2 * embed_dims,
                    conv_type='Conv2d',
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    norm_cfg=dict(type='LN'),
                    **downsample_cfg)
                self.downsample = ConvPatchEmbed(**_downsample_cfg)
            else:
                _downsample_cfg = dict(
                    in_channels=embed_dims,
                    input_size=None,
                    embed_dims=2 * embed_dims,
                    conv_type='Conv2d',
                    kernel_size=2,
                    stride=2,
                    norm_cfg=dict(type='LN'),
                    **downsample_cfg)
                self.downsample = PatchEmbed(**_downsample_cfg)
        else:
            self.downsample = None

    def forward(self, x, in_shape, do_downsample=True):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        for blk in self.blocks:
            x = blk(x, in_shape)
        if self.downsample is not None and do_downsample:
            x = x.transpose(1, 2).reshape(x.shape[0], -1, *in_shape)
            x, out_shape = self.downsample(x, in_shape)
        else:
            out_shape = in_shape
        return x, out_shape

    @property
    def out_channels(self):
        if self.downsample:
            return self.downsample.embed_dims
        else:
            return self.embed_dims


class ConvPatchEmbed(PatchEmbed):
    """Image to Patch Embedding.

    We use a conv layer to implement PatchEmbed.
    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (str): The type of convolution
            to generate patch embedding. Default: "Conv2d".
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int): The slide stride of embedding conv.
            Default: 16.
        padding (int | tuple | string): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int): The dilation rate of embedding conv. Default: 1.
        bias (bool): Bias of embed conv. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        input_size (int | tuple | None): The size of input, which will be
            used to calculate the out size. Only works when `dynamic_size`
            is False. Default: None.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=768,
                 conv_type='Conv2d',
                 kernel_size=16,
                 stride=16,
                 padding='corner',
                 dilation=1,
                 bias=True,
                 norm_cfg=None,
                 input_size=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        self.adaptive_padding = AdaptivePadding(
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding='corner')
        padding = to_2tuple(padding)

        self.projection = build_conv_layer(
            dict(type=conv_type),
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

        if input_size:
            input_size = to_2tuple(input_size)
            # `init_out_size` would be used outside to
            # calculate the num_patches
            # e.g. when `use_abs_pos_embed` outside
            self.init_input_size = input_size
            if self.adaptive_padding:
                pad_h, pad_w = self.adaptive_padding.get_pad_shape(input_size)
                input_h, input_w = input_size
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                input_size = (input_h, input_w)

            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            h_out = (input_size[0] + 2 * padding[0] - dilation[0] *
                     (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (input_size[1] + 2 * padding[1] - dilation[1] *
                     (kernel_size[1] - 1) - 1) // stride[1] + 1
            self.init_out_size = (h_out, w_out)
        else:
            self.init_input_size = None
            self.init_out_size = None


@MODELS.register_module()
class FocalNet(BaseBackbone):
    """FocalNet backbone.
    Args:
        pretrain_img_size (int): Input image size for training the pretrained
            model, used in absolute position embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels.
            Default: 96.
        depths (tuple[int]): Depths of each FocalNet stage.
        ffn_ratio (float): Ratio of ffn hidden dim to embedding dim.
            Default: 4.
        drop_rate (float): Dropout rate.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding.
            Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        focal_levels (Sequence[int]): Number of focal levels at four stages
        focal_windows (Sequence[int]): Focal window sizes at first focal level
            at four stages
        use_conv_embed (bool): Whether use overlapped convolution for patch
            embedding
        use_checkpoint (bool): Whether to use checkpointing to save memory.
            Default: False.
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
    """
    arch_zoo = {
        **dict.fromkeys(
            ['t-srf', 'tiny-srf'], {
                'embed_dims': 96,
                'ffn_ratio': 4.,
                'depths': [2, 2, 6, 2],
                'focal_levels': [2, 2, 2, 2],
                'focal_windows': [3, 3, 3, 3],
                'num_heads': [3, 6, 12, 24],
                'use_conv_embed': False,
                'use_postln': False,
                'use_layer_scale': False,
                'normalize_modulator': False,
                'use_postln_in_modulation': False,
            }),
        **dict.fromkeys(
            ['t-lrf', 'tiny-lrf'], {
                'embed_dims': 96,
                'ffn_ratio': 4.,
                'depths': [2, 2, 6, 2],
                'focal_levels': [3, 3, 3, 3],
                'focal_windows': [3, 3, 3, 3],
                'num_heads': [3, 6, 12, 24],
                'use_conv_embed': False,
                'use_postln': False,
                'use_layer_scale': False,
                'normalize_modulator': False,
                'use_postln_in_modulation': False,
            }),
        **dict.fromkeys(
            ['s-srf', 'small-srf'], {
                'embed_dims': 96,
                'ffn_ratio': 4.,
                'depths': [2, 2, 18, 2],
                'focal_levels': [2, 2, 2, 2],
                'focal_windows': [3, 3, 3, 3],
                'num_heads': [3, 6, 12, 24],
                'use_conv_embed': False,
                'use_postln': False,
                'use_layer_scale': False,
                'normalize_modulator': False,
                'use_postln_in_modulation': False,
            }),
        **dict.fromkeys(
            ['s-lrf', 'small-lrf'], {
                'embed_dims': 96,
                'ffn_ratio': 4.,
                'depths': [2, 2, 18, 2],
                'focal_levels': [3, 3, 3, 3],
                'focal_windows': [3, 3, 3, 3],
                'num_heads': [3, 6, 12, 24],
                'use_conv_embed': False,
                'use_postln': False,
                'use_layer_scale': False,
                'normalize_modulator': False,
                'use_postln_in_modulation': False,
            }),
        **dict.fromkeys(
            ['b-srf', 'base-srf'], {
                'embed_dims': 128,
                'ffn_ratio': 4.,
                'depths': [2, 2, 18, 2],
                'focal_levels': [2, 2, 2, 2],
                'focal_windows': [3, 3, 3, 3],
                'num_heads': [3, 6, 12, 24],
                'use_conv_embed': False,
                'use_postln': False,
                'use_layer_scale': False,
                'normalize_modulator': False,
                'use_postln_in_modulation': False,
            }),
        **dict.fromkeys(
            ['b-lrf', 'base-lrf'], {
                'embed_dims': 128,
                'ffn_ratio': 4.,
                'depths': [2, 2, 18, 2],
                'focal_levels': [3, 3, 3, 3],
                'focal_windows': [3, 3, 3, 3],
                'num_heads': [3, 6, 12, 24],
                'use_conv_embed': False,
                'use_postln': False,
                'use_layer_scale': False,
                'normalize_modulator': False,
                'use_postln_in_modulation': False,
            }),
        **dict.fromkeys(
            ['l-fl3', 'large-fl3'], {
                'embed_dims': 192,
                'ffn_ratio': 4.,
                'depths': [2, 2, 18, 2],
                'focal_levels': [3, 3, 3, 3],
                'focal_windows': [5, 5, 5, 5],
                'num_heads': [3, 6, 12, 24],
                'use_conv_embed': True,
                'use_postln': True,
                'use_layer_scale': True,
                'normalize_modulator': False,
                'use_postln_in_modulation': False,
            }),
        **dict.fromkeys(
            ['l-fl4', 'large-fl4'], {
                'embed_dims': 192,
                'ffn_ratio': 4.,
                'depths': [2, 2, 18, 2],
                'focal_levels': [4, 4, 4, 4],
                'focal_windows': [3, 3, 3, 3],
                'num_heads': [3, 6, 12, 24],
                'use_conv_embed': True,
                'use_postln': True,
                'use_layer_scale': True,
                'normalize_modulator': True,
                'use_postln_in_modulation': False,
            }),
        **dict.fromkeys(
            ['xl-fl3', 'xlarge-fl3'], {
                'embed_dims': 256,
                'ffn_ratio': 4.,
                'depths': [2, 2, 18, 2],
                'focal_levels': [3, 3, 3, 3],
                'focal_windows': [5, 5, 5, 5],
                'num_heads': [3, 6, 12, 24],
                'use_conv_embed': True,
                'use_postln': True,
                'use_layer_scale': True,
                'normalize_modulator': False,
                'use_postln_in_modulation': False,
            }),
        **dict.fromkeys(
            ['xl-fl4', 'xlarge-fl4'], {
                'embed_dims': 256,
                'ffn_ratio': 4.,
                'depths': [2, 2, 18, 2],
                'focal_levels': [4, 4, 4, 4],
                'focal_windows': [3, 3, 3, 3],
                'num_heads': [3, 6, 12, 24],
                'use_conv_embed': True,
                'use_postln': True,
                'use_layer_scale': True,
                'normalize_modulator': False,
                'use_postln_in_modulation': False,
            }),
        **dict.fromkeys(
            ['h-fl3', 'huge-fl3'], {
                'embed_dims': 352,
                'ffn_ratio': 4.,
                'depths': [2, 2, 18, 2],
                'focal_levels': [3, 3, 3, 3],
                'focal_windows': [3, 3, 3, 3],
                'num_heads': [3, 6, 12, 24],
                'use_conv_embed': True,
                'use_postln': True,
                'use_layer_scale': True,
                'normalize_modulator': False,
                'use_postln_in_modulation': True,
            }),
        **dict.fromkeys(
            ['h-fl4', 'huge-fl4'], {
                'embed_dims': 352,
                'ffn_ratio': 4.,
                'depths': [2, 2, 18, 2],
                'focal_levels': [4, 4, 4, 4],
                'focal_windows': [3, 3, 3, 3],
                'num_heads': [3, 6, 12, 24],
                'use_conv_embed': True,
                'use_postln': True,
                'use_layer_scale': True,
                'normalize_modulator': False,
                'use_postln_in_modulation': True,
            }),
    }

    def __init__(self,
                 arch='t-srf',
                 patch_size=4,
                 in_channels=3,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_cfg=dict(type='LN'),
                 out_indices=(3, ),
                 out_after_downsample=False,
                 frozen_stages=-1,
                 with_cp=False,
                 stage_cfgs=dict(),
                 patch_cfg=dict(),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'ffn_ratio', 'depths', 'focal_levels',
                'focal_windows', 'num_heads', 'use_conv_embed', 'use_postln',
                'use_layer_scale', 'normalize_modulator',
                'use_postln_in_modulation'
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.depths = self.arch_settings['depths']
        self.num_layers = len(self.depths)

        self.out_indices = out_indices
        self.out_after_downsample = out_after_downsample
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        # Set patch embedding
        if self.arch_settings['use_conv_embed']:
            _patch_cfg = dict(
                in_channels=in_channels,
                input_size=None,
                embed_dims=self.embed_dims,
                conv_type='Conv2d',
                kernel_size=7,
                padding=3,
                stride=4,
                norm_cfg=dict(type='LN'),
            )
            _patch_cfg.update(patch_cfg)
            self.patch_embed = ConvPatchEmbed(**_patch_cfg)
        else:
            _patch_cfg = dict(
                in_channels=in_channels,
                input_size=None,
                embed_dims=self.embed_dims,
                conv_type='Conv2d',
                kernel_size=patch_size,
                stride=patch_size,
                norm_cfg=dict(type='LN'),
            )
            _patch_cfg.update(patch_cfg)
            self.patch_embed = PatchEmbed(**_patch_cfg)
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # stochastic depth
        total_depth = sum(self.depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule

        # build layers
        self.layers = ModuleList()
        embed_dims = [self.embed_dims]
        for i, (depth, focal_level, focal_window) in enumerate(
                zip(self.depths, self.arch_settings['focal_levels'],
                    self.arch_settings['focal_windows'])):
            if isinstance(stage_cfgs, Sequence):
                stage_cfg = stage_cfgs[i]
            else:
                stage_cfg = deepcopy(stage_cfgs)
            downsample = True if i < self.num_layers - 1 else False
            _stage_cfg = {
                'embed_dims':
                int(self.embed_dims * 2**i),
                'depth':
                depth,
                'ffn_ratio':
                self.arch_settings['ffn_ratio'],
                'drop_rate':
                drop_rate,
                'drop_paths':
                dpr[:depth],
                'norm_cfg':
                norm_cfg,
                'downsample':
                downsample,
                'focal_level':
                focal_level,
                'focal_window':
                focal_window,
                'use_conv_embed':
                self.arch_settings['use_conv_embed'],
                'use_layer_scale':
                self.arch_settings['use_layer_scale'],
                'use_postln':
                self.arch_settings['use_postln'],
                'normalize_modulator':
                self.arch_settings['normalize_modulator'],
                'use_postln_in_modulation':
                self.arch_settings['use_postln_in_modulation'],
                'with_cp':
                with_cp,
                **stage_cfg
            }
            layer = BasicLayer(**_stage_cfg)
            self.layers.append(layer)

            dpr = dpr[depth:]
            embed_dims.append(layer.out_channels)

        if self.out_after_downsample:
            self.num_features = embed_dims[1:]
        else:
            self.num_features = embed_dims[:-1]

        for i in out_indices:
            if norm_cfg is not None:
                norm_layer = build_norm_layer(norm_cfg,
                                              self.num_features[i])[1]
            else:
                norm_layer = nn.Identity()

            self.add_module(f'norm{i}', norm_layer)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

            self.drop_after_pos.eval()

        for i in range(0, self.frozen_stages + 1):
            m = self.layers[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        for i in self.out_indices:
            if i <= self.frozen_stages:
                for param in getattr(self, f'norm{i}').parameters():
                    param.requires_grad = False

    def init_weights(self):
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None or 'checkpoint' not in self.init_cfg:
            super().init_weights()
        else:
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                state_dict = ckpt

            prefix = self.init_cfg.get('prefix', None)
            if prefix is not None:
                if not prefix.endswith('.'):
                    prefix += '.'
                prefix_len = len(prefix)

                state_dict = {
                    k[prefix_len:]: v
                    for k, v in state_dict.items() if k.startswith(prefix)
                }

                assert state_dict, f'{prefix} is not in the pretrained model'

            focal_layers_keys = [
                k for k in state_dict.keys()
                if ('focal_layers' in k and 'bias' not in k)
            ]
            for table_key in focal_layers_keys:
                if table_key not in self.state_dict():
                    continue
                table_pretrained = state_dict[table_key]
                table_current = self.state_dict()[table_key]

                if len(table_pretrained.shape) != 4:
                    L1 = table_pretrained.shape[1]
                    L2 = table_current.shape[1]

                    if L1 != L2:
                        S1 = int(L1**0.5)
                        S2 = int(L2**0.5)
                        table_pretrained_resized = F.interpolate(
                            table_pretrained.view(1, 1, S1, S1),
                            size=(S2, S2),
                            mode='bicubic')
                        state_dict[table_key] = table_pretrained_resized.view(
                            1, L2) * L1 / L2
                else:
                    fsize1 = table_pretrained.shape[2]
                    fsize2 = table_current.shape[2]

                    # NOTE: different from interpolation used in
                    # self-attention, we use padding or clipping for focal conv
                    if fsize1 < fsize2:
                        table_pretrained_resized = torch.zeros(
                            table_current.shape)
                        table_pretrained_resized[:, :, (fsize2 - fsize1) //
                                                 2:-(fsize2 - fsize1) // 2,
                                                 (fsize2 - fsize1) //
                                                 2:-(fsize2 - fsize1) //
                                                 2] = table_pretrained
                        state_dict[table_key] = table_pretrained_resized
                    elif fsize1 > fsize2:
                        table_pretrained_resized = table_pretrained[:, :, (
                            fsize1 - fsize2) // 2:-(fsize1 - fsize2) // 2, (
                                fsize1 - fsize2) // 2:-(fsize1 - fsize2) // 2]
                        state_dict[table_key] = table_pretrained_resized

            f_layers_keys = [
                k for k in state_dict.keys() if ('modulation.f' in k)
            ]
            for table_key in f_layers_keys:
                if table_key not in self.state_dict():
                    continue
                table_pretrained = state_dict[table_key]
                table_current = self.state_dict()[table_key]
                if table_pretrained.shape != table_current.shape:
                    if len(table_pretrained.shape) == 2:
                        # for linear weights
                        dim = table_pretrained.shape[1]
                        assert table_current.shape[1] == dim
                        L1 = table_pretrained.shape[0]
                        L2 = table_current.shape[0]

                        if L1 < L2:
                            table_pretrained_resized = torch.zeros(
                                table_current.shape)
                            # copy for linear project
                            (table_pretrained_resized[:2 * dim]
                             ) = table_pretrained[:2 * dim]
                            # copy for global token gating
                            table_pretrained_resized[-1] = table_pretrained[-1]
                            # copy for first multiple focal levels
                            table_pretrained_resized[2 * dim:2 * dim + (
                                L1 - 2 * dim - 1)] = table_pretrained[2 *
                                                                      dim:-1]
                            # reassign pretrained weights
                            state_dict[table_key] = table_pretrained_resized
                        elif L1 > L2:
                            raise NotImplementedError
                    elif len(table_pretrained.shape) == 1:
                        # for linear bias
                        L1 = table_pretrained.shape[0]
                        L2 = table_current.shape[0]
                        if L1 < L2:
                            table_pretrained_resized = torch.zeros(
                                table_current.shape)
                            # copy for linear project
                            (table_pretrained_resized[:2 * dim]
                             ) = table_pretrained[:2 * dim]
                            # copy for global token gating
                            table_pretrained_resized[-1] = table_pretrained[-1]
                            # copy for first multiple focal levels
                            table_pretrained_resized[2 * dim:2 * dim + (
                                L1 - 2 * dim - 1)] = table_pretrained[2 *
                                                                      dim:-1]
                            # reassign pretrained weights
                            state_dict[table_key] = table_pretrained_resized
                        elif L1 > L2:
                            raise NotImplementedError

            # load state_dict
            self.load_state_dict(state_dict, False)

    def forward(self, x):
        """Forward function."""
        x, hw_shape = self.patch_embed(x)
        x = self.drop_after_pos(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x, hw_shape = layer(
                x, hw_shape, do_downsample=self.out_after_downsample)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(x)
                out = out.view(-1, *hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)
            if layer.downsample is not None and not self.out_after_downsample:
                x = x.transpose(1, 2).reshape(x.shape[0], -1, *hw_shape)
                x, hw_shape = layer.downsample(x)

        return tuple(outs)

    def train(self, mode=True):
        super(FocalNet, self).train(mode)
        self._freeze_stages()
