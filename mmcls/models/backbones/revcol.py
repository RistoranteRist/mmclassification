# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch.autograd import Function as Function

from mmcls.models.backbones.convnext import ConvNeXtBlock
from mmcls.models.utils import build_norm_layer
from mmcls.registry import MODELS


def get_gpu_states(fwd_gpu_devices):
    # This will not error out if "arg" is a CPU tensor or a non-tensor type
    # because the conditionals short-circuit.
    fwd_gpu_states = []
    for device in fwd_gpu_devices:
        with torch.cuda.device(device):
            fwd_gpu_states.append(torch.cuda.get_rng_state())

    return fwd_gpu_states


def get_gpu_device(*args):

    fwd_gpu_devices = list(
        set(arg.get_device() for arg in args
            if isinstance(arg, torch.Tensor) and arg.is_cuda))
    return fwd_gpu_devices


def set_device_states(fwd_cpu_state, devices, states):
    torch.set_rng_state(fwd_cpu_state)
    for device, state in zip(devices, states):
        with torch.cuda.device(device):
            torch.cuda.set_rng_state(state)


def detach_and_grad(inputs):
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                out.append(inp)
                continue

            x = inp.detach()
            x.requires_grad = True
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            'Only tuple of tensors is supported. Got Unsupported input type: ',
            type(inputs).__name__)


def get_cpu_and_gpu_states(gpu_devices):
    return torch.get_rng_state(), get_gpu_states(gpu_devices)


class ReverseFunction(Function):
    """Custom Backpropagation function to allow (A) flushing memory in forward
    and (B) activation recomputation reversibly in backward for gradient
    calculation.

    Inspired by
    https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
    """

    @staticmethod
    def forward(ctx, x, c0, c1, c2, c3, run_functions, alpha):
        """Reversible Forward pass.

        Any intermediate activations from `buffer_layers` are cached in ctx for
        forward pass. This is not necessary for standard usecases. Each
        reversible layer implements its own forward pass logic.
        """
        l0, l1, l2, l3 = run_functions
        alpha0, alpha1, alpha2, alpha3 = alpha
        ctx.run_functions = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        ctx.gpu_autocast_kwargs = {
            'enabled': torch.is_autocast_enabled(),
            'dtype': torch.get_autocast_gpu_dtype(),
            'cache_enabled': torch.is_autocast_cache_enabled()
        }
        ctx.cpu_autocast_kwargs = {
            'enabled': torch.is_autocast_cpu_enabled(),
            'dtype': torch.get_autocast_cpu_dtype(),
            'cache_enabled': torch.is_autocast_cache_enabled()
        }

        if type(c0) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False

        with torch.no_grad():
            gpu_devices = get_gpu_device(x, c0, c1, c2, c3)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_0, ctx.gpu_states_0 = get_cpu_and_gpu_states(
                gpu_devices)
            c0 = l0(x, c1) + c0 * alpha0
            ctx.cpu_states_1, ctx.gpu_states_1 = get_cpu_and_gpu_states(
                gpu_devices)
            c1 = l1(c0, c2) + c1 * alpha1
            ctx.cpu_states_2, ctx.gpu_states_2 = get_cpu_and_gpu_states(
                gpu_devices)
            c2 = l2(c1, c3) + c2 * alpha2
            ctx.cpu_states_3, ctx.gpu_states_3 = get_cpu_and_gpu_states(
                gpu_devices)
            c3 = l3(c2, None) + c3 * alpha3
        ctx.save_for_backward(x, c0, c1, c2, c3)
        return x, c0, c1, c2, c3

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, c0, c1, c2, c3 = ctx.saved_tensors
        l0, l1, l2, l3 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right = grad_outputs
        (x, c0, c1, c2, c3) = detach_and_grad((x, c0, c1, c2, c3))

        with (torch.enable_grad(),
              torch.random.fork_rng(
                  devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state),
              torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs),
              torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs)):

            g3_up = g3_right
            g3_left = g3_up * alpha3
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices,
                              ctx.gpu_states_3)
            oup3 = l3(c2, None)
            torch.autograd.backward(oup3, g3_up, retain_graph=True)
            with torch.no_grad():
                c3_left = (1 / alpha3) * (c3 - oup3)
            g2_up = g2_right + c2.grad
            g2_left = g2_up * alpha2

            (c3_left, ) = detach_and_grad((c3_left, ))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices,
                              ctx.gpu_states_2)
            oup2 = l2(c1, c3_left)
            torch.autograd.backward(oup2, g2_up, retain_graph=True)
            c3_left.requires_grad = False
            cout3 = c3_left * alpha3
            torch.autograd.backward(cout3, g3_up)

            with torch.no_grad():
                c2_left = (1 / alpha2) * (c2 - oup2)
            g3_left = (g3_left +
                       c3_left.grad) if c3_left.grad is not None else g3_left
            g1_up = g1_right + c1.grad
            g1_left = g1_up * alpha1

            (c2_left, ) = detach_and_grad((c2_left, ))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices,
                              ctx.gpu_states_1)
            oup1 = l1(c0, c2_left)
            torch.autograd.backward(oup1, g1_up, retain_graph=True)
            c2_left.requires_grad = False
            cout2 = c2_left * alpha2
            torch.autograd.backward(cout2, g2_up)

            with torch.no_grad():
                c1_left = (1 / alpha1) * (c1 - oup1)
            g0_up = g0_right + c0.grad
            g0_left = g0_up * alpha0
            g2_left = (g2_left +
                       c2_left.grad) if c2_left.grad is not None else g2_left

            (c1_left, ) = detach_and_grad((c1_left, ))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices,
                              ctx.gpu_states_0)
            oup0 = l0(x, c1_left)
            torch.autograd.backward(oup0, g0_up, retain_graph=True)
            c1_left.requires_grad = False
            cout1 = c1_left * alpha1
            torch.autograd.backward(cout1, g1_up)

            with torch.no_grad():
                c0_left = (1 / alpha0) * (c0 - oup0)
            gx_up = x.grad
            g1_left = (g1_left +
                       c1_left.grad) if c1_left.grad is not None else g1_left
            c0_left.requires_grad = False
            cout0 = c0_left * alpha0
            torch.autograd.backward(cout0, g0_up)

        if ctx.first_col:
            return None, None, gx_up, None, None, None, None
        else:
            return None, None, gx_up, g0_left, g1_left, g2_left, g3_left


class UpSampleConvnext(BaseModule):

    def __init__(self,
                 ratio,
                 inchannel,
                 outchannel,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.ratio = ratio
        self.channel_reschedule = nn.Linear(inchannel, outchannel)
        self.norm = build_norm_layer(norm_cfg, self.channels[0])

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.channel_reschedule(x)
        x = self.norm(x, data_format='channels_last')
        x = x.permute(0, 3, 1, 2)

        return x


class Fusion(BaseModule):

    def __init__(self,
                 level,
                 channels,
                 first_col,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 init_cfg=None) -> None:
        super().__init__(init_cfg)

        self.down = nn.Sequential(
            nn.Conv2d(
                channels[level - 1], channels[level], kernel_size=2, stride=2),
            build_norm_layer(norm_cfg, self.channels[0])) if level in [
                1, 2, 3
            ] else nn.Identity()
        if not first_col and level in [0, 1, 2]:
            self.up = UpSampleConvnext(1, channels[level + 1], channels[level])
        else:
            self.up = None

    def forward(self, x_down, x_up):

        if self.up is None:
            x = self.down(x_down)
        else:
            x_down = self.down(x_down)
            x_up = self.up(x_up)
            x_up = F.interpolate(x_up, size=x_down.shape[2:], mode='nearest')
            x = x_up + x_down
        return x


class Level(BaseModule):

    def __init__(self,
                 level,
                 channels,
                 depths,
                 first_col,
                 drop_paths=0.0,
                 layer_scale_init_value=1e-6,
                 dw_conv_cfg=dict(kernel_size=3, padding=1),
                 init_cfg=None) -> None:
        super().__init__(init_cfg)
        countlayer = sum(depths[:level])
        self.fusion = Fusion(level, channels, first_col)
        modules = [
            ConvNeXtBlock(
                in_channels=channels[level],
                dw_conv_cfg=dw_conv_cfg,
                drop_path_rate=drop_paths[countlayer + i],
                layer_scale_init_value=layer_scale_init_value)
            for i in range(depths[level])
        ]
        self.blocks = nn.Sequential(*modules)

    def forward(self, x_down, x_up):
        x = self.fusion(x_down, x_up)
        x = self.blocks(x)
        return x


class SubNet(BaseModule):

    def __init__(self,
                 channels,
                 layers,
                 first_col,
                 drop_paths,
                 shortcut_scale_init_value=0.5,
                 init_cfg=None) -> None:
        super().__init__(init_cfg)
        self.alpha0 = nn.Parameter(
            shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
            requires_grad=True)
        self.alpha1 = nn.Parameter(
            shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
            requires_grad=True)
        self.alpha2 = nn.Parameter(
            shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
            requires_grad=True)
        self.alpha3 = nn.Parameter(
            shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
            requires_grad=True)

        self.level0 = Level(0, channels, layers, first_col, drop_paths)

        self.level1 = Level(1, channels, layers, first_col, drop_paths)

        self.level2 = Level(2, channels, layers, first_col, drop_paths)

        self.level3 = Level(3, channels, layers, first_col, drop_paths)

    def forward(self, x, c0, c1, c2, c3):

        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)

        local_funs = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]
        _, c0, c1, c2, c3 = ReverseFunction.apply(x, c0, c1, c2, c3,
                                                  local_funs, alpha)

        return c0, c1, c2, c3

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign


@MODELS.register_module()
class RevCol(BaseModule):
    arch_zoo = {
        'tiny': {
            'depths': [2, 2, 4, 2],
            'channels': [64, 128, 256, 512],
            'num_subnet': 4
        },
        'small': {
            'depths': [2, 2, 4, 2],
            'channels': [64, 128, 256, 512],
            'num_subnet': 8
        },
        'base': {
            'depths': [1, 1, 3, 2],
            'channels': [72, 144, 288, 576],
            'num_subnet': 16
        },
        'large': {
            'depths': [1, 2, 6, 2],
            'channels': [128, 256, 512, 1024],
            'num_subnet': 8
        },
        'xlarge': {
            'depths': [1, 2, 6, 2],
            'channels': [224, 448, 896, 1792],
            'num_subnet': 8
        }
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 drop_path_rate=0.,
                 init_cfg=None):
        super().__init__(init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        elif isinstance(arch, dict):
            essential_keys = {'channels', 'depths', 'num_subnet'}
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.depths = self.arch_settings['depths']
        self.channels = self.arch_settings['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_subnet = self.arch_settings['num_subnet']

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]

        for i in range(self.num_subnet):
            first_col = True if i == 0 else False
            self.add_module(
                f'subnet{str(i)}',
                SubNet(self.channels, self.depths, first_col, drop_paths=dpr))

    def forward(self, x):
        out = []
        c0, c1, c2, c3 = 0, 0, 0, 0
        interval = self.num_subnet // 4

        x = self.stem(x)
        for i in range(self.num_subnet):
            c0, c1, c2, c3 = getattr(self, f'subnet{str(i)}')(x, c0, c1, c2,
                                                              c3)
            if (i + 1) % interval == 0:
                out.append(c3)

        return out
