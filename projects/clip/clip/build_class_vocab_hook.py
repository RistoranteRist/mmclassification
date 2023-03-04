# Copyright (c) OpenMMLab. All rights reserved
import warnings

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmcls.registry import HOOKS
from .zero_shot_cls_head import ZeroShotClsHead


@HOOKS.register_module()
class BuildClassVocabHook(Hook):
    """The hook to build class vocabulary in ZeroShotClsHead."""

    def _build_text_encoder(self, runner, dataset):
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        if isinstance(model.head, ZeroShotClsHead):
            if hasattr(model.head, '_build_class_vocabulary'):
                model.head._build_class_vocabulary(dataset.CLASSES)
        else:
            warnings.warn(
                'Only the `ZeroShotClsHead` can execute '
                f'`BuildTextEncoderHook`, but got `{type(model.head)}`')

    def before_train(self, runner) -> None:
        dataset = runner.train_dataloader.dataset
        self._build_text_encoder(runner, dataset)

    def before_val(self, runner) -> None:
        dataset = runner.val_dataloader.dataset
        self._build_text_encoder(runner, dataset)

    def before_test(self, runner) -> None:
        dataset = runner.test_dataloader.dataset
        self._build_text_encoder(runner, dataset)
