from mmengine.model.wrappers.utils import is_model_wrapper
from mmengine.registry import HOOKS
from mmengine.hooks import Hook


@HOOKS.register_module()
class SetEpochInfoHook(Hook):
    """Set runner's epoch information to the model."""

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        max_epochs = runner.max_epochs
        model = runner.model
        
        if is_model_wrapper(model):
            model = model.module
        model.set_epoch(epoch, max_epochs)