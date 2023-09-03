# Copyright (c) OpenMMLab. All rights reserved.
import functools
import warnings
import torch


class ZeroOutHook:
    """Zero out feature map of some layers.

    Args:
        module (nn.Module): The whole module to zero out.
        zero_out_feats (dict[layer_name:torch.Tensor]): Mean features on validation set, used to zero out target layers.
        zero_out_modules (list[str]): Determine which layers to zero out.
    """

    def __init__(self, module, zero_out_feats, zero_out_modules=None):
        self.zero_out_modules = zero_out_modules
        self.zero_out_feats = zero_out_feats
        self.handles = []
        self.register(module)

    def register(self, module):

        def hook_wrapper(name):
            def hook(model, input, output):
                assert isinstance(output, torch.Tensor)
                new_output = self.zero_out_feats[name]
                batch_size = output.size()[0] // new_output.size()[0]
                new_output = new_output.repeat(batch_size, 1, 1)
                assert new_output.size() == output.size()
                output = new_output
                return output
            return hook

        if isinstance(self.zero_out_modules, (list, tuple)):
            for name in self.zero_out_modules:
                try:
                    layer = rgetattr(module, name)
                    h = layer.register_forward_hook(hook_wrapper(name))
                except AttributeError:
                    raise AttributeError(f'Module {name} not found')
                self.handles.append(h)

    def remove(self):
        for h in self.handles:
            h.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()

# using wonder's beautiful simplification:
# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects
def rgetattr(obj, attr, *args):

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))
