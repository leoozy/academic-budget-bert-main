import sys
import torch
import torch.nn.init as init
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Module

def get_apex_layer_norm():
    try:
        import apex

        # apex.amp.register_half_function(apex.normalization.fused_layer_norm, 'FusedLayerNorm')
        import apex.normalization

        # apex.amp.register_float_function(apex.normalization.FusedLayerNorm, 'forward')
        return apex.normalization.FusedLayerNorm
    except ImportError:
        raise Exception(f"Layer norm of type apex is not available, apex not installed.")

ACT2FN = {"gelu": F.gelu, "relu": F.relu, "swish": swish, "tanh": F.tanh}
LAYER_NORM_TYPES = {"pytorch": nn.LayerNorm, "apex": get_apex_layer_norm(), "rms_norm": RMSNorm}

def get_layer_norm_type(config):
    if config.layer_norm_type in LAYER_NORM_TYPES:
        return LAYER_NORM_TYPES[config.layer_norm_type]
    else:
        raise Exception(f"Layer norm of type {config.layer_norm_type} is not available.")

class LinearActivation(Module):
    r"""Fused Linear and activation Module."""
    __constants__ = ["bias"]

    def __init__(self, in_features, out_features, act="gelu", bias=True):
        super(LinearActivation, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fused_gelu = False
        self.fused_tanh = False
        self.fused_relu = False
        if isinstance(act, str) or (sys.version_info[0] == 2 and isinstance(act, unicode)):
            if bias and act == "gelu":
                self.fused_gelu = True
            elif bias and act == "tanh":
                self.fused_tanh = True
            elif bias and act == "relu":
                self.fused_relu = True
            else:
                self.act_fn = ACT2FN[act]
        else:
            self.act_fn = act
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, inp):
        if self.fused_gelu:
            return bias_gelu(self.bias, F.linear(inp, self.weight, None))
        elif self.fused_tanh:
            return bias_tanh(self.bias, F.linear(inp, self.weight, None))
        elif self.fused_relu:
            return bias_relu(self.bias, F.linear(inp, self.weight, None))
        else:
            return self.act_fn(F.linear(inp, self.weight, self.bias))

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )

if __name__ == '__main__':
    f = LinearActivation(10,10)