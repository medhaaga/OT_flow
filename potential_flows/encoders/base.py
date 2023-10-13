import torch
import torch.nn as nn
from typing import Union
from potential_flows import potential

class EncoderDecoder(nn.Module):

    def __init__(self):
        super(EncoderDecoder, self).__init__()

    def encode(self):
        raise NotImplementedError()

    def decode(self):
        raise NotImplementedError()

class EncoderDecoder_OT(nn.Module):
    """Base class for flows that use an encoder-decoder for both souce and target distribution
        A potential flow is constructed between the encoded source and target."""

    def __init__(self,
                transform_x: EncoderDecoder,
                transform_y: EncoderDecoder,
                potential: potential.Potential):

        super(EncoderDecoder_OT, self).__init__()

        self.transform_x =transform_x
        self.transform_y =transform_y
        self.potential = potential

    def encode_x(self, x):
        return self.transform_x.encode(x)

    def encode_y(self, y):
        return self.transform_y.encode(y)

    def decode_x(self, x):
        return self.transform_x.decode(x)

    def decode_y(self, y):
        return self.transform_y.decode(y)

    def gradient(self, x):
        T_x = self.encode_x(x)
        f_T_x = self.potential.gradient(T_x)
        return self.decode_y(f_T_x)

    def gradient_inv(self, y):
        T_y = self.encode_y(y)
        f_inv_T_x = self.potential.gradient_inv(T_y)
        return self.decode_x(f_inv_T_x)