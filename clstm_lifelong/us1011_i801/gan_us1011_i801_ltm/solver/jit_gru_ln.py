# MIT License
#
# Copyright (c) 2020 Mehran Maghoumi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ----------------------------------------------------------------------------------------------------------------------
import torch
import torch.jit as jit
import torch.nn as nn
from torch.nn import Parameter
from typing import List, Tuple
from torch import Tensor
import math


# ----------------------------------------------------------------------------------------------------------------------
class LayerNormGRUCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LayerNormGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.ln_i2h = torch.nn.LayerNorm(2*hidden_size, elementwise_affine=False)
        self.ln_h2h = torch.nn.LayerNorm(2*hidden_size, elementwise_affine=False)
        self.ln_cell_1 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln_cell_2 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.i2h = torch.nn.Linear(input_size, 2 * hidden_size, bias=bias)
        self.h2h = torch.nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        self.h_hat_W = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.h_hat_U = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    @jit.script_method
    def forward(self, x, h):
        # type: (Tensor, Tensor) -> Tensor
        h = h
        h = h.view(h.size(0), -1)
        x = x.view(x.size(0), -1)

        # Linear mappings
        i2h = self.i2h(x)
        h2h = self.h2h(h)

        # Layer norm
        i2h = self.ln_i2h(i2h)
        h2h = self.ln_h2h(h2h)

        preact = i2h + h2h

        # activations
        gates = preact[:, :].sigmoid()
        z_t, r_t = gates.chunk(2, 1)
        # z_t = gates[:, :self.hidden_size]
        # r_t = gates[:, -self.hidden_size:]

        # h_hat
        h_hat_first_half = self.h_hat_W(x)
        h_hat_last_half = self.h_hat_U(h)

        # layer norm
        h_hat_first_half = self.ln_cell_1( h_hat_first_half )
        h_hat_last_half = self.ln_cell_2( h_hat_last_half )

        h_hat = torch.tanh(h_hat_first_half + torch.mul(r_t, h_hat_last_half))

        h_t = torch.mul(1-z_t, h) + torch.mul(z_t, h_hat)

        # Reshape for compatibility

        h_t = h_t.view(h_t.size(0), -1)
        return h_t


# ----------------------------------------------------------------------------------------------------------------------
class JitGRULayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(JitGRULayer, self).__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, x, hidden):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
        inputs = x.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])

        for i in range(len(inputs)):
            hidden = self.cell(inputs[i], hidden)
            outputs += [hidden]

        return torch.stack(outputs), hidden


# ----------------------------------------------------------------------------------------------------------------------
class JitGRULN(jit.ScriptModule):
    __constants__ = ['hidden_size', 'num_layers', 'batch_first', 'layers']

    def __init__(self, input_size, hidden_size, num_layers, batch_first=False, bias=True):
        super(JitGRULN, self).__init__()
        # The following are not implemented.
        assert bias

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        if num_layers == 1:
            self.layers = nn.ModuleList([JitGRULayer(LayerNormGRUCell, input_size, hidden_size)])
        else:
            self.layers = nn.ModuleList([JitGRULayer(LayerNormGRUCell, input_size, hidden_size)] + [JitGRULayer(LayerNormGRUCell, hidden_size, hidden_size)
                                                                                              for _ in range(num_layers - 1)])

    @jit.script_method
    def forward(self, x, h=None):
        # type: (Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor]
        output_states = jit.annotate(List[Tensor], [])

        # Handle batch_first cases
        if self.batch_first:
            x = x.permute(1, 0, 2)

        if h is None:
            h = torch.zeros(self.num_layers, x.shape[1], self.hidden_size, dtype=x.dtype, device=x.device)

        output = x
        i = 0

        for rnn_layer in self.layers:
            output, hidden = rnn_layer(output, h[i])
            output_states += [hidden]
            i += 1

        # Don't forget to handle batch_first cases for the output too!
        if self.batch_first:
            output = output.permute(1, 0, 2)

        return output, torch.stack(output_states)
