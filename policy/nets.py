# adopted from
# https://github.com/jrobine/twm/blob/main/twm/nets.py
import copy
import math
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn, optim


class AdamOptim:

    def __init__(
        self, parameters, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, grad_clip=0
    ):
        self.parameters = list(parameters)
        self.grad_clip = grad_clip
        self.optimizer = optim.Adam(
            self.parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )

    def step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.parameters, self.grad_clip)
        self.optimizer.step()


def get_activation(nonlinearity, param=None):
    if nonlinearity is None or nonlinearity == "none" or nonlinearity == "linear":
        return nn.Identity()
    elif nonlinearity == "relu":
        return nn.ReLU()
    elif nonlinearity == "leaky_relu":
        if param is None:
            param = 1e-2
        return nn.LeakyReLU(negative_slope=param)
    elif nonlinearity == "elu":
        if param is None:
            param = 1.0
        return nn.ELU(alpha=param)
    elif nonlinearity == "silu":
        return nn.SiLU()
    else:
        raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")


def get_norm_1d(norm, k):
    if norm is None or norm == "none":
        return nn.Identity()
    elif norm == "batch_norm":
        return nn.BatchNorm1d(k)
    elif norm == "layer_norm":
        return nn.LayerNorm(k)
    else:
        raise ValueError(f"Unsupported norm: {norm}")


def get_norm_2d(norm, c, h=None, w=None):
    if norm == "none":
        return nn.Identity()
    elif norm == "batch_norm":
        return nn.BatchNorm2d(c)
    elif norm == "layer_norm":
        assert h is not None and w is not None
        return nn.LayerNorm([c, h, w])
    else:
        raise ValueError(f"Unsupported norm: {norm}")


def _calculate_gain(nonlinearity, param=None):
    if nonlinearity == "elu":
        nonlinearity = "selu"
        param = 1
    elif nonlinearity == "silu":
        nonlinearity = "relu"
        param = None
    return torch.nn.init.calculate_gain(nonlinearity, param)


def _kaiming_uniform_(tensor, gain):
    # same as torch.nn.init.kaiming_uniform_, but uses gain
    fan = torch.nn.init._calculate_correct_fan(tensor, mode="fan_in")
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    torch.nn.init._no_grad_uniform_(tensor, -bound, bound)


def _get_initializer(name, nonlinearity=None, param=None):
    if nonlinearity is None:
        assert param is None
    if name == "kaiming_uniform":
        if nonlinearity is None:
            # defaults from PyTorch
            nonlinearity = "leaky_relu"
            param = math.sqrt(5)
        return lambda x: _kaiming_uniform_(x, gain=_calculate_gain(nonlinearity, param))
    elif name == "xavier_uniform":
        if nonlinearity is None:
            nonlinearity = "relu"
        return lambda x: torch.nn.init.xavier_uniform_(
            x, gain=_calculate_gain(nonlinearity, param)
        )
    elif name == "orthogonal":
        if nonlinearity is None:
            nonlinearity = "relu"
        return lambda x: torch.nn.init.orthogonal_(
            x, gain=_calculate_gain(nonlinearity, param)
        )
    elif name == "zeros":
        return lambda x: torch.nn.init.zeros_(x)
    else:
        raise ValueError(f"Unsupported initializer: {name}")


def init_(
    mod, weight_initializer=None, bias_initializer=None, nonlinearity=None, param=None
):
    weight_initializer = (
        _get_initializer(weight_initializer, nonlinearity, param)
        if weight_initializer is not None
        else lambda x: x
    )
    bias_initializer = (
        _get_initializer(bias_initializer, nonlinearity="linear", param=None)
        if bias_initializer is not None
        else lambda x: x
    )

    def fn(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            weight_initializer(m.weight)
            if m.bias is not None:
                bias_initializer(m.bias)

    return mod.apply(fn)


class _MultilayerModule(nn.Module):

    def __init__(
        self,
        layer_prefix,
        ndim,
        in_dim,
        num_layers,
        nonlinearity,
        param,
        norm,
        dropout_p,
        pre_activation,
        post_activation,
        weight_initializer,
        bias_initializer,
        final_bias_init,
    ):
        super().__init__()
        self.layer_prefix = layer_prefix
        self.ndim = ndim
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.param = param
        self.pre_activation = pre_activation
        self.post_activation = post_activation
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.final_bias_init = final_bias_init

        self.has_norm = norm is not None and norm != "none"
        self.has_dropout = dropout_p != 0
        self.unsqueeze = in_dim == 0

        self.act = get_activation(nonlinearity, param)

    def reset_parameters(self):
        init_(
            self,
            self.weight_initializer,
            self.bias_initializer,
            self.nonlinearity,
            self.param,
        )
        final_layer = getattr(self, f"{self.layer_prefix}{self.num_layers}")
        if not self.post_activation:
            init_(
                final_layer,
                self.weight_initializer,
                self.bias_initializer,
                nonlinearity="linear",
                param=None,
            )
        if self.final_bias_init is not None:

            def final_init(m):
                if isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None:
                    with torch.no_grad():
                        m.bias.data.fill_(self.final_bias_init)

            final_layer.apply(final_init)

    def forward(self, x):
        if self.unsqueeze:
            x = x.unsqueeze(-self.ndim)

        if x.ndim > self.ndim + 1:
            batch_shape = x.shape[: -self.ndim]
            x = x.reshape(-1, *x.shape[-self.ndim :])
        else:
            batch_shape = None

        if self.pre_activation:
            if self.has_norm:
                x = getattr(self, "norm0")(x)
            x = self.act(x)

        for i in range(self.num_layers - 1):
            x = getattr(self, f"{self.layer_prefix}{i + 1}")(x)
            if self.has_norm:
                x = getattr(self, f"norm{i + 1}")(x)
            x = self.act(x)
            if self.has_dropout:
                x = self.dropout(x)
        x = getattr(self, f"{self.layer_prefix}{self.num_layers}")(x)

        if self.post_activation:
            if self.has_norm:
                x = getattr(self, f"norm{self.num_layers}")(x)
            x = self.act(x)

        if batch_shape is not None:
            x = x.unflatten(0, batch_shape)
        return x


class MLP(_MultilayerModule):

    def __init__(
        self,
        in_dim,
        hidden_dims,
        out_dim,
        nonlinearity,
        param=None,
        norm=None,
        dropout_p=0,
        bias=True,
        pre_activation=False,
        post_activation=False,
        weight_initializer="kaiming_uniform",
        bias_initializer="zeros",
        final_bias_init=None,
    ):
        dims = (in_dim,) + tuple(hidden_dims) + (out_dim,)
        super().__init__(
            "linear",
            1,
            in_dim,
            len(dims) - 1,
            nonlinearity,
            param,
            norm,
            dropout_p,
            pre_activation,
            post_activation,
            weight_initializer,
            bias_initializer,
            final_bias_init,
        )
        if self.unsqueeze:
            dims = (1,) + dims[1:]

        if pre_activation and self.has_norm:
            norm_layer = get_norm_1d(norm, in_dim)
            self.add_module(f"norm0", norm_layer)

        for i in range(self.num_layers - 1):
            linear_layer = nn.Linear(dims[i], dims[i + 1], bias=bias)
            self.add_module(f"linear{i + 1}", linear_layer)
            if self.has_norm:
                norm_layer = get_norm_1d(norm, dims[i + 1])
                self.add_module(f"norm{i + 1}", norm_layer)

        linear_layer = nn.Linear(dims[-2], dims[-1], bias=bias)
        self.add_module(f"linear{self.num_layers}", linear_layer)

        if post_activation and self.has_norm:
            norm_layer = get_norm_1d(norm, dims[-1])
            self.add_module(f"norm{self.num_layers}", norm_layer)

        if self.has_dropout:
            self.dropout = nn.Dropout(dropout_p)

        self.reset_parameters()


class CNN(_MultilayerModule):

    def __init__(
        self,
        in_dim,
        hidden_dims,
        out_dim,
        kernel_sizes,
        strides,
        paddings,
        nonlinearity,
        param=None,
        norm=None,
        dropout_p=0,
        bias=True,
        padding_mode="zeros",
        in_shape=None,
        pre_activation=False,
        post_activation=False,
        weight_initializer="kaiming_uniform",
        bias_initializer="zeros",
        final_bias_init=None,
    ):
        assert len(kernel_sizes) == len(hidden_dims) + 1
        assert len(strides) == len(kernel_sizes) and len(paddings) == len(kernel_sizes)
        dims = (in_dim,) + tuple(hidden_dims) + (out_dim,)
        super().__init__(
            "conv",
            3,
            in_dim,
            len(dims) - 1,
            nonlinearity,
            param,
            norm,
            dropout_p,
            pre_activation,
            post_activation,
            weight_initializer,
            bias_initializer,
            final_bias_init,
        )
        if self.unsqueeze:
            dims = (1,) + dims[1:]

        def to_pair(x):
            if isinstance(x, int):
                return x, x
            assert isinstance(x, tuple) and len(x) == 2
            return x

        def calc_out_shape(shape, kernel_size, stride, padding):
            kernel_size, padding, stride = [
                to_pair(x) for x in (kernel_size, stride, padding)
            ]
            return tuple(
                (shape[j] + 2 * padding[j] - kernel_size[j]) / stride[j] + 1
                for j in [0, 1]
            )

        if pre_activation and self.has_norm:
            norm_layer = get_norm_2d(norm, in_dim, in_shape[0], in_shape[1])
            self.add_module("norm0", norm_layer)

        shape = in_shape
        for i in range(self.num_layers - 1):
            conv_layer = nn.Conv2d(
                dims[i],
                dims[i + 1],
                kernel_sizes[i],
                strides[i],
                paddings[i],
                bias=bias,
                padding_mode=padding_mode,
            )
            self.add_module(f"conv{i + 1}", conv_layer)
            if self.has_norm:
                if shape is not None:
                    shape = calc_out_shape(
                        shape, kernel_sizes[i], strides[i], paddings[i]
                    )
                norm_layer = get_norm_2d(norm, dims[i + 1], shape[0], shape[1])
                self.add_module(f"norm{i + 1}", norm_layer)

        conv_layer = nn.Conv2d(
            dims[-2],
            dims[-1],
            kernel_sizes[-1],
            strides[-1],
            paddings[-1],
            bias=bias,
            padding_mode=padding_mode,
        )
        self.add_module(f"conv{self.num_layers}", conv_layer)

        if post_activation and self.has_norm:
            shape = calc_out_shape(shape, kernel_sizes[-1], strides[-1], paddings[-1])
            norm_layer = get_norm_2d(norm, dims[-1], shape[0], shape[1])
            self.add_module(f"norm{self.num_layers}", norm_layer)

        if self.has_dropout:
            self.dropout = nn.Dropout2d(dropout_p)

        self.reset_parameters()


class TransposeCNN(_MultilayerModule):

    def __init__(
        self,
        in_dim,
        hidden_dims,
        out_dim,
        kernel_sizes,
        strides,
        paddings,
        nonlinearity,
        param=None,
        norm=None,
        dropout_p=0,
        bias=True,
        padding_mode="zeros",
        in_shape=None,
        pre_activation=False,
        post_activation=False,
        weight_initializer="kaiming_uniform",
        bias_initializer="zeros",
        final_bias_init=None,
    ):
        assert len(kernel_sizes) == len(hidden_dims) + 1
        assert len(strides) == len(kernel_sizes) and len(paddings) == len(kernel_sizes)
        dims = (in_dim,) + tuple(hidden_dims) + (out_dim,)
        super().__init__(
            "conv_transpose",
            3,
            in_dim,
            len(dims) - 1,
            nonlinearity,
            param,
            norm,
            dropout_p,
            pre_activation,
            post_activation,
            weight_initializer,
            bias_initializer,
            final_bias_init,
        )
        if self.unsqueeze:
            dims = (1,) + dims[1:]

        def to_pair(x):
            if isinstance(x, int):
                return x, x
            assert isinstance(x, tuple) and len(x) == 2
            return x

        def calc_out_shape(shape, kernel_size, stride, padding):
            kernel_size, padding, stride = [
                to_pair(x) for x in (kernel_size, stride, padding)
            ]
            return tuple(
                (shape[j] - 1) * stride[j] - 2 * padding[j] + kernel_size[j]
                for j in [0, 1]
            )

        if pre_activation and self.has_norm:
            norm_layer = get_norm_2d(norm, in_dim, in_shape[0], in_shape[1])
            self.add_module("norm0", norm_layer)

        shape = in_shape
        for i in range(self.num_layers - 1):
            conv_transpose_layer = nn.ConvTranspose2d(
                dims[i],
                dims[i + 1],
                kernel_sizes[i],
                strides[i],
                paddings[i],
                bias=bias,
                padding_mode=padding_mode,
            )
            self.add_module(f"conv_transpose{i + 1}", conv_transpose_layer)
            if self.has_norm:
                if shape is not None:
                    shape = calc_out_shape(
                        shape, kernel_sizes[i], strides[i], paddings[i]
                    )
                norm_layer = get_norm_2d(norm, dims[i + 1], shape[0], shape[1])
                self.add_module(f"norm{i + 1}", norm_layer)

        conv_transpose_layer = nn.ConvTranspose2d(
            dims[-2],
            dims[-1],
            kernel_sizes[-1] + 1,  # w is odd
            # kernel_sizes[-1],
            strides[-1],
            paddings[-1],
            bias=bias,
            padding_mode=padding_mode,
        )
        self.add_module(f"conv_transpose{self.num_layers}", conv_transpose_layer)

        if post_activation and self.has_norm:
            shape = calc_out_shape(shape, kernel_sizes[-1], strides[-1], paddings[-1])
            norm_layer = get_norm_2d(norm, dims[-1], shape[0], shape[1])
            self.add_module(f"norm{self.num_layers}", norm_layer)

        if self.has_dropout:
            self.dropout = nn.Dropout2d(dropout_p)

        self.reset_parameters()


# adopted from
# https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py
# and https://github.com/sooftware/attentions/blob/master/attentions.py
class TransformerXLDecoder(nn.Module):

    def __init__(
        self, decoder_layer, num_layers, max_length, mem_length, batch_first=False
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.mem_length = mem_length
        self.batch_first = batch_first

        self.pos_enc = PositionalEncoding(
            decoder_layer.dim, max_length, dropout_p=decoder_layer.dropout_p
        )
        self.u_bias = nn.Parameter(
            torch.Tensor(decoder_layer.num_heads, decoder_layer.head_dim)
        )
        self.v_bias = nn.Parameter(
            torch.Tensor(decoder_layer.num_heads, decoder_layer.head_dim)
        )
        nn.init.xavier_uniform_(self.u_bias)
        nn.init.xavier_uniform_(self.v_bias)

    def init_mems(self):
        if self.mem_length > 0:
            param = next(self.parameters())
            dtype, device = param.dtype, param.device
            mems = []
            for i in range(self.num_layers + 1):
                mems.append(torch.empty(0, dtype=dtype, device=device))
            return mems
        else:
            return None

    def forward(
        self,
        x,
        positions,
        attn_mask,
        mems=None,
        tgt_length=None,
        return_attention=False,
    ):
        if self.batch_first:
            x = x.transpose(0, 1)

        if mems is None:
            mems = self.init_mems()

        if tgt_length is None:
            tgt_length = x.shape[0]
        assert tgt_length > 0

        pos_enc = self.pos_enc(positions)
        hiddens = [x]
        attentions = []
        out = x
        for i, layer in enumerate(self.layers):
            out, attention = layer(
                out,
                pos_enc,
                self.u_bias,
                self.v_bias,
                attn_mask=attn_mask,
                mems=mems[i],
            )
            hiddens.append(out)
            attentions.append(attention)

        out = out[-tgt_length:]

        if self.batch_first:
            out = out.transpose(0, 1)

        assert len(hiddens) == len(mems)
        with torch.no_grad():
            new_mems = []
            for i in range(len(hiddens)):
                cat = torch.cat([mems[i], hiddens[i]], dim=0)
                new_mems.append(cat[-self.mem_length :].detach())
        if return_attention:
            attention = torch.stack(attentions, dim=-2)
            return out, new_mems, attention
        return out, new_mems


class TransformerXLDecoderLayer(nn.Module):

    def __init__(
        self,
        dim,
        feedforward_dim,
        head_dim,
        num_heads,
        activation,
        dropout_p,
        layer_norm_eps=1e-5,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.self_attn = RelativeMultiheadSelfAttention(
            dim, head_dim, num_heads, dropout_p
        )
        self.linear1 = nn.Linear(dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.act = get_activation(activation)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

    def _ff(self, x):
        x = self.linear2(self.dropout(self.act(self.linear1(x))))
        return self.dropout(x)

    def forward(self, x, pos_encodings, u_bias, v_bias, attn_mask=None, mems=None):
        out, attention = self.self_attn(
            x, pos_encodings, u_bias, v_bias, attn_mask, mems
        )
        out = self.dropout(out)
        out = self.norm1(x + out)
        out = self.norm2(out + self._ff(out))
        return out, attention


class RelativeMultiheadSelfAttention(nn.Module):

    def __init__(self, dim, head_dim, num_heads, dropout_p):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.scale = 1 / (dim**0.5)

        self.qkv_proj = nn.Linear(dim, 3 * num_heads * head_dim, bias=False)
        self.pos_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

    def _rel_shift(self, x):
        zero_pad = torch.zeros(
            (x.shape[0], 1, *x.shape[2:]), device=x.device, dtype=x.dtype
        )
        x_padded = torch.cat([zero_pad, x], dim=1)
        x_padded = x_padded.view(x.shape[1] + 1, x.shape[0], *x.shape[2:])
        x = x_padded[1:].view_as(x)
        return x

    def forward(self, x, pos_encodings, u_bias, v_bias, attn_mask=None, mems=None):
        tgt_length, batch_size = x.shape[:2]
        pos_len = pos_encodings.shape[0]

        if mems is not None:
            cat = torch.cat([mems, x], dim=0)
            qkv = self.qkv_proj(cat)
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            q = q[-tgt_length:]
        else:
            qkv = self.qkv_proj(x)
            q, k, v = torch.chunk(qkv, 3, dim=-1)

        pos_encodings = self.pos_proj(pos_encodings)

        src_length = k.shape[0]
        num_heads = self.num_heads
        head_dim = self.head_dim

        q = q.view(tgt_length, batch_size, num_heads, head_dim)
        k = k.view(src_length, batch_size, num_heads, head_dim)
        v = v.view(src_length, batch_size, num_heads, head_dim)
        pos_encodings = pos_encodings.view(pos_len, num_heads, head_dim)

        content_score = torch.einsum("ibnd,jbnd->ijbn", (q + u_bias, k))
        pos_score = torch.einsum("ibnd,jnd->ijbn", (q + v_bias, pos_encodings))
        pos_score = self._rel_shift(pos_score)

        # [tgt_length x src_length x batch_size x num_heads]
        attn_score = content_score + pos_score
        attn_score.mul_(self.scale)

        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_score = attn_score.masked_fill(
                    attn_mask[:, :, None, None], -float("inf")
                )
            elif attn_mask.ndim == 3:
                attn_score = attn_score.masked_fill(
                    attn_mask[:, :, :, None], -float("inf")
                )

        # [tgt_length x src_length x batch_size x num_heads]
        attn = F.softmax(attn_score, dim=1)
        return_attn = attn
        attn = self.dropout(attn)

        context = torch.einsum("ijbn,jbnd->ibnd", (attn, v))
        context = context.reshape(
            context.shape[0], context.shape[1], num_heads * head_dim
        )
        return self.out_proj(context), return_attn


class PositionalEncoding(nn.Module):

    def __init__(self, dim, max_length, dropout_p=0, batch_first=False):
        super().__init__()
        self.dim = dim
        self.max_length = max_length
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

        encodings = torch.zeros(max_length, dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        encodings[:, 0::2] = torch.sin(position * div_term)
        encodings[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("encodings", encodings)

    def forward(self, positions):
        out = self.encodings[positions]
        out = self.dropout(out)
        return out.unsqueeze(0) if self.batch_first else out.unsqueeze(1)


class PredictionNet(nn.Module):

    def __init__(
        self,
        modality_order,
        num_current,
        embeds,
        out_heads,
        embed_dim,
        activation,
        norm,
        dropout_p,
        feedforward_dim,
        head_dim,
        num_heads,
        num_layers,
        memory_length,
        max_length,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.memory_length = memory_length
        self.modality_order = tuple(modality_order)
        self.num_current = num_current

        self.embeds = nn.ModuleDict(
            {
                name: (
                    nn.Embedding(embed["in_dim"], embed_dim)
                    if embed.get("categorical", False)
                    else MLP(
                        embed["in_dim"],
                        [],
                        embed_dim,
                        activation,
                        norm=norm,
                        dropout_p=dropout_p,
                        post_activation=True,
                    )
                )
                for name, embed in embeds.items()
            }
        )

        decoder_layer = TransformerXLDecoderLayer(
            embed_dim, feedforward_dim, head_dim, num_heads, activation, dropout_p
        )

        num_modalities = len(modality_order)
        max_length = max_length * num_modalities + self.num_current
        mem_length = memory_length * num_modalities + self.num_current
        self.transformer = TransformerXLDecoder(
            decoder_layer, num_layers, max_length, mem_length, batch_first=True
        )

        self.out_heads = nn.ModuleDict(
            {
                name: MLP(
                    embed_dim,
                    head["hidden_dims"],
                    head["out_dim"],
                    activation,
                    norm=norm,
                    dropout_p=dropout_p,
                    pre_activation=True,
                    final_bias_init=head.get("final_bias_init", None),
                )
                for name, head in out_heads.items()
            }
        )

    @lru_cache(maxsize=20)
    def _get_base_mask(self, src_length, tgt_length, device):
        src_mask = torch.ones(tgt_length, src_length, dtype=torch.bool, device=device)
        num_modalities = len(self.modality_order)
        for tgt_index in range(tgt_length):
            # the last indices are always 'current'
            start_index = src_length - self.num_current
            src_index = src_length - tgt_length + tgt_index
            modality_index = (src_index - start_index) % num_modalities
            if modality_index < self.num_current:
                start = max(src_index - (self.memory_length + 1) * num_modalities, 0)
            else:
                start = max(
                    src_index - modality_index - self.memory_length * num_modalities, 0
                )
            src_mask[tgt_index, start : src_index + 1] = False
        return src_mask

    def _get_mask(self, src_length, tgt_length, device, batch_size):
        # prevent attention over episode ends using stop_mask
        src_mask = self._get_base_mask(src_length, tgt_length, device)
        src_mask = src_mask.unsqueeze(-1).tile(1, 1, batch_size)
        return src_mask

    @staticmethod
    def same_batch_shape(tensors, ndim=2):
        batch_shape = tensors[0].shape[:ndim]
        assert all(t.ndim >= ndim for t in tensors)
        return all(
            tensors[i].shape[:ndim] == batch_shape for i in range(1, len(tensors))
        )

    def forward(
        self,
        inputs,
        tgt_length,
        heads=None,
        mems=None,
        return_attention=False,
    ):
        modality_order = self.modality_order
        num_modalities = len(modality_order)
        num_current = self.num_current

        assert self.same_batch_shape(
            [inputs[name] for name in modality_order[:num_current]], 1
        )
        if num_modalities > num_current:
            assert self.same_batch_shape(
                [inputs[name] for name in modality_order[num_current:]]
            )

        embeds = {name: mod(inputs[name]) for name, mod in self.embeds.items()}

        def cat_modalities(xs):
            batch_size, seq_len, dim = xs[0].shape
            return torch.cat(xs, dim=2).reshape(batch_size, seq_len * len(xs), dim)

        if mems is None:
            history_length = embeds[modality_order[0]].shape[1] - 1
            if num_modalities == num_current:
                inputs = cat_modalities([embeds[name] for name in modality_order])
            else:
                history = cat_modalities(
                    [embeds[name][:, :history_length] for name in modality_order]
                )
                current = cat_modalities(
                    [
                        embeds[name][:, history_length:]
                        for name in modality_order[:num_current]
                    ]
                )
                inputs = torch.cat([history, current], dim=1)
            tgt_length = (tgt_length - 1) * num_modalities + num_current
            src_length = history_length * num_modalities + num_current
            assert inputs.shape[1] == src_length
            src_mask = self._get_mask(src_length, src_length, inputs.device, inputs.shape[0])
        else:
            sequence_length = embeds[modality_order[0]].shape[1]
            # switch order so that 'currents' are last
            inputs = cat_modalities(
                [
                    embeds[name]
                    for name in (
                        modality_order[num_current:] + modality_order[:num_current]
                    )
                ]
            )
            tgt_length = tgt_length * num_modalities
            mem_length = mems[0].shape[0]
            src_length = mem_length + sequence_length * num_modalities
            src_mask = self._get_mask(src_length, tgt_length, inputs.device, inputs.shape[0])

        positions = torch.arange(src_length - 1, -1, -1, device=inputs.device)
        outputs = self.transformer(
            inputs,
            positions,
            attn_mask=src_mask,
            mems=mems,
            tgt_length=tgt_length,
            return_attention=return_attention,
        )
        hiddens, mems, attention = outputs if return_attention else (outputs + (None,))

        # take outputs at last current
        assert hiddens.shape[1] == tgt_length
        out_idx = torch.arange(
            tgt_length - 1, -1, -num_modalities, device=inputs.device
        ).flip([0])
        hiddens = hiddens[:, out_idx]
        if return_attention:
            attention = attention[out_idx]

        if heads is None:
            heads = self.out_heads.keys()

        out = {name: self.out_heads[name](hiddens) for name in heads}

        return (
            (out, hiddens, mems, attention)
            if return_attention
            else (out, hiddens, mems)
        )
