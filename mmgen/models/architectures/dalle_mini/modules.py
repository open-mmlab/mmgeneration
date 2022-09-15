# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn.bricks import Linear, build_activation_layer, build_norm_layer
from mmgen.registry import MODULES


@MODULES.register_module()
class GLU(nn.Module):
    """GLU variants used to improve Transformer.

    Args:
        in_out_channels (int): The channel number of the input
                                and the output feature map.
        mid_channels (int): The channel number of the middle layer feature map.
    """

    def __init__(self, in_out_channels, mid_channels):
        super().__init__()
        _, self.norm1 = build_norm_layer(dict(type='LN'), in_out_channels)
        _, self.norm2 = build_norm_layer(dict(type='LN'), mid_channels)
        self.fc1 = Linear(in_out_channels, mid_channels, bias=False)
        self.fc2 = Linear(in_out_channels, mid_channels, bias=False)
        self.fc3 = Linear(mid_channels, in_out_channels, bias=False)
        self.gelu = build_activation_layer(dict(type='GELU'))

    def forward(self, z):
        """Forward function.

        Args:
            z (torch.FloatTensor): Input feature map.

        Returns:
            z (torch.FloatTensor): Output feature map.
        """
        z = self.norm1(z)
        w = self.fc1(z)
        w = self.gelu(w)
        v = self.fc2(z)
        z = self.norm2(w * v)
        z = self.fc3(z)
        return z


@MODULES.register_module()
class AttentionBase(nn.Module):
    """An Muti-head Attention block used in Bart model.

    Ref:
    https://github.com/kuprel/min-dalle/blob/main/min_dalle/models

    Args:
        in_channels (int): The channel number of the input feature map.
        num_heads (int): Number of heads in the attention.
    """

    def __init__(self, in_channels, num_heads):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.querie = Linear(in_channels, in_channels, bias=False)
        self.key = Linear(in_channels, in_channels, bias=False)
        self.value = Linear(in_channels, in_channels, bias=False)
        self.proj = Linear(in_channels, in_channels, bias=False)

    def qkv(self, x):
        """Calculate queries, keys and values for the embedding map.

        Args:
            x (torch.FloatTensor): Input feature map.

        Returns:
            q (torch.FloatTensor): Querie feature map.
            k (torch.FloatTensor): Key feature map.
            v (torch.FloatTensor): Value feature map.
        """
        q = self.querie(x)
        k = self.key(x)
        v = self.value(x)

        return q, k, v

    def forward(self, q, k, v, attention_mask):
        """Forward function for attention.

        Args:
            q (torch.FloatTensor): Querie feature map.
            k (torch.FloatTensor): Key feature map.
            v (torch.FloatTensor): Value feature map.
            attention_mask (torch.BoolTensor): whether to use
                                                an attention mask.

        Returns:
            weights (torch.FloatTensor): Feature map after attention.
        """
        q = q.reshape(q.shape[:2] + (self.num_heads, -1))
        q /= q.shape[-1]**0.5
        k = k.reshape(k.shape[:2] + (self.num_heads, -1))
        v = v.reshape(v.shape[:2] + (self.num_heads, -1))

        attention_bias = (1 - attention_mask.to(torch.float32)) * -1e12
        weights = torch.einsum('bqhc,bkhc->bhqk', q, k)
        weights += attention_bias
        weights = torch.softmax(weights, -1)
        weights = torch.einsum('bhqk,bkhc->bqhc', weights, v)
        shape = weights.shape[:2] + (self.in_channels, )
        weights = weights.reshape(shape)
        weights = self.proj(weights)
        return weights


@MODULES.register_module()
class BartEncoderLayer(nn.Module):
    # yapf: disable
    """EncoderLayer of the Bart model.

    Ref:
    https://github.com/kuprel/min-dalle/blob/main/min_dalle/models

    Args:
        in_channels (int): The channel number of the input feature map.
        head_num (int): Number of heads in the attention.
        out_channels (int): The channel number of the output feature map.
    """

    def __init__(self, in_channels, head_num, out_channels):
        super().__init__()
        self.attn = AttentionBase(in_channels, head_num)
        _, self.norm = build_norm_layer(dict(type='LN'), in_channels)
        self.glu = GLU(in_channels, out_channels)

    def forward(self, x, attention_mask):
        """Forward function for the encoder layer.

        Args:
            x (torch.FloatTensor): Input feature map.
            attention_mask (torch.BoolTensor): Whether to use
                                            an attention mask.

        Returns:
            x (torch.FloatTensor): Output feature map.
        """
        
        h = self.norm(x)
        q, k, v = self.attn.qkv(h)
        h = self.attn(q, k, v, attention_mask)
        h = self.norm(h)
        x = x + h
        h = self.glu(x)
        x = x + h
        return x


@MODULES.register_module()
class BartDecoderLayer(nn.Module):
    # yapf: disable
    """DecoderLayer of the Bart model.

    Ref:
    https://github.com/kuprel/min-dalle/blob/main/min_dalle/models

    Args:
        in_channels (int): The channel number of the input feature map.
        head_num (int): Number of heads in the attention.
        out_channels (int): The channel number of the output feature map.
        token_length (int): The length of tokens.
    """

    def __init__(self, in_channels, head_num, out_channels, token_length=256):
        super().__init__()
        self.attn = AttentionBase(in_channels, head_num)
        self.cross_attn = AttentionBase(in_channels, head_num)
        _, self.norm = build_norm_layer(dict(type='LN'), in_channels)
        self.glu = GLU(in_channels, out_channels)
        self.token_indices = torch.arange(token_length)

    def forward(self, x, encoder_state, attention_state,
                attention_mask, token_index):
        """Forward function for the decoder layer.

        Args:
            x (torch.FloatTensor): Input feature map of
                                                the decoder embeddings.
            encoder_state (torch.FloatTensor): Input feature map of
                                                the encoder embeddings.
            attention_state (torch.FloatTensor): Input feature map of
                                                the attention.
            attention_mask (torch.BoolTensor): whether to use
                                                an attention mask.
            token_index (torch.LongTensor): The index of tokens.

        Returns:
            x (torch.FloatTensor): Output feature map of
                                                the decoder embeddings.
            attention_state (torch.FloatTensor): Output feature map of
                                                the attention.
        """

        # Self Attention
        token_count = token_index.shape[1]
        if token_count == 1:
            self_attn_mask = self.token_indices <= token_index
            self_attn_mask = self_attn_mask[:, None, None, :]
        else:
            self_attn_mask = (self.token_indices[None, None, :token_count] <=
                              token_index[:, :, None])
            self_attn_mask = self_attn_mask[:, None, :, :]

        h = self.norm(x)
        q, k, v = self.attn.qkv(h)
        token_count = token_index.shape[1]
        if token_count == 1:
            batch_count = h.shape[0]
            attn_state_new = torch.cat([k, v]).to(attention_state.dtype)
            attention_state[:, token_index[0]] = attn_state_new
            k = attention_state[:batch_count]
            v = attention_state[batch_count:]
        h = self.attn(q, k, v, self_attn_mask)
        h = self.norm(h)
        x = x + h

        # Cross Attention
        h = self.norm(x)
        q, _, _ = self.cross_attn.qkv(h)
        _, k, v = self.cross_attn.qkv(h)
        h = self.cross_attn(q, k, v, attention_mask)
        h = self.norm(h)
        x = x + h

        h = self.glu(x)
        x = x + h

        return x, attention_state
