# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import numpy as np
import torch
from torch import nn
from .torch_nn import BasicConv, batched_index_select, act_layer
from .torch_edge import DenseDilatedKnnGraph
from .pos_embed import get_2d_relative_pos_embed
import torch.nn.functional as F
from timm.models.layers import DropPath
from typing import List, Optional, Tuple, Union
import math
import pdb


class MRConv2d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)
        return self.nn(x)


class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value


class GraphSAGE(nn.Module):
    """
    GraphSAGE Graph Convolution (Paper: https://arxiv.org/abs/1706.02216) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GraphSAGE, self).__init__()
        self.nn1 = BasicConv([in_channels, in_channels], act, norm, bias)
        self.nn2 = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(self.nn1(x_j), -1, keepdim=True)
        return self.nn2(torch.cat([x, x_j], dim=1))


class GINConv2d(nn.Module):
    """
    GIN Graph Convolution (Paper: https://arxiv.org/abs/1810.00826) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GINConv2d, self).__init__()
        self.nn = BasicConv([in_channels, out_channels], act, norm, bias)
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init]))

    def forward(self, x, edge_index, y=None):

        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j = torch.sum(x_j, -1, keepdim=True)
        return self.nn((1 + self.eps) * x + x_j)

class AttnConv(nn.Module):
    def __init__(self, config):
        super(AttnConv, self).__init__()
        self.selfatten = RobertaSelfAttention(config)

    def forward(self, x, edge_index, attention_mask):
        # assert attention_mask is not None
        hidden_states_neighbor = batched_index_select(x, edge_index)  # B, L, k, D
        x = self.selfatten(x, hidden_states_neighbor, attention_mask)
        return x




class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, opt,  conv='edge'):
        super(GraphConv2d, self).__init__()
        if conv == 'attn':
            self.gconv = AttnConv(opt)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, edge_index, attn_mask=None):

        return self.gconv(x, edge_index, attn_mask)


class DyGraphConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, opt, conv='attn', r=1):
        super(DyGraphConv2d, self).__init__(opt, conv)
        self.r = r
        self.neighbor_mode = opt.neighbor_mode
        self.dilated_knn_graph = DenseDilatedKnnGraph(self.neighbor_mode)
        self.output = RobertaSelfOutput(opt)

    def forward(self, hidden_states, attention_mask, relative_pos=None):
        _temp = hidden_states
        hidden_states = hidden_states.unsqueeze(-1).permute(0, 2, 1, 3)
        if self.neighbor_mode == "full":
            edge_index = self.dilated_knn_graph(hidden_states, relative_pos)  # B, L, L
            hidden_states = super(DyGraphConv2d, self).forward(hidden_states, edge_index, attention_mask)
            x = self.output(hidden_states, _temp)
        else:
            raise NotImplementedError("neighbor_mode : {} is not implemented".format(self.neighbor_mode))
        return x # x.reshape(B, -1, L).contiguous()


class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, hidden_size, conv='attn', r=1, relative_pos=False, config=None):
        super(Grapher, self).__init__()
        self.channels = hidden_size
        self.r = r

        self.fc1 = nn.Sequential(
            nn.Linear(self.channels, self.channels),
            ##TODO drop? layernorm?
            nn.Dropout(config.hidden_dropout_prob),
            nn.LayerNorm(self.channels, eps=config.layer_norm_eps)
        )


        self.graph_conv = DyGraphConv2d(opt=config,
                                        conv=conv,
                                        r=r)
        self.fc2 = nn.Sequential(
            nn.Linear(self.channels, self.channels),
            ##TODO drop? layernorm?
            nn.Dropout(config.hidden_dropout_prob),
            nn.LayerNorm(self.channels, eps=config.layer_norm_eps)
        )
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.relative_pos = None
        # TODO relative_pos
    def forward(self, inputs):
        hidden_states, attention_mask = inputs
        _tmp = hidden_states
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.graph_conv(hidden_states, attention_mask, self.relative_pos)
        hidden_states = self.fc2(hidden_states)
        hidden_states = hidden_states + _tmp
        return hidden_states

class RobertaSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)


    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states_center: torch.Tensor, # B, D, L
        hidden_states_neighbor: torch.Tensor, # B, L, k, D
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:

        B, D, L, K = hidden_states_neighbor.shape
        assert hidden_states_center.shape == torch.Size([B, D, L, 1])
        hidden_states_center = hidden_states_center.permute(0, 2, 3, 1).contiguous() # B L 1 D
        hidden_states_center = hidden_states_center.reshape(-1, 1, D) # B*L, 1, D
        hidden_states_neighbor = hidden_states_neighbor.permute(0, 3, 1, 2).contiguous() #B L K D
        hidden_states_neighbor = hidden_states_neighbor.reshape(-1, K, D) #B*L, K, D

        mixed_query_layer = self.query(hidden_states_center) #  B*L, 1, D

        #CROSS ATTENTION
        key_layer = self.transpose_for_scores(self.key(hidden_states_neighbor)) # B*L, C, K, D/C
        value_layer = self.transpose_for_scores(self.value(hidden_states_neighbor)) # B*L, C, K, D/C

        query_layer = self.transpose_for_scores(mixed_query_layer) # B*L, C, 1, D/C

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # B*L, C, 1, K
        # TODO if self.position_embedding_type == "relative_key"

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_mask.reshape(-1)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer) #B*L, C, 1, D/C
        context_layer = context_layer.reshape(B, L, 1, self.num_attention_heads, self.attention_head_size).squeeze() # B, L, C, D/C
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer

class RobertaSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
