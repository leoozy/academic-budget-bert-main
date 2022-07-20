# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
# !/usr/bin/env python
# -*- coding: utf-8 -*-
import math
from turtle import forward
from sklearn import neighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq

# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from timm.models.helpers import load_pretrained
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from timm.models.resnet import resnet26d, resnet50d
# from timm.models.registry import register_model

from pretraining.modeling import BertOnlyMLMHead
from pretraining.gbert_lib import Grapher, act_layer
import pdb


# def _cfg(url='', **kwargs):
#     return {
#         'url': url,
#         'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
#         'crop_pct': .9, 'interpolation': 'bicubic',
#         'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
#         'first_conv': 'patch_embed.proj', 'classifier': 'head',
#         **kwargs
#     }


# default_cfgs = {
#     'vig_224_gelu': _cfg(
#         mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
#     ),
#     'vig_b_224_gelu': _cfg(
#         crop_pct=0.95, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
#     ),
# }


class FFN(nn.Module):
    def __init__(self, config, in_features, hidden_features=None, out_features=None, act='relu'):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.norm = nn.LayerNorm(out_features, eps=config.layer_norm_eps)
        self.act = act_layer(act)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        ###TODO ADD NORM
        x = self.fc2(x)
        x = self.drop(x) + shortcut
        x = self.norm(x)
        return x  # .reshape(B, C, N, 1)



class Embedding(nn.Module):
    """ Sentence Embedding
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
       # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        # End copy
        self.padding_idx = config.pad_token_id
        if self.position_embedding_type == "absolute":
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
            )
        elif self.position_embedding_type == "relative":
            raise Exception("not implement")

    def forward(
                self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None,
                past_key_values_length=0
        ):
            # Create the position ids from the input token ids. Any padded tokens remain padded.#?
            ##TODO why use this position_ids
            if position_ids is None:
                if input_ids is not None:
                    # Create the position ids from the input token ids. Any padded tokens remain padded.
                    position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx,
                                                                      past_key_values_length)
                else:
                    position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

            if input_ids is not None:
                input_shape = input_ids.size()
            else:
                input_shape = inputs_embeds.size()[:-1]

            seq_length = input_shape[1]
            # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
            # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
            # issue #5664
            if token_type_ids is None:
                if hasattr(self, "token_type_ids"):
                    buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                    buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                    token_type_ids = buffered_token_type_ids_expanded
                else:
                    token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)

            embeddings = inputs_embeds + token_type_embeddings
            if self.position_embedding_type == "absolute":
                position_embeddings = self.position_embeddings(position_ids)
                embeddings += position_embeddings
            elif self.position_embedding_type == 'relative':
                raise Exception("not implement")

            embeddings = self.LayerNorm(embeddings)
            embeddings = self.dropout(embeddings)
            return embeddings




class DeepGCN(torch.nn.Module):
    def __init__(self, opt, args):
        super(DeepGCN, self).__init__()
        print(opt)
        act = opt.hidden_act
        conv = opt.conv
        hidden_size = opt.hidden_size
        intermediate_size = opt.intermediate_size
        depth = opt.num_hidden_layers
        reduce_ratios = [1] * depth
        # blocks = opt.blocks
        # self.n_blocks = sum(blocks)
        # channels = opt.channels
        # dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule

        self.embed = Embedding(opt)
        self.backbone = nn.ModuleList([])

        for i in range(depth):
            self.backbone += [
                Seq(Grapher(hidden_size=hidden_size,
                            conv=conv,
                            r=reduce_ratios[i],
                            relative_pos=(opt.position_embedding_type == "relative"),
                            config=opt),
                    FFN(opt, hidden_size, intermediate_size, act=act)
                    )]

        self.backbone = Seq(*self.backbone)
        """
        self.prediction = Seq(nn.Conv2d(channels[-1], 1024, 1, bias=True),
                              nn.BatchNorm2d(1024),
                              act_layer(act),
                              nn.Dropout(opt.dropout),
                              nn.Conv2d(1024, opt.n_classes, 1, bias=True))
        """
        # self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def maybe_layer_norm(self, hidden_states, layer_norm, current_ln_mode):
        if self.config.useLN and self.config.encoder_ln_mode in current_ln_mode:
            return layer_norm(hidden_states)
        else:
            return hidden_states

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=self.embed.word_embeddings.weight.dtype  # should be of same dtype
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        input_ids = self.embed(input_ids=input_ids, token_type_ids=token_type_ids)
        x = input_ids
        for i in range(len(self.backbone)):
            x = self.backbone[i]((x, extended_attention_mask))
        return x



# @register_model
def graphBert(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
            self.neighbor_mode = 'full'
            self.conv = 'attn'  # graph conv layer {attn}
            self.hidden_act = 'gelu'  # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'layer'  # batch or instance normalization {batch, instance}
            self.bias = True  # bias of conv layer True or False
            self.dropout = 0.0  # dropout rate
            self.epsilon = 0.2  # stochastic epsilon for gcn
            self.num_hidden_layers = 12  # number of basic blocks in the backbone
            self.n_classes = num_classes  # Dimension of out_channels


            # for bert config
            self.hidden_size = 768  # Dimension of embeddings
            self.vocab_size = 30522
            self.pad_token_id = 0
            self.type_vocab_size = 2
            self.layer_norm_eps = 1e-12
            self.num_attention_heads = 12
            self.attention_probs_dropout_prob = 0.1
            self.max_position_embeddings = 512
            self.hidden_dropout_prob = 0.1
            self.intermediate_size = 3072
            self.position_embedding_type='absolute'
    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    # model.default_cfg = default_cfgs['vig_224_gelu']
    return model



def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.
    Args:
        x: torch.Tensor x:
    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx



# class BertOnlyMLMHead(nn.Module):
#     def __init__(self, config, bert_model_embedding_weights):
#         super(BertOnlyMLMHead, self).__init__()
#         self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

#     def forward(self, sequence_output, masked_token_indexes=None):
#         prediction_scores = self.predictions(sequence_output, masked_token_indexes)
#         return prediction_scores

class GraphBert(nn.Module):
    def __init__(self, config, args):
        super(GraphBert, self).__init__()
        self.gbert = DeepGCN(config, args)
        self.config = config
        self.cls = BertOnlyMLMHead(config, self.gbert.embed.word_embeddings.weight)
        self.apply(self._init_weights)

    def forward(self, batch):


        #input_ids = batch[1]
        #token_type_ids = batch[3]
        #attention_mask = batch[2]
       # masked_lm_labels = batch[4]

        input_ids = batch
        token_type_ids = None
        attention_mask = None
        masked_lm_labels = None

        gbert_output = self.gbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # follow the implementation in modeling.py
        if masked_lm_labels is None:
            prediction_scores = self.cls(gbert_output)
            return prediction_scores

        masked_token_indexes = torch.nonzero((masked_lm_labels + 1).view(-1), as_tuple=False).view(
            -1
        )
        prediction_scores = self.cls(gbert_output, masked_token_indexes)

        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            target = torch.index_select(masked_lm_labels.view(-1), 0, masked_token_indexes)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), target)

            outputs = (masked_lm_loss,)
            # if output_attentions:
            #     outputs += (bert_output[-1],)

            return outputs
        else:
            return prediction_scores
    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

if __name__ == "__main__":
    model = graphBert()
    senten = torch.randint(low=0, high=20000, size=(16, 100))
    model(senten)