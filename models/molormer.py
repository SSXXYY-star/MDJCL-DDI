from __future__ import print_function
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from math import sqrt


class Molormer(nn.Sequential):
    '''
        Molormer Network with spatial graph encoder and lightweight attention block
    '''

    def __init__(self, args):
        super(Molormer, self).__init__()
        self.args = args
        self.gpus = torch.cuda.device_count()
        self.num_layers = args.Mol_num_layers
        self.num_heads = args.Mol_num_heads
        self.hidden_dim = args.Mol_hidden_dim
        self.inter_dim = args.Mol_inter_dim
        self.flatten_dim = args.Mol_flatten_dim
        self.multi_hop_max_dist = args.Mol_longest_path

        # dropout
        self.encoder_dropout = args.Mol_encoder_dropout_rate
        self.attention_dropout = args.Mol_attention_dropout_rate
        self.input_dropout = nn.Dropout(args.Mol_input_dropout_rate)

        # Embeddings
        self.d_node_encoder = nn.Embedding(512 * 9 + 1, self.hidden_dim, padding_idx=0)
        self.d_edge_encoder = nn.Embedding(512 * 3 + 1, self.num_heads, padding_idx=0)
        self.d_edge_dis_encoder = nn.Embedding(128 * self.num_heads * self.num_heads, 1)
        self.d_spatial_pos_encoder = nn.Embedding(512, self.num_heads, padding_idx=0)
        self.d_in_degree_encoder = nn.Embedding(512, self.hidden_dim, padding_idx=0)
        self.d_out_degree_encoder = nn.Embedding(512, self.hidden_dim, padding_idx=0)

        self.d_encoders = Encoder(hidden_dim=self.hidden_dim, inter_dim=self.inter_dim,
                                  n_layers=self.num_layers, n_heads=self.num_heads)  # 注意力+卷积

        self.d_final_ln = nn.LayerNorm(self.hidden_dim)
        self.d_graph_token = nn.Embedding(1, self.hidden_dim)
        self.graph_token_virtual_distance = nn.Embedding(1, self.num_heads)
        self.icnn = nn.Conv1d(self.hidden_dim, 16, 3)
        # self.decoder = nn.Sequential(
        #     nn.Linear(self.flatten_dim, 512),
        #     nn.ReLU(True),
        #     nn.BatchNorm1d(512),
        #     nn.Linear(512, 256),
        #     nn.ReLU(True),
        #     nn.BatchNorm1d(256),
        #     nn.Linear(256, config['num_classes'])
        # )

    def forward(self, d1_node, d1_attn_bias, d1_spatial_pos, d1_in_degree, d1_out_degree, d1_edge_input,
                d2_node, d2_attn_bias, d2_spatial_pos, d2_in_degree, d2_out_degree, d2_edge_input):
        drug1_n_graph, drug1_n_node = d1_node.size()[:2]  # 2,256
        drug2_n_graph, drug2_n_node = d2_node.size()[:2]

        drug1_graph_attn_bias = d1_attn_bias.clone()
        # print('drug1_graph_attn_bias: ',drug1_graph_attn_bias.shape) torch.Size([2, 257, 257])
        # assert False
        drug1_graph_attn_bias = drug1_graph_attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # 为多头注意力机制而重复
        # print('drug1_graph_attn_bias: ',drug1_graph_attn_bias.shape)  #torch.Size([2, 8, 257, 257])
        # assert False

        drug2_graph_attn_bias = d2_attn_bias.clone()
        drug2_graph_attn_bias = drug2_graph_attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        # print('d1_spatial_pos: ',d1_spatial_pos.shape) # torch.Size([2, 256, 256])
        drug1_spatial_pos_bias = self.d_spatial_pos_encoder(d1_spatial_pos).permute(0, 3, 1,
                                                                                    2)  # 把两点间最短路径emb成8位并转换维度对应8个头
        # print('self.d_spatial_pos_encoder(d1_spatial_pos): ',self.d_spatial_pos_encoder(d1_spatial_pos).shape) #[2, 256, 256, 8]
        # print('drug1_spatial_pos_bias: ',drug1_spatial_pos_bias.shape) #[2, 8, 256, 256]
        # assert False

        # 取前两个维度的全部，第三四个维度从第二个开始取
        drug1_graph_attn_bias[:, :, 1:, 1:] = drug1_graph_attn_bias[:, :, 1:,
                                              1:] + drug1_spatial_pos_bias  # [2, 8, 257, 257] 掩码

        drug2_spatial_pos_bias = self.d_spatial_pos_encoder(d2_spatial_pos).permute(0, 3, 1, 2)
        drug2_graph_attn_bias[:, :, 1:, 1:] = drug2_graph_attn_bias[:, :, 1:, 1:] + drug2_spatial_pos_bias

        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)  # 每个头的权重
        # print(self.graph_token_virtual_distance.weight.shape) #[1,8]
        # print(t.shape) #[1,8,1]
        # assert False
        drug1_graph_attn_bias[:, :, 1:, 0] = drug1_graph_attn_bias[:, :, 1:, 0] + t  # 第一列，从第二行开始 相同的值
        drug1_graph_attn_bias[:, :, 0, :] = drug1_graph_attn_bias[:, :, 0, :] + t  # 第一行
        drug2_graph_attn_bias[:, :, 1:, 0] = drug2_graph_attn_bias[:, :, 1:, 0] + t
        drug2_graph_attn_bias[:, :, 0, :] = drug2_graph_attn_bias[:, :, 0, :] + t

        drug1_spatial_pos = d1_spatial_pos.clone()  # [2, 256, 256]

        drug1_spatial_pos[drug1_spatial_pos == 0] = 1  # set pad to 1

        drug1_spatial_pos = torch.where(drug1_spatial_pos > 1, drug1_spatial_pos - 1, drug1_spatial_pos)

        drug1_spatial_pos = drug1_spatial_pos.clamp(0, self.multi_hop_max_dist)

        drug1_edge_input = d1_edge_input[:, :, :, :self.multi_hop_max_dist, :]

        # [n_graph, n_node, n_node, max_dist, n_head]
        drug1_edge_input = self.d_edge_encoder(drug1_edge_input).mean(-2)  # 对最短路径上的每个边embedding，并将最短路径上边emb平均

        max_dist = drug1_edge_input.size(-2)
        drug1_edge_input_flat = drug1_edge_input.permute(3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
        drug1_edge_input_flat = torch.bmm(drug1_edge_input_flat,
                                          self.d_edge_dis_encoder.weight.reshape(-1, self.num_heads, self.num_heads)[
                                          :max_dist, :, :])
        drug1_edge_input = drug1_edge_input_flat.reshape(max_dist, drug1_n_graph, drug1_n_node, drug1_n_node,
                                                         self.num_heads).permute(1, 2, 3, 0, 4)  # 2,43,43,20,8
        drug1_edge_input = (drug1_edge_input.sum(-2) / (drug1_spatial_pos.float().unsqueeze(-1))).permute(0, 3, 1, 2)  # 归一化 得到8个头，每个头对应一个最短路径(两节点关系)的权重

        #################################################################################################################

        # edge_input
        drug2_spatial_pos = d2_spatial_pos.clone()
        drug2_spatial_pos[drug2_spatial_pos == 0] = 1  # set pad to 1

        drug2_spatial_pos = torch.where(drug2_spatial_pos > 1, drug2_spatial_pos - 1, drug2_spatial_pos)
        drug2_spatial_pos = drug2_spatial_pos.clamp(0, self.multi_hop_max_dist)
        drug2_edge_input = d2_edge_input[:, :, :, :self.multi_hop_max_dist, :]

        # [n_graph, n_node, n_node, max_dist, n_head]
        drug2_edge_input = self.d_edge_encoder(drug2_edge_input).mean(-2)

        max_dist = drug2_edge_input.size(-2)
        drug2_edge_input_flat = drug2_edge_input.permute(3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
        drug2_edge_input_flat = torch.bmm(drug2_edge_input_flat,
                                          self.d_edge_dis_encoder.weight.reshape(-1, self.num_heads, self.num_heads)[
                                          :max_dist, :, :])
        drug2_edge_input = drug2_edge_input_flat.reshape(max_dist, drug2_n_graph, drug2_n_node, drug2_n_node,
                                                         self.num_heads).permute(1, 2, 3, 0, 4)  # 2,43,43,20,8
        drug2_edge_input = (drug2_edge_input.sum(-2) / (drug2_spatial_pos.float().unsqueeze(-1))).permute(0, 3, 1,
                                                                                                          2)  # 2,8,43,43

        drug1_graph_attn_bias[:, :, 1:, 1:] = drug1_graph_attn_bias[:, :, 1:, 1:] + drug1_edge_input
        drug1_graph_attn_bias = drug1_graph_attn_bias + d1_attn_bias.unsqueeze(1)  # 得到注意力偏置

        drug2_graph_attn_bias[:, :, 1:, 1:] = drug2_graph_attn_bias[:, :, 1:, 1:] + drug2_edge_input
        drug2_graph_attn_bias = drug2_graph_attn_bias + d2_attn_bias.unsqueeze(1)

        # node feauture + graph token

        # print("d1_node: ",d1_node,d1_node.shape) -------torch.Size([2, 256, 9])
        # print("d1_nnEmbedding: ",self.d_node_encoder(d1_node).shape) ------torch.Size([2, 256, 9, 256])
        drug1_node_feature = self.d_node_encoder(d1_node).sum(
            dim=-2)  # 把分子的最多256个原子编码从9扩展到256，原理为对每个原子编码进行embed，然后把embed加和
        # print("drug1_node_feature: ",drug1_node_feature.shape) ------torch.Size([2, 256, 256])

        # print("d1_in_degree: ", d1_in_degree.shape) ------torch.Size([2, 256])
        drug1_node_feature = drug1_node_feature + self.d_in_degree_encoder(d1_in_degree) + self.d_out_degree_encoder(
            d1_out_degree)  # 公式9
        # print("d_in_degree_encoder: ")  -------torch.Size([2, 256, 256])
        # print("drug1_node_feature: ")  -----torch.Size([2, 256, 256])

        drug1_graph_token_feature = self.d_graph_token.weight.unsqueeze(0).repeat(drug1_n_graph, 1, 1)  # [2, 1, 256]
        drug1_graph_node_feature = torch.cat([drug1_graph_token_feature, drug1_node_feature], dim=1)  # [2, 257, 256]

        drug2_node_feature = self.d_node_encoder(d2_node).sum(dim=-2)
        drug2_node_feature = drug2_node_feature + self.d_in_degree_encoder(d2_in_degree) + self.d_out_degree_encoder(
            d2_out_degree)
        drug2_graph_token_feature = self.d_graph_token.weight.unsqueeze(0).repeat(drug2_n_graph, 1, 1)
        drug2_graph_node_feature = torch.cat([drug2_graph_token_feature, drug2_node_feature], dim=1)

        # transfomrer encoder
        drug1_output = self.input_dropout(drug1_graph_node_feature)  # [2, 257, 256]-->[2, 257, 256]
        drug1_output = self.d_encoders(drug1_output, drug1_graph_attn_bias)  # probSparse self-attention 公式11

        drug2_output = self.input_dropout(drug2_graph_node_feature)
        drug2_output = self.d_encoders(drug2_output, drug2_graph_attn_bias)

        return drug1_output, drug2_output



class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super(AttentionLayer, self).__init__()

        key_dim = hidden_dim // n_heads
        value_dim = hidden_dim // n_heads

        self.inner_attention = ProbAttention(False, factor=5, attention_dropout=0.0, output_attention=False)
        self.query_projection = nn.Linear(hidden_dim, key_dim * n_heads)
        self.key_projection = nn.Linear(hidden_dim, key_dim * n_heads)
        self.value_projection = nn.Linear(hidden_dim, value_dim * n_heads)
        self.out_projection = nn.Linear(value_dim * n_heads, hidden_dim)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)  # todo 在这里改变的维度
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values, mask)

        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top, atom_mask):  # n_top: c*ln(L_q)

        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)  # todo 把lk lq的长度设为掩码后的长度
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # todo lk变，lq不变；即只限制随机值，拓展为[32,257,30]
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)  # 会议公式4
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:

            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)  # qk/d *v
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u, atom_mask=mask)

        scale = self.scale or 1. / sqrt(D)  # 1/根号d
        if scale is not None:
            scores_top = scores_top * scale  # qk除根号d

        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, mask)

        return context.transpose(2, 1).contiguous(), attn


class Encoder(nn.Module):
    def __init__(self, hidden_dim, inter_dim, n_layers, n_heads, dropout=0.0):  # 255，255，3
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(
            Encoder_layer(hidden_dim, inter_dim, n_heads, dropout) for l in range(n_layers))  # self-att+卷积
        self.conv_layers = nn.ModuleList(Distilling_layer(hidden_dim) for _ in range(n_layers - 1))  # 一个卷积池化层
        self.norm = torch.nn.LayerNorm(hidden_dim)

    def forward(self, x, mask=None):
        attns = []
        for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
            x, attn = attn_layer(x, mask=mask)
            # x = conv_layer(x)
            attns.append(attn)
        x, attn = self.attn_layers[-1](x, mask=mask)
        attns.append(attn)

        x = self.norm(x)

        return x


class Encoder_layer(nn.Module):
    def __init__(self, hidden_dim, inter_dim, n_heads, dropout):
        super(Encoder_layer, self).__init__()
        self.attention = AttentionLayer(hidden_dim=hidden_dim, n_heads=n_heads)  # 自注意力
        self.conv1 = nn.Conv1d(hidden_dim, inter_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(inter_dim, hidden_dim, kernel_size=1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = F.relu

    def forward(self, x, mask=None):
        attn_x, attn = self.attention(x, x, x, mask=mask)
        x = x + self.dropout(attn_x)
        y = x = self.norm1(x)
        y = self.dropout(self.relu(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Distilling_layer(nn.Module):
    def __init__(self, channel):
        super(Distilling_layer, self).__init__()

        self.conv = nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=3, padding=1,
                              padding_mode='circular')
        self.norm = nn.BatchNorm1d(channel)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1))
        out = self.maxPool(self.activation(self.norm(x))).transpose(1, 2)

        return out
