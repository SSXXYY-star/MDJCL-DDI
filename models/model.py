from __future__ import print_function
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
from models.AMDE import AMDE
from models.GNN import GNN
from models.SchNet import SchNet
from models.cross_att import Cross_MultiAttention
from models.my_molormer import Molormer

torch.manual_seed(1)
np.random.seed(1)


class BilinearDecoder(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.0):
        super(BilinearDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.relation = Parameter(torch.FloatTensor(input_dim, input_dim))
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.relation.data)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs_row = inputs
        inputs_col = inputs.transpose(0, 1)
        inputs_row = self.dropout(inputs_row)
        inputs_col = self.dropout(inputs_col)
        intermediate_product = torch.mm(inputs_row, self.relation)
        rec = torch.mm(intermediate_product, inputs_col)
        rec = nn.ReLU(True)(rec)
        n = rec.size(0)
        # print(n)
        rec = nn.BatchNorm1d(n).cuda()(rec)
        outputs = nn.Linear(n, 1).cuda()(rec)
        print('outputs: ', outputs)
        return outputs


class MultiLevelDDI(nn.Module):
    def __init__(self, args, batch_size=None):
        super(MultiLevelDDI, self).__init__()
        self.args = args
        self.batch_size = batch_size if batch_size else args.batch_size
        self.hidden_dim = args.conv_hidden_dim
        self.input_dropout = nn.Dropout(args.input_dropout_rate)
        self.icnn = nn.Conv1d(self.hidden_dim, 16, 3, 3)
        self.ilin = nn.Linear(1360, 336)
        # self.decoder = BilinearDecoder(539)
        self.decoder_input_dim = 0

        if args.use_SCH:
            self.decoder_input_dim += args.SCH_out_channels
            self.sch1 = SchNet(energy_and_force=False, cutoff=args.SCH_cutoff, num_layers=args.SCH_num_layers,
                               hidden_channels=args.SCH_hidden_channels, num_filters=args.SCH_num_filters,
                               num_gaussians=args.SCH_num_gaussians, out_channels=args.SCH_out_channels)
            self.sch2 = SchNet(energy_and_force=False, cutoff=args.SCH_cutoff, num_layers=args.SCH_num_layers,
                               hidden_channels=args.SCH_hidden_channels, num_filters=args.SCH_num_filters,
                               num_gaussians=args.SCH_num_gaussians, out_channels=args.SCH_out_channels)
        if args.use_AMDE:
            self.decoder_input_dim += args.AMDE_out_dim
            self.amde = AMDE(args)
        if args.use_Mol:
            self.decoder_input_dim += 336
            self.molor = Molormer(args)  #
        if args.use_GNN:
            self.decoder_input_dim += args.GNN_gather_width
            self.gnn = GNN(args)

        self.mod_num = int(args.use_SCH) + int(args.use_Mol) + int(args.use_AMDE) + int(args.use_GNN)

        if args.use_cross_attention:
            assert (args.use_SCH or args.use_Mol or args.use_GNN) and not (args.use_SCH and args.use_Mol and args.use_GNN)

            if self.mod_num == 1 or (self.mod_num == 2 and args.use_AMDE):
                qkv_dim = args.SCH_out_channels if args.use_SCH else args.GNN_gather_width if args.use_GNN else 256
                self.inner_cross = Cross_MultiAttention(qkv_dim, qkv_dim, args.cross_out_dim, args.cross_inner_num_head)
            else:
                # sch在前
                q_dim = args.SCH_out_channels
                kv_dim = args.GNN_gather_width if args.use_GNN else 256

                # mol在前
                # kv_dim = args.SCH_out_channels
                # q_dim = args.GNN_gather_width if args.use_GNN else 256
                self.inner_cross = Cross_MultiAttention(q_dim, kv_dim, args.cross_out_dim, args.cross_inner_num_head)

            self.outer_cross = Cross_MultiAttention(args.cross_out_dim, args.cross_out_dim, args.cross_out_dim, args.cross_outer_num_head)
            self.decoder_input_dim = args.cross_out_dim + (args.AMDE_out_dim if args.use_AMDE else 0)
            self.mod_num = 2 if args.use_AMDE else 1

        # self.decoder = nn.Sequential(
        #     nn.Linear(self.decoder_input_dim, 128),
        #     nn.ReLU(True),
        #     nn.BatchNorm1d(128),
        #     nn.Linear(128, 32),
        #     nn.ReLU(True),
        #     nn.BatchNorm1d(32),
        #     nn.Linear(32, 1)
        # )
        self.decoder = nn.Sequential(
                # nn.Linear(self.flatten_dim, 512),
                nn.Linear(self.decoder_input_dim, 512),
                # nn.Linear(539, 256),
                nn.ReLU(True),
                nn.BatchNorm1d(512),
                nn.Linear(512, 256),
                nn.ReLU(True),
                nn.BatchNorm1d(256),
                nn.Linear(256, args.num_class)
            )
        # 权重归一化
        self.w = nn.Parameter(F.softmax(torch.ones(self.mod_num), dim=0))


    def padding_sch_result(self, batch, batch_atoms):
        """
        The schNet's result is [all_batch_atoms, atom_dim], this method change it to [batch_size, largest_num_atom, atom_dim].
        If atom len less than largest number of atom in mol, padding it by torch.zeros()
        :param adj: each adjacency matrix in batch, like [batch_size, atom_num, atom_num]. This method will abandon the atom that degree is zero
        :param batch_atoms: schNet result, like [all_batch_atoms, atom_dim]
        :return: changed result, like [batch_size, most_atom_in_mol, atom_dim]
        """
        batch_size = self.batch_size
        each_node_num = torch.bincount(batch)
        all_node_num = each_node_num.sum(-1)
        assert all_node_num == batch_atoms.shape[0]
        output = torch.zeros((batch_size, self.args.max_mol_len, batch_atoms.shape[-1])).to(self.args.device)
        cnt = 0
        for i in range(batch_size):
            output[i, :each_node_num[i], :] = batch_atoms[cnt: cnt + each_node_num[i], :]
            cnt += each_node_num[i]
        return output

    def padding_GNN_result(self, batch_atoms):
        """
        :param batch_atoms: GNN's or molormer's result, like [batch_size, batch_largest_num_atom, atom_dim]
        :return: changed result, like [batch_size, most_atom_in_mol, atom_dim]
        """
        batch_size = self.batch_size
        input_atom_num = batch_atoms.shape[1]
        output = torch.zeros((batch_size, self.args.max_mol_len, batch_atoms.shape[-1])).to(self.args.device)
        for i in range(batch_size):
            output[i, :input_atom_num, :] = batch_atoms[i, :, :]
        return output

    def padding_adj(self, batch_adjs):
        """
        :param batch_adjs: each adjacency matrix in batch, like [batch_size, atom_num, atom_num].
        :return: pad result, like [batch_size, most_atom_in_mol, most_atom_in_mol]
        """
        batch_size = self.batch_size
        input_atom_num = batch_adjs.shape[1]
        output = torch.zeros((batch_size, self.args.max_mol_len, self.args.max_mol_len)).to(self.args.device)
        for i in range(batch_size):
            output[i, :input_atom_num, :batch_adjs.shape[-1]] = batch_adjs[i, :, :]
        return output

    def padding_mask(self, batch1, batch2):
        """
         :param batch_adjs2: each adjacency matrix in batch, like [batch_size, atom_num, atom_num].
         :param batch_adjs1:
         :return: pad mask, like [batch_size, most_atom_in_mol, most_atom_in_mol].
         """
        batch_size = self.batch_size
        each_node_num_1 = torch.bincount(batch1)
        each_node_num_2 = torch.bincount(batch2)
        output = torch.zeros((batch_size, self.args.max_mol_len, self.args.max_mol_len)).to(self.args.device).to(self.args.device)
        for i in range(batch_size):
            output[i, :each_node_num_1[i], :each_node_num_2[i]] = torch.ones([each_node_num_1[i], each_node_num_2[i]])
        return output

    def padding_col(self, batch):
        batch_size = self.batch_size
        each_node_num = torch.bincount(batch)
        output = torch.zeros((batch_size, self.args.max_mol_len, self.args.max_mol_len)).to(self.args.device)
        cnt = 0
        for i in range(batch_size):
            output[i, :, :each_node_num[i]] = torch.ones([self.args.max_mol_len, each_node_num[i]])
            cnt += each_node_num[i]
        return output

    def forward(self, d1_node, d1_in_degree, d1_out_degree, d2_node, d2_in_degree, d2_out_degree,
                d1, d2, mask_1, mask_2, d1_positions, d1_z, d1_batch, d2_positions, d2_z, d2_batch,
                adj_1, nd_1, ed_1, adj_2, nd_2, ed_2
                ):

        drug1_n_graph = d1_node.size()[0]
        d1_mol_features, d1_atom_features = [], []
        d2_mol_features, d2_atom_features = [], []
        cnt_mod = 0

        # sequence-based channel
        if self.args.use_AMDE:
            d1_seq_fts_layer1, d2_seq_fts_layer1 = self.amde(d1, d2, mask_1, mask_2)

            d1_mol_features.append(self.w[cnt_mod] * d1_seq_fts_layer1)
            d2_mol_features.append(self.w[cnt_mod] * d1_seq_fts_layer1)
            cnt_mod += 1

        # 3D channel
        if self.args.use_SCH:
            (d1_3d_ft, d1_atom_ft), (d2_3d_ft, d2_atom_ft) = self.sch1((d1_z, d1_positions, d1_batch)), self.sch2((d2_z, d2_positions, d2_batch))
            if not self.args.use_cross_attention:
                d1_mol_features.append(self.w[cnt_mod] * d1_3d_ft)
                d2_mol_features.append(self.w[cnt_mod] * d2_3d_ft)
                cnt_mod += 1
            d1_atom_ft = self.padding_sch_result(d1_batch, d1_atom_ft)
            d2_atom_ft = self.padding_sch_result(d2_batch, d2_atom_ft)
            d1_atom_features.append(d1_atom_ft)
            d2_atom_features.append(d2_atom_ft)

        # semantic info-based channel
        if self.args.use_Mol:
            molor_d1_feature, molor_d2_feature = self.molor(d1_node, d1_in_degree, d1_out_degree,
                                                            d2_node, d2_in_degree, d2_out_degree)
            if not self.args.use_cross_attention:
                molor_d1_feature = self.icnn(molor_d1_feature.permute(0, 2, 1))  # [b, 16, 21])
                d1_feature = self.ilin(molor_d1_feature.view(drug1_n_graph, -1))  # [b, 336]
                molor_d2_feature = self.icnn(molor_d2_feature.permute(0, 2, 1))
                d2_feature = self.ilin(molor_d2_feature.view(drug1_n_graph, -1))
                d1_mol_features.append(self.w[cnt_mod] * d1_feature)
                d2_mol_features.append(self.w[cnt_mod] * d2_feature)
                cnt_mod += 1
            else:
                d1_atom_features.append(molor_d1_feature)
                d2_atom_features.append(molor_d2_feature)

        if self.args.use_GNN:
            graph_embedding1, node_embedding1, graph_embedding2, node_embedding2 = self.gnn(adj_1, nd_1, ed_1, adj_2, nd_2, ed_2)
            if not self.args.use_cross_attention:
                d1_mol_features.append(self.w[cnt_mod] * graph_embedding1)
                d2_mol_features.append(self.w[cnt_mod] * graph_embedding2)
                cnt_mod += 1
            node_embedding1 = self.padding_GNN_result(node_embedding1)
            node_embedding2 = self.padding_GNN_result(node_embedding2)
            d1_atom_features.append(node_embedding1)
            d2_atom_features.append(node_embedding2)

        if self.args.use_cross_attention:
            if self.args.use_Mol:
                # mol当q
                # pad_adjs_1 = self.padding_col(d1_batch)
                # pad_adjs_2 = self.padding_col(d2_batch)
                # pad_mask_1 = None
                # pad_mask_2 = None

                # sch当q  第一个掩码只掩列，第二个行列都掩
                pad_adjs_1 = torch.ones([self.batch_size, self.args.max_mol_len, self.args.max_mol_len]).to(self.args.device)
                pad_adjs_2 = torch.ones([self.batch_size, self.args.max_mol_len, self.args.max_mol_len]).to(self.args.device)
                pad_mask_1 = self.padding_mask(d1_batch, d2_batch)
                pad_mask_2 = self.padding_mask(d2_batch, d1_batch)
            else:
                pad_adjs_1 = self.padding_adj(adj_1)
                pad_adjs_2 = self.padding_adj(adj_2)
                pad_mask_1 = self.padding_mask(d1_batch, d2_batch)
                pad_mask_2 = self.padding_mask(d2_batch, d1_batch)
            d1_atom_ft, att_w_in_1 = self.inner_cross(d1_atom_features[0], d1_atom_features[-1], pad_adjs_1)
            d2_atom_ft, att_w_in_2 = self.inner_cross(d2_atom_features[0], d2_atom_features[-1], pad_adjs_2)

            # fixme 存在标注错误，此处得出的为d2和d1 cross atom ft
            d1_cross_atom_ft, att_w_out_1 = self.outer_cross(d1_atom_ft, d2_atom_ft, pad_mask_1)
            d2_cross_atom_ft, att_w_out_2 = self.outer_cross(d2_atom_ft, d1_atom_ft, pad_mask_2)

            d1_cross_atom_ft += d1_atom_ft
            d2_cross_atom_ft += d2_atom_ft
            # d1_cross_atom_ft, att_w_out_1 = self.outer_cross(d2_atom_ft, d1_atom_ft, pad_mask_2)
            # d2_cross_atom_ft, att_w_out_2 = self.outer_cross(d1_atom_ft, d2_atom_ft, pad_mask_1)
            #
            # d1_cross_atom_ft += d1_atom_ft
            # d2_cross_atom_ft += d2_atom_ft

            d1_mol_features.append(self.w[cnt_mod] * torch.sum(d1_cross_atom_ft, dim=1))
            d2_mol_features.append(self.w[cnt_mod] * torch.sum(d2_cross_atom_ft, dim=1))

        d1_feature = torch.cat(d1_mol_features, dim=1)
        d2_feature = torch.cat(d2_mol_features, dim=1)  # [b, 539]
        final_fts_sum = d1_feature + d2_feature
        score = self.decoder(final_fts_sum)

        return score

