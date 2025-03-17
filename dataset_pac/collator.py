# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# fixme have changed
import torch


def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen):
    x = x + 1
    xlen1, xlen2, xlen3 = x.size()
    if xlen1 < padlen:
        new_x = x.new_zeros([padlen, padlen, xlen3], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_4d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def collator(batch, max_d_node=256, multi_hop_max_dist=20, spatial_pos_max=20):
    drug1_node, drug1_in_degree, drug1_out_degree = [], [], []
    drug2_node, drug2_in_degree, drug2_out_degree = [], [], []
    labels = []

    # 2D collator
    for d1_node, d1_in_degree, d1_out_degree, \
        d2_node, d2_in_degree, d2_out_degree, \
        label, d1, d2, mask_1, mask_2, d1_position, d1_z, d2_position, d2_z, \
        (adj_1, node_fts_1, edge_fts_1),(adj_2, node_fts_2, edge_fts_2) in batch:

        if d1_node.size(0) <= max_d_node and d2_node.size(0) <= max_d_node:
            drug1_node.append(d1_node)
            drug1_in_degree.append(d1_in_degree)
            drug1_out_degree.append(d1_out_degree)

            drug2_node.append(d2_node)
            drug2_in_degree.append(d2_in_degree)
            drug2_out_degree.append(d2_out_degree)
            labels.append(label)

    # node
    drug1_node = torch.cat([pad_2d_unsqueeze(i, max_d_node) for i in drug1_node])  # 把分子中的原子数扩展到最大256
    drug2_node = torch.cat([pad_2d_unsqueeze(i, max_d_node) for i in drug2_node])

    # in_degree
    drug1_in_degree = torch.cat([pad_1d_unsqueeze(i, max_d_node) for i in drug1_in_degree])  # 扩展入度矩阵
    drug2_in_degree = torch.cat([pad_1d_unsqueeze(i, max_d_node) for i in drug2_in_degree])

    # out_degree
    drug1_out_degree = torch.cat([pad_1d_unsqueeze(i, max_d_node) for i in drug1_out_degree])  # 扩展出度矩阵
    drug2_out_degree = torch.cat([pad_1d_unsqueeze(i, max_d_node) for i in drug2_out_degree])

    # 1D collator
    n_samples = drug1_node.size()[0]  # batchsize
    n_targets = 1
    n_emb = d1.shape[0]
    n_mask = mask_1.shape[0]

    target_tensor = torch.zeros(n_samples, n_targets)
    d1_emb_tensor = torch.zeros(n_samples, n_emb)
    d2_emb_tensor = torch.zeros(n_samples, n_emb)
    mask_1_tensor = torch.zeros(n_samples, n_mask)
    mask_2_tensor = torch.zeros(n_samples, n_mask)

    for i in range(n_samples):
        target, d1, d2, mask_1, mask_2 = batch[i][6:11]
        target_tensor[i] = torch.tensor(target)
        d1_emb_tensor[i] = torch.IntTensor(d1)
        d2_emb_tensor[i] = torch.IntTensor(d2)
        mask_1_tensor[i] = torch.tensor(mask_1)
        mask_2_tensor[i] = torch.tensor(mask_2)

    # 3D collator
    d1_batch_positions, d1_batch_z, d1_batch, d2_batch_positions, d2_batch_z, d2_batch = [], [], [], [], [], []
    for i in range(n_samples):
        d1_position, d1_z, d2_position, d2_z = batch[i][-6:-2]

        d1_batch_positions.append(d1_position)
        d1_batch_z.append(d1_z)
        d1_batch.append([i for _ in range(len(d1_z))])

        d2_batch_positions.append(d2_position)
        d2_batch_z.append(d2_z)
        d2_batch.append([i for _ in range(len(d2_z))])

    d1_batch_positions = torch.tensor(torch.cat(d1_batch_positions, dim=0))
    d1_batch_z = torch.tensor(torch.cat(d1_batch_z, dim=0))
    d1_batch = torch.tensor([_ for line in d1_batch for _ in line])

    d2_batch_positions = torch.tensor(torch.cat(d2_batch_positions, dim=0))
    d2_batch_z = torch.tensor(torch.cat(d2_batch_z, dim=0))
    d2_batch = torch.tensor([_ for line in d2_batch for _ in line])

    # GNN collator
    n_nodes_largest_graph_1 = max(map(lambda sample: sample[-2][0].shape[0], batch))
    n_nodes_largest_graph_2 = max(map(lambda sample: sample[-1][0].shape[0], batch))

    n_node_fts_1 = node_fts_1.shape[1]
    n_edge_fts_1 = edge_fts_1.shape[2]
    n_node_fts_2 = node_fts_2.shape[1]
    n_edge_fts_2 = edge_fts_2.shape[2]

    adjacency_tensor_1 = torch.zeros(n_samples, n_nodes_largest_graph_1, n_nodes_largest_graph_1)
    node_tensor_1 = torch.zeros(n_samples, n_nodes_largest_graph_1, n_node_fts_1)
    edge_tensor_1 = torch.zeros(n_samples, n_nodes_largest_graph_1, n_nodes_largest_graph_1, n_edge_fts_1)

    adjacency_tensor_2 = torch.zeros(n_samples, n_nodes_largest_graph_2, n_nodes_largest_graph_2)
    node_tensor_2 = torch.zeros(n_samples, n_nodes_largest_graph_2, n_node_fts_2)
    edge_tensor_2 = torch.zeros(n_samples, n_nodes_largest_graph_2, n_nodes_largest_graph_2, n_edge_fts_2)

    for i in range(n_samples):

        (adj_1, node_fts_1, edge_fts_1), (adj_2, node_fts_2, edge_fts_2) = batch[i][-2:]

        n_nodes_1 = adj_1.shape[0]
        n_nodes_2 = adj_2.shape[0]
        adjacency_tensor_1[i, :n_nodes_1, :n_nodes_1] = torch.Tensor(adj_1)
        node_tensor_1[i, :n_nodes_1, :] = torch.Tensor(node_fts_1)
        edge_tensor_1[i, :n_nodes_1, :n_nodes_1, :] = torch.Tensor(edge_fts_1)

        adjacency_tensor_2[i, :n_nodes_2, :n_nodes_2] = torch.Tensor(adj_2)
        node_tensor_2[i, :n_nodes_2, :] = torch.Tensor(node_fts_2)
        edge_tensor_2[i, :n_nodes_2, :n_nodes_2, :] = torch.Tensor(edge_fts_2)
    return drug1_node, drug1_in_degree, drug1_out_degree, drug2_node, drug2_in_degree, drug2_out_degree, \
           target_tensor, d1_emb_tensor, d2_emb_tensor, mask_1_tensor, mask_2_tensor, \
           d1_batch_positions, d1_batch_z, d1_batch, d2_batch_positions, d2_batch_z, d2_batch, \
           (adjacency_tensor_1, node_tensor_1, edge_tensor_1), (adjacency_tensor_2, node_tensor_2, edge_tensor_2)
