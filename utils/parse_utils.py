from torch import cuda
import argparse
import torch
from utils.logging_utils import LOSS_FUNCTIONS


def get_args():
    parser = argparse.ArgumentParser(description='Welcome to the training and inference system')

    # args for train
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--loss', type=str, default='CrossEntropy1', choices=[k for k, v in LOSS_FUNCTIONS.items()])  # 换dataset要改
    parser.add_argument('--score', type=str, default='bi_All', help='roc-auc or MSE or All')  # 换dataset要改
    parser.add_argument('--savemodel', action='store_true', default=True, help='Saves model with highest validation score')
    parser.add_argument('--logging', type=str, default='bi_less')  # 换dataset要改
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--model_path', type=str, default=None, help='Pretrain model path. One of option is savedmodels_zhang/MultiLevelDDI')
    parser.add_argument('--savemodels_dir', type=str, default='SavedModel_bi_all/', help='Save model path')

    # args for path
    parser.add_argument('--dataset_path', type=str, default='./dataset/')  # 换dataset要改
    parser.add_argument('--dataset_name', type=str, default='')
    parser.add_argument('--ESPF_path', type=str, default='./ESPF/')
    parser.add_argument('--train_3D_structure', type=str, default='drug_3D_structure_train.pkl')
    parser.add_argument('--val_3D_structure', type=str, default='drug_3D_structure_val.pkl')
    parser.add_argument('--test_3D_structure', type=str, default='drug_3D_structure_test.pkl')

    # args for main module
    parser.add_argument('--conv_hidden_dim', type=int, default=256)
    parser.add_argument('--input_dropout_rate', type=float, default=0.0)
    parser.add_argument('--mix_method', type=str, default='weigh')
    parser.add_argument('--num_class', type=int, default=1, help='dataset need 1, dataset1 need 86')  # 换dataset要改

    # args for schNet (3D)
    parser.add_argument('--use_SCH', type=bool, default=True, help='')
    parser.add_argument('--SCH_cutoff', type=float, default=10.0, help='Cutoff distance for interatomic interactions')
    parser.add_argument('--SCH_num_layers', type=int, default=6, help='num of layer of update edge and node embed')
    parser.add_argument('--SCH_hidden_channels', type=int, default=256, help='schNet hidden embedding size')
    parser.add_argument('--SCH_num_filters', type=int, default=256, help='the number of filters to use')
    parser.add_argument('--SCH_num_gaussians', type=int, default=50, help='The number of gaussians')
    parser.add_argument('--SCH_out_channels', type=int, default=128, help='The number of out dim')

    # args for Molormer (2D)
    parser.add_argument('--use_Mol', type=bool, default=True, help='')
    parser.add_argument('--Mol_num_layers', type=int, default=3)
    parser.add_argument('--Mol_num_heads', type=int, default=8)
    parser.add_argument('--Mol_hidden_dim', type=int, default=256)
    parser.add_argument('--Mol_inter_dim', type=int, default=256)
    parser.add_argument('--Mol_flatten_dim', type=int, default=2048)
    parser.add_argument('--Mol_encoder_dropout_rate', type=float, default=0.0)
    parser.add_argument('--Mol_attention_dropout_rate', type=float, default=0.0)
    parser.add_argument('--Mol_input_dropout_rate', type=float, default=0.0)
    parser.add_argument('--Mol_longest_path', type=int, default=20, help='the longest shortest path len')

    # args for AMDE (1D)
    parser.add_argument('--use_AMDE', type=bool, default=True, help='')
    parser.add_argument('--AMDE_max_d', type=int, default=50, help='the max num of small structure in mol')
    parser.add_argument('--AMDE_vocab_size', type=int, default=23532, help='num of subword units in subword_units_map_chembl.csv')
    parser.add_argument('--AMDE_n_layer', type=int, default=2)
    parser.add_argument('--AMDE_emb_size', type=int, default=384)
    parser.add_argument('--AMDE_dropout_rate', type=float, default=0.0)
    parser.add_argument('--AMDE_hidden_size', type=int, default=384)
    parser.add_argument('--AMDE_intermediate_size', type=int, default=1536)
    parser.add_argument('--AMDE_attention_head', type=int, default=8)
    parser.add_argument('--AMDE_attention_dropout', type=float, default=0.1)
    parser.add_argument('--AMDE_hidden_dropout', type=float, default=0.1)
    parser.add_argument('--AMDE_out_dim', type=int, default=128)

    # args for GNN
    parser.add_argument('--use_GNN', type=bool, default=False)
    parser.add_argument('--GNN_node_features', type=int, default=75)
    parser.add_argument('--GNN_edge_features', type=int, default=4)
    parser.add_argument('--GNN_out_features', type=int, default=1)
    parser.add_argument('--GNN_message_passes', type=int, default=2)
    parser.add_argument('--GNN_message_size', type=int, default=25)
    parser.add_argument('--GNN_message_depth', type=int, default=2)
    parser.add_argument('--GNN_message_hidden_dim', type=int, default=50)
    parser.add_argument('--GNN_attention_depth', type=int, default=2)
    parser.add_argument('--GNN_attention_hidden_dim', type=int, default=50)
    parser.add_argument('--GNN_gather_width', type=int, default=128, help='GNN output dim')
    parser.add_argument('--GNN_gather_attention_depth', type=int, default=2)
    parser.add_argument('--GNN_gather_attention_hidden_dim', type=int, default=45)
    parser.add_argument('--GNN_gather_embedding_depth', type=int, default=2)
    parser.add_argument('--GNN_gather_embedding_hidden_dim', type=int, default=26)
    parser.add_argument('--GNN_dropout_rate', type=float, default=0.0)
    parser.add_argument('--GNN_out_depth', type=int, default=2)
    parser.add_argument('--GNN_out_hidden_dim', type=int, default=90)
    parser.add_argument('--GNN_out_layer_shrinkage', type=float, default=0.6)

    # cross attention args
    parser.add_argument('--use_cross_attention', type=bool, default=True)
    parser.add_argument('--cross_inner_num_head', type=int, default=4)
    parser.add_argument('--cross_outer_num_head', type=int, default=4)
    parser.add_argument('--cross_out_dim', type=int, default=128, help='cross attention output dim')
    parser.add_argument('--max_mol_len', type=int, default=257, help='the largest number of atom in a mol')

    return parser
