import os
import pickle
from collections import defaultdict
from copy import deepcopy

import joblib
import numpy as np
import rdkit
import torch
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType
from torch.utils import data
from rdkit import Chem

from dataset_pac.collator import collator
from utils.gen_mol_graph import *
import pandas as pd
from subword_nmt.apply_bpe import BPE
import codecs
from utils import algos
from utils.graph_features import atom_features


class Dataset(data.Dataset):

    def __init__(self, df_dti, data_type_or_path, args):
        """Initialization"""
        self.args = args
        self.atomType = {'C': 1, 'H': 2, 'O': 3, 'N': 4, 'S': 5, 'Li': 6, 'Mg': 7, 'F': 8, 'K': 9, 'Al': 10, 'Cl': 11,
                         'Au': 12, 'Ca': 13, 'Hg': 14, 'Na': 15, 'P': 16, 'Ti': 17, 'Br': 18, 'Gd': 19}
        self.NOTINDICT = 19
        self.pass_smiles = set()

        # set 1D file and parameter
        vocab_path = self.args.ESPF_path + 'drug_codes_chembl.txt'  # 这是一个可解释的子结构分区分子指纹数据集 todo 这两个文件内容有何不同？
        bpe_codes_drug = codecs.open(vocab_path)
        self.dbpe = BPE(bpe_codes_drug, merges=-1, separator='')  # 设置用于分词的数据集
        self.sub_csv = pd.read_csv(self.args.ESPF_path + 'subword_units_map_chembl.csv')

        # 加载所有分子的3D信息
        if data_type_or_path is 'train':
            self.load_3D_structure(data_type_or_path + '.csv', self.args.train_3D_structure)
        elif data_type_or_path is 'val':
            self.load_3D_structure(data_type_or_path + '.csv', self.args.val_3D_structure)
        elif data_type_or_path is 'test':
            self.load_3D_structure(data_type_or_path + '.csv', self.args.test_3D_structure)
        else:
            self.load_3D_structure(data_type_or_path)

        self.BONDTYPE_TO_INT = defaultdict(
            lambda: 0,
            {
                BondType.SINGLE: 0,
                BondType.DOUBLE: 1,
                BondType.TRIPLE: 2,
                BondType.AROMATIC: 3
            }
        )

        self.df = self.do_filter(df_dti)  # 将不能3D化的分子删掉

        try:
            self.labels = self.df.label.values  # 边类型
        except AttributeError:
            self.labels = np.zeros(len(self.df.D1.values))
        self.list_IDs = self.df.index.values  # 边的编号

        self.drug1_id = self.df["D2"].values  # 边尾节点药物编号
        self.drug2_id = self.df["D1"].values

        self.smiles1 = self.df["S2"].values  # 边尾节点药物分子smiles
        self.smiles2 = self.df["S1"].values

    def load_3D_structure(self, data_path=None, drug_3D_structure_file=None):
        """
        Loads a dataset's dictionary files and splits the data according to the train_val_test_split variable stored
        in the args object.
        args:
            drug_3D_structure_file: 存储分子3D特征的文件
            data_path: 数据文件路径
        :return: Three sets, the training set, validation set and test sets (referred to as the meta-train,
        meta-val and meta-test in the paper)
        """
        if data_path in ['train.csv', 'val.csv', 'test.csv']:
            save_path = os.path.join(self.args.dataset_path, self.args.dataset_name, drug_3D_structure_file)
            data_path = os.path.join(self.args.dataset_path, self.args.dataset_name, data_path)

        else:
            directory = os.path.dirname(data_path)
            save_path = os.path.join(directory, 'drug_3D_structure.pkl')
        if os.path.exists(save_path):
            print("Load data at", save_path)
            self.h_to_t_dict, self.t_to_h_dict, self.id_set, \
            self.id_smiles_dict, self.smile_pos_dict, self.smile_z_dict, self.pass_smiles = joblib.load(save_path)
        else:
            print("Loading data into RAM")
            self.data_loaded_in_memory = True
            # if "drugbank" in self.dataset_name:
            self.h_to_t_dict, self.t_to_h_dict, self.id_set, \
            self.id_smiles_dict, self.smile_pos_dict, self.smile_z_dict = self.get_all_3D_structure(
                data_path)  # 读smile，并把smile转为3D
            joblib.dump((self.h_to_t_dict, self.t_to_h_dict, self.id_set,
                         self.id_smiles_dict, self.smile_pos_dict, self.smile_z_dict, self.pass_smiles),
                        save_path)  # 把得到的分子3D数据存储到硬盘里

        print("self.pass_smiles", len(self.pass_smiles))

    def get_all_3D_structure(self, file_path):
        df_all = pd.read_csv(file_path)
        h_to_t_dict = {}  # 记录药物节点的邻居list
        t_to_h_dict = {}
        id_set = set()  # 记录所有药物ID
        id_smiles_dict = {}  # 记录所有药物ID对应的smile序列
        for i in range(df_all.shape[0]):  # todo train中的药物包括val和test中的药物吗？
            head = df_all.loc[i, 'D1']
            tail = df_all.loc[i, 'D2']
            head_smile = df_all.loc[i, 'S1']  # 出度节点smile
            tail_smile = df_all.loc[i, 'S2']  # 入度节点smile
            id_smiles_dict[head] = head_smile
            id_smiles_dict[tail] = tail_smile
            id_set.add(head)
            id_set.add(tail)

            if h_to_t_dict.__contains__(head):
                h_to_t_dict[head].append(tail)
            else:
                h_to_t_dict[head] = []
                h_to_t_dict[head].append(tail)

            if t_to_h_dict.__contains__(tail):
                t_to_h_dict[tail].append(head)
            else:
                t_to_h_dict[tail] = []
                t_to_h_dict[tail].append(head)

        smile_pos_dict = {}
        smile_z_dict = {}
        for i, (id, smiles) in enumerate(id_smiles_dict.items()):
            print('Converting SMILES to 3Dgraph: {}/{}'.format(i + 1, len(id_set)))
            if smile_pos_dict.__contains__(smiles):
                continue
            else:
                ten_pos1, z1 = self.get_pos_z(smiles)  # 获得分子中原子的位置信息以及原子的数字表示，如果位置无法收敛则加入pass集忽略
                if ten_pos1 == None:
                    self.pass_smiles.add(smiles)
                    smile_pos_dict[smiles] = None
                    smile_z_dict[smiles] = None
                else:
                    smile_pos_dict[smiles] = ten_pos1
                    smile_z_dict[smiles] = z1

        return h_to_t_dict, t_to_h_dict, id_set, id_smiles_dict, smile_pos_dict, smile_z_dict

    def get_pos_z(self, smile1):
        # print(smile1)
        m1 = rdkit.Chem.MolFromSmiles(smile1)  # 从编码生成分子结构

        if m1 is None:
            self.pass_smiles.add(smile1)
            return None, None

        if m1.GetNumAtoms() == 1:
            self.pass_smiles.add(smile1)
            return None, None
        m1 = Chem.AddHs(m1)  # 根据分子价态自动添加氢原子

        ignore_flag1 = 0
        ignore1 = False

        while AllChem.EmbedMolecule(m1) == -1:  # 生成合理的三维构象，但可能不是全局最低能量构象 返回构象ID
            print('retry')
            ignore_flag1 = ignore_flag1 + 1
            if ignore_flag1 >= 10:
                ignore1 = True
                break
        if ignore1:
            self.pass_smiles.add(smile1)
            return None, None
        AllChem.MMFFOptimizeMolecule(m1)  # 优化三维构象
        m1 = Chem.RemoveHs(m1)  # 把刚才添加进来的氢原子去掉
        m1_con = m1.GetConformer(id=0)  # 获取构象

        pos1 = []  # 存储分子中所有原子的三维位置
        for j in range(m1.GetNumAtoms()):
            pos1.append(list(m1_con.GetAtomPosition(j)))
        np_pos1 = np.array(pos1)
        ten_pos1 = torch.Tensor(np_pos1)

        z1 = []  # 将原子转化为数字表示，如C->1
        for atom in m1.GetAtoms():
            if self.atomType.__contains__(atom.GetSymbol()):
                z = self.atomType[atom.GetSymbol()]
            else:
                z = self.NOTINDICT
            z1.append(z)

        z1 = np.array(z1)
        z1 = torch.tensor(z1)
        return ten_pos1, z1

    def do_filter(self, task_data):
        new_data = deepcopy(task_data)
        for smi in self.pass_smiles:
            new_data = new_data[new_data['S1'] != smi]
            new_data = new_data[new_data['S2'] != smi]
        new_data = new_data.reset_index(drop=True)
        return new_data

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.list_IDs)

    def __getitem__(self, index):
        """AMDE-dataset"""
        index = self.list_IDs[index]

        drug1_smile = self.smiles1[index]
        drug2_smile = self.smiles2[index]
        drug1_id = self.df.iloc[index]['D1']
        drug2_id = self.df.iloc[index]['D2']

        if self.args.use_AMDE:
            d1, mask_1 = self.preprocess_for_1D(drug1_smile, drug1_id)  # 1D 通过分词方法获得分割后的smile，并对每块smile进行编码；掩码同NLP
            d2, mask_2 = self.preprocess_for_1D(drug2_smile, drug2_id)
        else:
            d1, mask_1, d2, mask_2 = np.zeros(50), np.zeros(50), np.zeros(50), np.zeros(50)

        'Generates one sample of data'
        # Select sample
        # Load data and get label

        if self.args.use_Mol:
            d_node, d_in_degree, d_out_degree = self.preprocess_for_2D(drug1_id)  # 2D 返回另一种节点编码、原子偏置？、两两原子间最短距离、入度、出度、最短路径上所有边特征
            p_node, p_in_degree, p_out_degree = self.preprocess_for_2D(drug2_id)
        else:
            d_node, d_in_degree, d_out_degree = torch.zeros((1, 1)), torch.zeros(1), torch.zeros(1)
            p_node, p_in_degree, p_out_degree = torch.zeros((1, 1)), torch.zeros(1), torch.zeros(1)
        label = self.labels[index]  # 分子边类型
        # print(index, drug1_id, self.smiles1[index], drug2_id, self.smiles2[index], label)
        # assert False
        if self.args.use_SCH:
            d1_position, d1_z = self.preprocess_for_3D(drug1_smile)
            d2_position, d2_z = self.preprocess_for_3D(drug2_smile)
        else:
            d1_position, d1_z = torch.zeros((1, 3)), torch.zeros(1)
            d2_position, d2_z = torch.zeros((1, 3)), torch.zeros(1)

        if self.args.use_GNN:
            adj_1, nd_1, ed_1 = self.smile_to_graph(self.smiles1[index])
            adj_2, nd_2, ed_2 = self.smile_to_graph(self.smiles2[index])
        else:
            adj_1, nd_1, ed_1 = torch.zeros((1, 1)), torch.zeros((1, 75)), torch.zeros((1, 1, 4))
            adj_2, nd_2, ed_2 = torch.zeros((1, 1)), torch.zeros((1, 75)), torch.zeros((1, 1, 4))

        return d_node, d_in_degree, d_out_degree, p_node, p_in_degree, p_out_degree, \
               label, d1, d2, mask_1, mask_2, d1_position, d1_z, d2_position, d2_z, \
               (adj_1, nd_1, ed_1), (adj_2, nd_2, ed_2)

    def preprocess_for_3D(self, drug_smile):
        if self.smile_pos_dict.__contains__(drug_smile):
            drug_position = self.smile_pos_dict[drug_smile]
            z1 = self.smile_z_dict[drug_smile]
        else:
            drug_position, z1 = self.get_pos_z(drug_smile)

        if drug_position is None or z1 is None:
            print(" drug_smiles pass", drug_smile)

        return drug_position, z1

    def preprocess_for_2D(self, drug_id):

        x, edge_attr, edge_index = sdf2graph(drug_id,
                                             self.args)  # x: 分子中的另一种原子特征向量  edge_attr: 另一种边特征  edge_index: 分子边的起始、终止点
        N = x.size(0)  # 原子数量
        x = mol_to_single_emb(x)  # 使原子各个特征处在不同的整数域中

        # node adj matrix [N, N] bool
        adj = torch.zeros([N, N], dtype=torch.bool)
        adj[edge_index[0, :], edge_index[1, :]] = True  # 又造了一个邻接矩阵

        # edge feature here
        if len(edge_attr.size()) == 1:
            edge_attr = edge_attr[:, None]
        attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
        attn_edge_type[edge_index[0, :], edge_index[1, :]] = mol_to_single_emb(edge_attr) + 1  # 使边各个特征处在不同的整数域

        node = x
        in_degree = adj.long().sum(dim=1).view(-1)
        out_degree = adj.long().sum(dim=0).view(-1)
        return node, in_degree, out_degree  # 返回另一种节点编码、原子偏置？、两两原子间最短距离、入度、出度、最短路径上所有边特征

    def preprocess_for_1D(self, drug_smile, drug_id):
        ## Sequence encoder parameter
        segmented_path = os.path.join(self.args.dataset_path, 'word_segmentation', drug_id + '.pkl')
        if os.path.exists(segmented_path):
                i, input_mask = joblib.load(segmented_path)
        else:
            idx2word_d = self.sub_csv['index'].values
            words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))
            max_d = self.args.AMDE_max_d
            t1 = self.dbpe.process_line(drug_smile).split()  # 通过数据集对分子结构进行线性分词
            try:
                i1 = np.asarray([words2idx_d[i] for i in t1])  # 分词后进行编码
            except:
                i1 = np.array([0])
                print('error:', drug_smile)

            l = len(i1)

            if l < max_d:
                i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
                input_mask = ([1] * l) + ([0] * (max_d - l))

            else:
                i = i1[:max_d]
                input_mask = [1] * max_d
            joblib.dump([i, input_mask], segmented_path)

        return i, np.asarray(input_mask)

    def smile_to_graph(self, smile):
        molecule = Chem.MolFromSmiles(smile)
        n_atoms = molecule.GetNumAtoms()
        atoms = [molecule.GetAtomWithIdx(i) for i in range(n_atoms)]  # 获取分子中的全部原子
        adjacency = Chem.rdmolops.GetAdjacencyMatrix(molecule)  # 生成表示分子结构的邻接矩阵
        node_features = np.array([atom_features(atom) for atom in atoms])  # 获取每个节点(原子)特征，共75维，包括原子的类型等信息，one-hot表示
        n_edge_features = 4
        edge_features = np.zeros([n_atoms, n_atoms, n_edge_features])  # 四种键，用one-hot表示成为键特征，表示第i到第j节点的边特征
        for bond in molecule.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = self.BONDTYPE_TO_INT[bond.GetBondType()]
            edge_features[i, j, bond_type] = 1
            edge_features[j, i, bond_type] = 1
        # print(adjacency.shape,node_features.shape,edge_features.shape)
        # (atom_num,atom_num), (atom_num,75), (atom_num,atom_num,4)

        return adjacency, node_features, edge_features


if __name__ == '__main__':
    import utils.parse_utils

    args = utils.parse_utils.get_args().parse_args()
    args.dataset_path = '.' + args.dataset_path
    args.ESPF_path = '.' + args.ESPF_path
    train_data = pd.read_csv(os.path.join(args.dataset_path, args.dataset_name, 'train.csv'))
    training_set = Dataset(train_data, 'train', args)
    val_data = pd.read_csv(os.path.join(args.dataset_path, args.dataset_name, 'val.csv'))
    val_set = Dataset(val_data, 'val', args)
    test_data = pd.read_csv(os.path.join(args.dataset_path, args.dataset_name, 'test.csv'))
    test_set = Dataset(train_data, 'test', args)
    params = {'batch_size': 16,
              'shuffle': True,
              'num_workers': 0,
              'drop_last': True,
              'collate_fn': collator}
    dataloader = data.DataLoader(training_set, **params)
    for i in dataloader:
        print(i)
        break
