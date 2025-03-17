import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils import data

from dataset_pac.collator import collator
from models.model import MultiLevelDDI

from dataset_pac.dataset import Dataset
import os
from utils.logging_utils import LOG, LOSS_FUNCTIONS
import utils.logging_utils as lu
import time
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm
from torch.autograd import Variable
from utils import parse_utils
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

args = parse_utils.get_args().parse_args()
assert lu.dataset_path == args.dataset_path, "utils.logging_utils dataset_path and utils.parse_utils dataset_path must be same"

torch.manual_seed(2)
np.random.seed(3)
# todo 五折没做，如果需要可以直接新建一个五折的数据集然后改dataset_name


def test(data_set, model):
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0

    for _, (d_node, d_attn_bias, d_spatial_pos, d_in_degree, d_out_degree, d_edge_input,
            p_node, p_attn_bias, p_spatial_pos, p_in_degree, p_out_degree, p_edge_input,
            label, (adj_1, nd_1, ed_1), (adj_2, nd_2, ed_2), d1, d2, mask_1, mask_2) in enumerate(tqdm(data_set)):
        score = model(d_node.cuda(), d_attn_bias.cuda(), d_spatial_pos.cuda(),
                      d_in_degree.cuda(), d_out_degree.cuda(), d_edge_input.cuda(), p_node.cuda(),
                      p_attn_bias.cuda(), p_spatial_pos.cuda(), p_in_degree.cuda(),
                      p_out_degree.cuda(), p_edge_input.cuda(),
                      adj_1.cuda(), nd_1.cuda(), ed_1.cuda(),
                      adj_2.cuda(), nd_2.cuda(), ed_2.cuda(),
                      d1.cuda(), d2.cuda(), mask_1.cuda(), mask_2.cuda()
                      )

        label = Variable(torch.from_numpy(np.array(label - 1)).long()).cuda()
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(score, label)
        loss_accumulate += loss
        count += 1

        outputs = score.argmax(dim=1).detach().cpu().numpy() + 1
        label_ids = label.to('cpu').numpy() + 1

        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + outputs.flatten().tolist()

    loss = loss_accumulate / count

    accuracy = accuracy_score(y_label, y_pred)
    micro_precision = precision_score(y_label, y_pred, average='micro')
    micro_recall = recall_score(y_label, y_pred, average='micro')
    micro_f1 = f1_score(y_label, y_pred, average='micro')

    macro_precision = precision_score(y_label, y_pred, average='macro')
    macro_recall = recall_score(y_label, y_pred, average='macro')
    macro_f1 = f1_score(y_label, y_pred, average='macro')
    return accuracy, micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1, loss.item()


def main():

    loss_history = []
    if args.model_path:
        model = torch.load(args.model_path)
    else:
        model = MultiLevelDDI(args)

    model = model.to(args.device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, dim=0)

    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': args.num_workers,
              'drop_last': True,
              'collate_fn': collator}

    train_data = pd.read_csv(os.path.join(args.dataset_path, args.dataset_name, 'train.csv'))
    val_data = pd.read_csv(os.path.join(args.dataset_path, args.dataset_name, 'val.csv'))
    test_data = pd.read_csv(os.path.join(args.dataset_path, args.dataset_name, 'test.csv'))

    training_set = Dataset(train_data, 'train', args)
    validation_set = Dataset(val_data, 'val', args)
    testing_set = Dataset(test_data, 'test', args)

    training_generator = data.DataLoader(training_set, **params)
    validation_generator = data.DataLoader(validation_set, **params)
    testing_generator = data.DataLoader(testing_set, **params)  # 直接用params中的collate_fn替换dataloader的collate_fn 类似于重写

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = LOSS_FUNCTIONS[args.loss]
    # scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=config['epochs'], eta_min=args.min_lr)

    print('--- Go for Training ---')
    torch.backends.cudnn.benchmark = True
    for epo in range(args.epochs):
        model.train()
        start_time = time.time()
        for i, (d_node, d_in_degree, d_out_degree, p_node, p_in_degree, p_out_degree,
                label, d1, d2, mask_1, mask_2, d1_positions, d1_z, d1_batch, d2_positions, d2_z, d2_batch,
                (adj_1, nd_1, ed_1), (adj_2, nd_2, ed_2)) in enumerate(training_generator):
            opt.zero_grad()
            score = model(d_node.to(args.device), d_in_degree.to(args.device), d_out_degree.to(args.device),
                          p_node.to(args.device), p_in_degree.to(args.device), p_out_degree.to(args.device),
                          d1.to(args.device), d2.to(args.device), mask_1.to(args.device), mask_2.to(args.device),
                          d1_positions.to(args.device), d1_z.to(args.device), d1_batch.to(args.device),
                          d2_positions.to(args.device), d2_z.to(args.device), d2_batch.to(args.device),
                          adj_1.to(args.device), nd_1.to(args.device), ed_1.to(args.device),
                          adj_2.to(args.device), nd_2.to(args.device), ed_2.to(args.device))  # torch tensor
            label = label.long().to(args.device)  # torch tensor
            # loss_fct = torch.nn.CrossEntropyLoss().cuda()
            loss = criterion(score, label)
            if torch.isnan(loss).any():
                for param_group in opt.param_groups:
                    param_group['lr'] = param_group['lr'] / 10
                model = torch.load(args.savemodels_dir + type(model).__name__)
                print('In Epoch ' + str(epo + 1) + ' iteration ' + str(i) + ' lr decrease')
                break
            # print(label.shape,label)
            # print(loss)
            # assert False
            # loss = loss_fct(score, label)
            loss_history.append(loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 5.0)
            # 用于裁剪梯度，梯度裁剪是一种正则化技术，用于防止在训练深度学习模型时发生梯度爆炸
            opt.step()
            # scheduler.step()
            end_time = time.time()
            if (i % 100 == 0):
                print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + ' with loss ' + str(
                    loss.cpu().detach().numpy()) + ' use time ' + str(end_time - start_time) + 's')
            start_time = end_time

        with torch.set_grad_enabled(False):
            model.eval()
            LOG[args.logging](model, training_generator, validation_generator, testing_generator, criterion, epo, args)
            # accuracy, micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1, loss = test(validation_generator, model)
            # print("[Validation metrics]: loss:{:.4f} accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(
            #     loss, accuracy, macro_precision, macro_recall, macro_f1))
            # if accuracy > max_auc:
            #    # torch.save(model, 'save_model/' + str(accuracy) + '_model.pth')
            #     torch.save(model, 'save_model/best_model.pth')
            #     model_max = copy.deepcopy(model)
            #     max_auc = accuracy
            #     print("*" * 30 + " save best model " + "*" * 30)

        # torch.cuda.empty_cache()

    # print('\n--- Go for Testing ---')
    # try:
    #     with torch.set_grad_enabled(False):
    #         accuracy, micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1, loss  = test(testing_generator, model_max)
    #         print("[Testing metrics]: loss:{:.4f} accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(
    #             loss, accuracy, macro_precision, macro_recall, macro_f1))
    # except:
    #     print('testing failed')
    return model, loss_history


'''
nohup python -u train_binary.py > new3_adataset_lr-5_1.log 2>&1 &
new1: AMDE + Molormer
new2: AMDE + Molormer + decoder-->BilinearDecoder  ×
new3: AMDE + Molormer + concat-->乘以权重
'''

if __name__ == '__main__':
    main()
