import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch import nn
import pandas as pd
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset_path = './dataset/'


# os.environ['CUDA_VISIBLE_DEVICES'] = '2,1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,1'


# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:2" if use_cuda else "cpu")
class CrossEntropy(nn.Module):  # loss= criterion(score, label)
    def __init__(self, label_type):
        super(CrossEntropy, self).__init__()
        df = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
        y_train = df['label'].to_numpy() - 1
        # class_weights = torch.tensor(compute_class_weight('balanced', classes=np.unique(y_train), y=y_train), dtype=torch.float32).to(device)
        # self.ls = nn.CrossEntropyLoss(weight=class_weights)
        self.ls = nn.CrossEntropyLoss()
        self.label_type = label_type

    def forward(self, input, target):
        if self.label_type == 2:
            target = target - 1
            target = target.view(-1)
            loss = self.ls(input, target)
            return loss
        else:
            scores = torch.sigmoid(input)
            target_active = (target == 1).float()
            loss_terms = -(target_active * torch.log(scores) + (1 - target_active) * torch.log(1 - scores))
            b = loss_terms.sum() / len(loss_terms)
            return b


LOSS_FUNCTIONS = {
    'CrossEntropy1': CrossEntropy(1),
    'CrossEntropy2': CrossEntropy(2)
}


class Globals:  # container for all objects getting passed between log calls
    evaluate_called = False


g = Globals()


def check_tensor(tensor):
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isnan(tensor).any()
    has_large_value = (tensor > 3.4e38).any()
    return has_nan, has_inf, has_large_value


def bi_all_evaluate(output, target):
    # print(target,output)
    lengh = len(output)
    scores = torch.sigmoid(output)
    # print(type(target),type(scores))
    # print(target.device,scores.device)
    target = target.cpu()
    scores = scores.cpu()
    has_nan, has_inf, has_large_values = check_tensor(scores)
    if has_nan or has_inf or has_large_values:
        print('\tThere are some nan, inf or large value number in scores.\n\tscores: ', scores)
        scores = torch.zeros(target.shape)
    auroc = roc_auc_score(target, scores)
    scores = np.array(scores).astype(float)
    sum_scores = np.sum(scores)
    ave_scores = sum_scores / lengh
    target = np.array(target).astype(int)

    Confusion_M = np.zeros((2, 2), dtype=float)  # (TN FP),(FN,TP)
    for i in range(lengh):
        if (scores[i] < ave_scores):
            scores[i] = 0
        else:
            scores[i] = 1
    scores = np.array(scores).astype(int)

    for i in range(lengh):
        if (target[i] == scores[i]):
            if (target[i] == 1):
                Confusion_M[0][0] += 1  # TP
            else:
                Confusion_M[1][1] += 1  # TN
        else:
            if (target[i] == 1):
                Confusion_M[0][1] += 1  # FP
            else:
                Confusion_M[1][0] += 1  # FN

    Confusion_M = np.array(Confusion_M, dtype=float)
    print('Confusion_M:', Confusion_M)
    accuracy = (Confusion_M[1][1] + Confusion_M[0][0]) / (
            Confusion_M[0][0] + Confusion_M[1][1] + Confusion_M[0][1] + Confusion_M[1][0])

    recall = Confusion_M[0][0] / (Confusion_M[0][0] + Confusion_M[0][1])
    precision = Confusion_M[0][0] / (Confusion_M[0][0] + Confusion_M[1][0])
    F1 = 2 * precision * recall
    h = precision + recall
    F1 = F1 / h
    sum = 0.0
    for i in range(lengh):
        sum = sum + (target[i] - scores[i]) * (target[i] - scores[i])

    return F1, accuracy, recall, precision, auroc


def muti_all_evaluate(output, target):
    scores = torch.sigmoid(output)
    outputs = scores.argmax(dim=1).detach().cpu().numpy()
    target = target.cpu() - 1
    has_nan, has_inf, has_large_values = check_tensor(scores)
    if has_nan or has_inf or has_large_values:
        print('\tThere are some nan, inf or large value number in scores.\n\tscores: ', scores)
        scores = torch.zeros(target.shape)

    accuracy = accuracy_score(target, outputs)
    micro_precision = precision_score(target, outputs, average='micro')
    micro_recall = recall_score(target, outputs, average='micro')
    micro_f1 = f1_score(target, outputs, average='micro')

    macro_precision = precision_score(target, outputs, average='macro')
    macro_recall = recall_score(target, outputs, average='macro')
    macro_f1 = f1_score(target, outputs, average='macro')

    return accuracy, micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1


SCORE_FUNCTIONS = {
    'bi_All': bi_all_evaluate,
    'muti_All': muti_all_evaluate}


def feed_net(net, dataloader, criterion):
    batch_outputs = []
    batch_losses = []
    batch_targets = []
    for i_batch, batch in enumerate(dataloader):
        d_node, d_in_degree, d_out_degree, p_node, p_in_degree, p_out_degree, label, d1, d2, mask_1, mask_2, d1_positions, d1_z, d1_batch, d2_positions, d2_z, d2_batch, (
        adj_1, nd_1, ed_1), (adj_2, nd_2, ed_2) = batch
        output = net(d_node.to(device), d_in_degree.to(device), d_out_degree.to(device),
                     p_node.to(device), p_in_degree.to(device), p_out_degree.to(device),
                     d1.to(device), d2.to(device), mask_1.to(device), mask_2.to(device),
                     d1_positions.to(device), d1_z.to(device), d1_batch.to(device), d2_positions.to(device),
                     d2_z.to(device), d2_batch.to(device),
                     adj_1.to(device), nd_1.to(device), ed_1.to(device),
                     adj_2.to(device), nd_2.to(device), ed_2.to(device)
                     )
        label = torch.from_numpy(np.array(label)).long().to(device)
        loss = criterion(output, label)
        batch_outputs.append(output)
        batch_losses.append(loss.item())
        batch_targets.append(label)

    outputs = torch.cat(batch_outputs)

    loss = np.mean(batch_losses)  # average loss
    targets = torch.cat(batch_targets)
    return outputs, loss, targets


def bi_evaluate_net(net, train_dataloader, validation_dataloader, test_dataloader, criterion, args, now_epoch):
    global g
    if not g.evaluate_called:
        g.evaluate_called = True
        g.best_mean_train_score, g.best_mean_validation_score, g.best_mean_test_score, g.best_mean_epoch = 0, 0, 0, 0
        # g.train_subset_loader = train_dataloader

    # train_output, train_loss, train_target = feed_net(net,g.train_subset_loader, criterion)
    validation_output, validation_loss, validation_target = feed_net(net, validation_dataloader, criterion)
    test_output, test_loss, test_target = feed_net(net, test_dataloader, criterion)

    # train_scores = SCORE_FUNCTIONS[args.score](train_output, train_target)
    validation_scores = SCORE_FUNCTIONS[args.score](validation_output, validation_target)
    test_scores = SCORE_FUNCTIONS[args.score](test_output, test_target)
    new_best_model_found = validation_scores[4] > g.best_mean_validation_score

    if new_best_model_found:
        # g.best_mean_train_score = train_scores[4]
        g.best_mean_validation_score = validation_scores[4]
        g.best_mean_test_score = test_scores[4]
        g.best_mean_epoch = now_epoch

        if args.savemodel:
            if not os.path.isdir(args.savemodels_dir):
                os.makedirs(args.savemodels_dir)
            path = args.savemodels_dir + type(net).__name__
            print(path)
            torch.save(net, path)

    if (args.score == 'bi_All'):
        return {
            #  'loss':{'train': train_loss},
            #  'F1 score':{'train': train_scores[0], 'validation': validation_scores[0], 'test': test_scores[0]},
            #  'Accuracy':{'train': train_scores[1], 'validation': validation_scores[1], 'test': test_scores[1]},
            #  'Recall':{'train': train_scores[2], 'validation': validation_scores[2], 'test': test_scores[2]},
            #  'Precision':{'train': train_scores[3], 'validation': validation_scores[3], 'test': test_scores[3]},
            #  'auroc':{'train': train_scores[4], 'validation': validation_scores[4], 'test': test_scores[4]},
            # 'best mean':{'train': g.best_mean_train_score, 'validation': g.best_mean_validation_score, 'test': g.best_mean_test_score}
            'F1 score': {'validation': validation_scores[0], 'test': test_scores[0]},
            'Accuracy': {'validation': validation_scores[1], 'test': test_scores[1]},
            'Recall': {'validation': validation_scores[2], 'test': test_scores[2]},
            'Precision': {'validation': validation_scores[3], 'test': test_scores[3]},
            'auroc': {'validation': validation_scores[4], 'test': test_scores[4]},
            'best mean': {'validation': g.best_mean_validation_score, 'test': g.best_mean_test_score,
                          'epoch': g.best_mean_epoch}
        }


def muti_evaluate_net(net, train_dataloader, validation_dataloader, test_dataloader, criterion, args, now_epoch):
    global g
    if not g.evaluate_called:
        g.evaluate_called = True
        g.best_mean_train_score, g.best_mean_validation_score, g.best_mean_test_score, g.best_mean_epoch = 0, 0, 0, 0
        # g.train_subset_loader = train_dataloader

    # train_output, train_loss, train_target = feed_net(net,g.train_subset_loader, criterion)
    validation_output, validation_loss, validation_target = feed_net(net, validation_dataloader, criterion)
    test_output, test_loss, test_target = feed_net(net, test_dataloader, criterion)

    # train_scores = SCORE_FUNCTIONS[args.score](train_output, train_target)
    validation_scores = SCORE_FUNCTIONS[args.score](validation_output, validation_target)
    test_scores = SCORE_FUNCTIONS[args.score](test_output, test_target)
    new_best_model_found = validation_scores[6] > g.best_mean_validation_score

    if new_best_model_found:
        # g.best_mean_train_score = train_scores[4]
        g.best_mean_validation_score = validation_scores[6]
        g.best_mean_test_score = test_scores[6]
        g.best_mean_epoch = now_epoch
        if args.savemodel:
            if not os.path.isdir(args.savemodels_dir):
                os.makedirs(args.savemodels_dir)
            path = args.savemodels_dir + type(net).__name__
            print(path)
            torch.save(net, path)

    if (args.score == 'muti_All'):
        return {
            #  'loss':{'train': train_loss},
            #  'F1 score':{'train': train_scores[0], 'validation': validation_scores[0], 'test': test_scores[0]},
            #  'Accuracy':{'train': train_scores[1], 'validation': validation_scores[1], 'test': test_scores[1]},
            #  'Recall':{'train': train_scores[2], 'validation': validation_scores[2], 'test': test_scores[2]},
            #  'Precision':{'train': train_scores[3], 'validation': validation_scores[3], 'test': test_scores[3]},
            #  'auroc':{'train': train_scores[4], 'validation': validation_scores[4], 'test': test_scores[4]},
            # 'best mean':{'train': g.best_mean_train_score, 'validation': g.best_mean_validation_score, 'test': g.best_mean_test_score}
            'Accuracy': {'validation': validation_scores[0], 'test': test_scores[0]},
            'Micro precision': {'validation': validation_scores[1], 'test': test_scores[1]},
            'Micro recall': {'validation': validation_scores[2], 'test': test_scores[2]},
            'Micro f1': {'validation': validation_scores[3], 'test': test_scores[3]},
            'Macro precision': {'validation': validation_scores[4], 'test': test_scores[4]},
            'Macro recall': {'validation': validation_scores[5], 'test': test_scores[5]},
            'Macro f1': {'validation': validation_scores[6], 'test': test_scores[6]},
            'Best macro f1': {'validation': g.best_mean_validation_score, 'test': g.best_mean_test_score,
                                  'epoch': g.best_mean_epoch}
        }


def get_run_info(net, args):
    return {
        'net': type(net).__name__,
        'args': ', '.join([str(k) + ': ' + str(v) for k, v in vars(args).items()]),
        'modules': {name: str(module) for name, module in net._modules.items()}
    }


def bi_less_log(net, train_dataloader, validation_dataloader, test_dataloader, criterion, epoch, args):
    scalars = bi_evaluate_net(net, train_dataloader, validation_dataloader, test_dataloader, criterion, args, epoch + 1)
    global g
    if not g.evaluate_called:
        run_info = get_run_info(net, args)
        print('net: ' + run_info['net'])
        print('args: {' + run_info['args'] + '}')
        print('****** MODULES: ******')
        for name, description in run_info['modules'].items():
            print(name + ': ' + description)
        print('**********************')

    if (args.score == 'bi_All'):
        # print('epoch {}, F1 score :training mean: {}, validation mean: {}, testing mean: {}'.format(
        s = 'epoch {}, F1 score: validation mean: {}, testing mean: {}\n'.format(
            epoch + 1,
            #  scalars['F1 score']['train'],
            scalars['F1 score']['validation'],
            scalars['F1 score']['test'])
        s += '          ACC: validation mean: {}, testing mean: {}\n'.format(
            # scalars['Accuracy']['train'],
            scalars['Accuracy']['validation'],
            scalars['Accuracy']['test'])
        s += '          Precision: validation mean: {}, testing mean: {}\n'.format(
            # scalars['Precision']['train'],
            scalars['Precision']['validation'],
            scalars['Precision']['test'])
        s += '          Recall: validation mean: {}, testing mean: {}\n'.format(
            # scalars['Recall']['train'],
            scalars['Recall']['validation'],
            scalars['Recall']['test'])
        s += '          AUROC: validation mean: {}, testing mean: {}\n'.format(
            # scalars['auroc']['train'],
            scalars['auroc']['validation'],
            scalars['auroc']['test'])
        s += '          best auroc: validation mean: {}, testing mean: {}, best epoch: {}\n\n'.format(
            #  scalars['best mean']['train'],
            scalars['best mean']['validation'],
            scalars['best mean']['test'],
            scalars['best mean']['epoch'])
        print(s)
        if epoch == 0:
            with open('log.txt', 'w') as f:
                f.write(s)
        else:
            with open('log.txt', 'a') as f:
                f.write(s)
    else:
        mean_score_key = 'mean {}'.format(args.score)
        print('epoch {}, training mean {}: {}, validation mean {}: {}, testing mean {}:{}'.format(
            epoch + 1,
            args.score, scalars[mean_score_key]['train'],
            args.score, scalars[mean_score_key]['validation'],
            args.score, scalars[mean_score_key]['test']),
        )


def muti_less_log(net, train_dataloader, validation_dataloader, test_dataloader, criterion, epoch, args):
    scalars = muti_evaluate_net(net, train_dataloader, validation_dataloader, test_dataloader, criterion, args,
                                epoch + 1)
    global g
    if not g.evaluate_called:
        run_info = get_run_info(net, args)
        print('net: ' + run_info['net'])
        print('args: {' + run_info['args'] + '}')
        print('****** MODULES: ******')
        for name, description in run_info['modules'].items():
            print(name + ': ' + description)
        print('**********************')

    if (args.score == 'muti_All'):
        s = 'epoch {}, Accuracy: validation mean: {}, testing mean: {}\n'.format(
            epoch + 1,
            #  scalars['F1 score']['train'],
            scalars['Accuracy']['validation'],
            scalars['Accuracy']['test'])
        s += '         Micro precision: validation mean: {}, testing mean: {}\n'.format(
            # scalars['Accuracy']['train'],
            scalars['Micro precision']['validation'],
            scalars['Micro precision']['test'])
        s += '         Micro recall: validation mean: {}, testing mean: {}\n'.format(
            # scalars['Precision']['train'],
            scalars['Micro recall']['validation'],
            scalars['Micro recall']['test'])
        s += '         Micro f1: validation mean: {}, testing mean: {}\n'.format(
            # scalars['Recall']['train'],
            scalars['Micro f1']['validation'],
            scalars['Micro f1']['test'])
        s += '         Macro precision: validation mean: {}, testing mean: {}\n'.format(
            # scalars['auroc']['train'],
            scalars['Macro precision']['validation'],
            scalars['Macro precision']['test'])
        s += '         Macro recall: validation mean: {}, testing mean: {}\n'.format(
            # scalars['auroc']['train'],
            scalars['Macro recall']['validation'],
            scalars['Macro recall']['test'])
        s += '         Macro f1: validation mean: {}, testing mean: {}\n'.format(
            # scalars['auroc']['train'],
            scalars['Macro f1']['validation'],
            scalars['Macro f1']['test'])
        s += '         Best macro f1: validation mean: {}, testing mean: {}, best epoch: {}\n\n'.format(
            #  scalars['best mean']['train'],
            scalars['Best macro f1']['validation'],
            scalars['Best macro f1']['test'],
            scalars['Best macro f1']['epoch'])
        print(s)
        if epoch == 0:
            with open('log.txt', 'w') as f:
                f.write(s)
        else:
            with open('log.txt', 'a') as f:
                f.write(s)
    else:
        mean_score_key = 'mean {}'.format(args.score)
        print('epoch {}, training mean {}: {}, validation mean {}: {}, testing mean {}:{}'.format(
            epoch + 1,
            args.score, scalars[mean_score_key]['train'],
            args.score, scalars[mean_score_key]['validation'],
            args.score, scalars[mean_score_key]['test']),
        )


LOG = {'bi_less': bi_less_log,
       'muti_less': muti_less_log}
