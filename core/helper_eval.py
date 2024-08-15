import torch
import numpy as np
# %% visualization package
from scipy import ndimage
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import skimage.transform
import torch.nn.functional as F
# %%
import pandas as pd
# %%
import pdb

def val_gzsl(test_X, test_label, target_classes, in_package, mix_lamb):
    batch_size = in_package['batch_size']
    model = in_package['model']
    device = in_package['device']
    with (torch.no_grad()):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, batch_size):
            end = min(ntest, start + batch_size)

            input = test_X[start:end].to(device)

            out_package1, out_package2 = model(input)

            #            if type(output) == tuple:        # if model return multiple output, take the first one
            #                output = output[0]
            # output = out_package1['S_pp']
            output = mix_lamb * out_package1['S_pp'] + (1 - mix_lamb) * out_package2['S_pp']
            output[:, target_classes] = output[:, target_classes]
            predicted_label[start:end] = torch.argmax(output.data, 1)

            start = end

        acc = compute_per_class_acc_gzsl(test_label, predicted_label, target_classes, in_package)
        return acc


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size()).fill_(-1)
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i

    return mapped_label


def val_zs_gzsl(test_X, test_label, unseen_classes, in_package, mix_lamb):
    batch_size = in_package['batch_size']
    model = in_package['model']
    device = in_package['device']
    with torch.no_grad():
        start = 0
        ntest = test_X.size()[0]
        predicted_label_gzsl = torch.LongTensor(test_label.size())
        predicted_label_zsl = torch.LongTensor(test_label.size())
        predicted_label_zsl_t = torch.LongTensor(test_label.size())
        for i in range(0, ntest, batch_size):
            end = min(ntest, start + batch_size)

            input = test_X[start:end].to(device)

            out_package1, out_package2 = model(input)

            #            if type(output) == tuple:        # if model return multiple output, take the first one
            #                output = output[0]
            #
            # output = out_package1['S_pp']
            output = mix_lamb * out_package1['S_pp'] + (1 - mix_lamb) * out_package2['S_pp']

            output_t = output.clone()
            output_t[:, unseen_classes] = output_t[:, unseen_classes] + torch.max(output) + 1
            predicted_label_zsl[start:end] = torch.argmax(output_t.data, 1)
            predicted_label_zsl_t[start:end] = torch.argmax(output.data[:, unseen_classes], 1)

            output[:, unseen_classes] = output[:, unseen_classes]
            predicted_label_gzsl[start:end] = torch.argmax(output.data, 1)

            start = end
        acc_gzsl = compute_per_class_acc_gzsl(test_label, predicted_label_gzsl, unseen_classes, in_package)
        acc_zs = compute_per_class_acc_gzsl(test_label, predicted_label_zsl, unseen_classes, in_package)
        acc_zs_t = compute_per_class_acc(map_label(test_label, unseen_classes), predicted_label_zsl_t,
                                         unseen_classes.size(0))

        # assert np.abs(acc_zs - acc_zs_t) < 0.001
        # print('acc_zs: {} acc_zs_t: {}'.format(acc_zs,acc_zs_t))
        return acc_gzsl, acc_zs_t


def compute_per_class_acc(test_label, predicted_label, nclass):
    acc_per_class = torch.FloatTensor(nclass).fill_(0)
    for i in range(nclass):
        idx = (test_label == i)
        acc_per_class[i] = torch.sum(test_label[idx] == predicted_label[idx]).float() / torch.sum(idx).float()
    return acc_per_class.mean().item()


def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes, in_package):
    device = in_package['device']
    per_class_accuracies = torch.zeros(target_classes.size()[0]).float().to(device).detach()

    predicted_label = predicted_label.to(device)

    for i in range(target_classes.size()[0]):
        is_class = test_label == target_classes[i]

        per_class_accuracies[i] = torch.div((predicted_label[is_class] == test_label[is_class]).sum().float(),
                                            is_class.sum().float())
    #        pdb.set_trace()
    return per_class_accuracies.mean().item()


def eval_zs_gzsl(dataloader, model, mix_lamd, device):
    model.eval()
    # print('bias_seen {} bias_unseen {}'.format(bias_seen,bias_unseen))
    test_seen_feature = dataloader.data['test_seen']['resnet_features']
    test_seen_label = dataloader.data['test_seen']['labels'].to(device)

    test_unseen_feature = dataloader.data['test_unseen']['resnet_features']
    test_unseen_label = dataloader.data['test_unseen']['labels'].to(device)

    seenclasses = dataloader.seenclasses
    unseenclasses = dataloader.unseenclasses

    batch_size = 100

    in_package = {'model': model, 'device': device, 'batch_size': batch_size}

    with torch.no_grad():
        acc_seen = val_gzsl(test_seen_feature, test_seen_label, seenclasses, in_package, mix_lamd)
        acc_novel, acc_zs = val_zs_gzsl(test_unseen_feature, test_unseen_label, unseenclasses, in_package, mix_lamd)

    if (acc_seen + acc_novel) > 0:
        H = (2 * acc_seen * acc_novel) / (acc_seen + acc_novel)
    else:
        H = 0

    return acc_seen, acc_novel, H, acc_zs