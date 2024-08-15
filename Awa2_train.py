import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from core.DEDN_Awa2 import DEDN
from core.AWA2DataLoader import AWA2DataLoader
from core.helper_eval import eval_zs_gzsl
# from global_setting import NFS_path
import importlib
import pdb
import numpy as np
import matplotlib.pyplot as plt
from config import get_args

args = get_args()
print(args)

NFS_path = './'
idx_GPU = 0
device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")
dataloader = AWA2DataLoader(NFS_path, device, is_scale=args.is_scale, is_balance=args.is_balance)
torch.backends.cudnn.benchmark = True

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

batch_size = args.batch_size
nepoches = args.epochs
niters = dataloader.ntrain * nepoches//batch_size
init_w2v_att = dataloader.w2v_att
att = dataloader.att
normalize_att = dataloader.normalize_att

seenclass = dataloader.seenclasses
unseenclass = dataloader.unseenclasses
report_interval = niters//nepoches

model = DEDN(args.dim_f,args.dim_v,args.dim_r,args.hidd_f,init_w2v_att,att,normalize_att,
            seenclass,unseenclass,args.lamb,args.trainable_w2v,args.normalize_V,args.normalize_F,
            args.is_bias,args.bias, args.c_mix, args.mal)
model.to(device)

params_to_update = []
for name,param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)

optimizer = optim.RMSprop(params_to_update, lr=args.lr,
                          weight_decay=args.weight_decay, momentum=args.momentum)

best_performance = [0, 0, 0]
best_acc = 0
for i in range(0, niters):
    model.train()
    optimizer.zero_grad()

    batch_label, batch_feature, batch_att = dataloader.next_batch(batch_size)

    out_package1, out_package2 = model(batch_feature)

    out_package1['batch_label'] = batch_label
    out_package2['batch_label'] = batch_label

    out1 = model.compute_loss(out_package1)
    loss = out1['loss']
    out2 = model.compute_loss(out_package2)
    lossa = out2['loss']
    contrastive_loss = model.compute_contrastive_loss(out_package1, out_package2)
    conloss1, conloss2 = out_package1['Con_loss'], out_package2['Con_loss']
    loss = loss + lossa + args.p_ce*contrastive_loss + args.p_crc*(conloss1 + conloss2)

    loss.backward()
    optimizer.step()
    if i % report_interval == 0:
        print('-' * 30)
        acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(dataloader, model, args.e_mix, device)

        if H > best_performance[2]:
            best_performance = [acc_novel, acc_seen, H]
        if acc_zs > best_acc:
            best_acc = acc_zs
        print('iter=%d, loss=%.3f, conloss=%.3f, distill_loss=%.3f' % (
            i, loss.item(), conloss1.item(), contrastive_loss.item()))
        print('lossa=%.3f, conlossa=%.3f' % (
            lossa.item(), conloss2.item()))
        print('acc_unseen=%.3f, acc_seen=%.3f, H=%.3f, acc_zs=%.3f' % (
            best_performance[0], best_performance[1], best_performance[2], best_acc))