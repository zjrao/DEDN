# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 17:39:45 2019

@author: badat
"""
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# %%

# texture 0: 1-18
# Limb 1: 19-33
# abstract 2: 34-64, 79-85
# environment 3: 65-78

indicator = torch.tensor([0] * 85)
indicator[18:33] = 1
indicator[33:64] = indicator[78:85] = 2
indicator[64:78] = 3

num_cls = torch.tensor([18, 14, 40, 13])
indexa = torch.tensor([0, 18, 32, 72, 85])


class SubNet(nn.Module):
    def __init__(self, init_w2v_att, dim_f, dim_v, dim_r, trainable):
        super(SubNet, self).__init__()
        self.init_w2v_att = F.normalize(torch.tensor(init_w2v_att))
        self.V = nn.Parameter(self.init_w2v_att.clone(), requires_grad=trainable)

        self.H_1 = nn.Parameter(nn.init.normal_(torch.empty(dim_v, dim_f)), requires_grad=True)
        self.H_2 = nn.Parameter(nn.init.zeros_(torch.empty(dim_v, dim_f)), requires_grad=True)

        self.H_3 = nn.Parameter(nn.init.normal_(torch.empty(dim_v, dim_r*dim_r)), requires_grad=True)
        self.H_4 = nn.Parameter(nn.init.zeros_(torch.empty(dim_v, dim_r*dim_r)), requires_grad=True)

    def forward(self, Fs):

        S_c = torch.einsum('iv,vf,bfr->bir', self.V, self.H_1, Fs)
        A_c = torch.einsum('iv,vf,bfr->bir', self.V, self.H_2, Fs)
        A_c = F.softmax(A_c, dim=-1)
        S_pc = torch.einsum('bir,bir->bir', A_c, S_c)

        Fs = Fs.transpose(1, 2)
        S_c = torch.einsum('iv,vf,bfr->bir', self.V, self.H_3, Fs)
        A_c = torch.einsum('iv,vf,bfr->bir', self.V, self.H_4, Fs)
        A_c = F.softmax(A_c, dim=-1)
        S_pg = torch.einsum('bir,bir->bir', A_c, S_c)

        return  S_pc, S_pg


class DEDN(nn.Module):
    #####
    # einstein sum notation
    # b: Batch size \ f: dim feature=2048 \ v: dim w2v=300 \ r: number of region=49 \ k: number of classes
    # i: number of attribute=312 \ h : hidden attention dim
    #####
    def __init__(self, dim_f, dim_v, dim_r, hidd_f, init_w2v_att, att, normalize_att,
                 seenclass, unseenclass, lambda_, trainable_w2v, normalize_V, normalize_F,
                 is_bias, bias, mixcr, mal):
        super(DEDN, self).__init__()
        self.dim_f = dim_f
        self.dim_v = dim_v
        self.dim_r = dim_r
        self.hidd_f = hidd_f
        self.mixcr = mixcr
        self.dim_att = att.shape[1]
        self.nclass = att.shape[0]
        self.hidden = self.dim_att // 2
        self.init_w2v_att = init_w2v_att
        self.lambda_ = lambda_
        self.mal = mal

        att = torch.cat((att[:, indicator==0], att[:, indicator==1], att[:, indicator==2],
                         att[:, indicator==3]), dim=1)
        self.att = nn.Parameter(F.normalize(torch.tensor(att)), requires_grad=False)

        self.GNet = SubNet(init_w2v_att, dim_f, dim_v, dim_r, trainable_w2v)

        self.Snet0 = SubNet(init_w2v_att[indicator == 0], dim_f, dim_v, dim_r, trainable_w2v)
        self.Snet1 = SubNet(init_w2v_att[indicator == 1], dim_f, dim_v, dim_r, trainable_w2v)
        self.Snet2 = SubNet(init_w2v_att[indicator == 2], dim_f, dim_v, dim_r, trainable_w2v)
        self.Snet3 = SubNet(init_w2v_att[indicator == 3], dim_f, dim_v, dim_r, trainable_w2v)

        self.weight_ce = nn.Parameter(torch.eye(self.nclass).float(),
                                      requires_grad=False)

        self.normalize_V = normalize_V
        self.normalize_F = normalize_F
        self.is_bias = is_bias

        self.seenclass = seenclass
        self.unseenclass = unseenclass
        self.normalize_att = normalize_att

        self.log_softmax_func = nn.LogSoftmax(dim=1)

        if is_bias:
            self.bias = nn.Parameter(torch.tensor(bias), requires_grad=False)
            mask_bias = np.ones((1, self.nclass))
            mask_bias[:, self.seenclass.cpu().numpy()] *= -1
            self.mask_bias = nn.Parameter(torch.tensor(mask_bias).float(), requires_grad=False)

    def compute_loss_Self_Calibrate(self, in_package):
        S_pp = in_package['S_pp']
        Prob_all = F.softmax(S_pp, dim=-1)
        Prob_unseen = Prob_all[:, self.unseenclass]
        assert Prob_unseen.size(1) == len(self.unseenclass)
        mass_unseen = torch.sum(Prob_unseen, dim=1)
        loss_pmp = -torch.log(torch.mean(mass_unseen))
        return loss_pmp

    def compute_V(self):
        if self.normalize_V:
            V_n = F.normalize(self.V)
        else:
            V_n = self.V
        return V_n

    def compute_aug_cross_entropy(self, in_package):
        batch_label = in_package['batch_label']
        S_pp = in_package['S_pp']

        Labels = batch_label

        if self.is_bias:
            S_pp = S_pp - self.vec_bias  # remove the margin +1/-1 from prediction scores
            if self.mal:
                _label = batch_label.clone().detach().float()
                S_pp[:, self.seenclass] = S_pp[:, self.seenclass] + self.bias
                S_pp = S_pp - _label * self.bias * 2

        Prob = self.log_softmax_func(S_pp)

        loss = -torch.einsum('bk,bk->b', Prob, Labels)
        loss = torch.mean(loss)
        return loss

    def compute_loss(self, in_package):

        if len(in_package['batch_label'].size()) == 1:
            in_package['batch_label'] = self.weight_ce[in_package['batch_label']]

        loss_CE = self.compute_aug_cross_entropy(in_package)

        ## loss self-calibration
        loss_cal = self.compute_loss_Self_Calibrate(in_package)

        ## total loss
        loss = loss_CE + self.lambda_ * loss_cal

        out_package = {'loss': loss, 'loss_CE': loss_CE,
                       'loss_cal': loss_cal}

        return out_package

    def compute_contrastive_loss1(self, S_pp1, S_pp2):
        #S_pp1, S_pp2 = in_package1['S_p'], in_package2['S_p']
        if S_pp1.dim() == 3:
            S_pp1 = S_pp1.reshape(S_pp1.shape[0] * S_pp1.shape[1], S_pp1.shape[2])
            S_pp2 = S_pp2.reshape(S_pp2.shape[0] * S_pp2.shape[1], S_pp2.shape[2])
        wt = (S_pp1 - S_pp2).pow(2)
        wt = wt / ((wt.sum(1)+1e-9).sqrt().unsqueeze(1).expand(wt.size(0), wt.size(1)))
        loss = wt * (S_pp1 - S_pp2).abs()
        loss = (loss.sum() / loss.size(0))

        # JSD
        KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        p_output = F.softmax(S_pp1)
        q_output = F.softmax(S_pp2)
        log_mean_output = ((p_output + q_output) / 2).log()
        loss += (KLDivLoss(log_mean_output, q_output) + KLDivLoss(log_mean_output, p_output)) / 2

        return loss

    def compute_contrastive_loss(self, in_package1, in_package2):

        closs = self.compute_contrastive_loss1(in_package1['S_pp'], in_package2['S_pp'])

        return closs

    def forward(self, Fs):

        shape = Fs.shape
        Fs = Fs.reshape(shape[0], shape[1], shape[2] * shape[3])  # batch x 2048 x 49

        if self.normalize_F:  # true
            Fs = F.normalize(Fs, dim=1)

        S_p0, S_pg0 = self.Snet0(Fs)
        S_p1, S_pg1 = self.Snet1(Fs)
        S_p2, S_pg2 = self.Snet2(Fs)
        S_p3, S_pg3 = self.Snet3(Fs)

        S_paa = torch.cat((S_p0, S_p1, S_p2, S_p3), dim=1)
        S_pa = S_paa.sum(2)
        S_pg = torch.cat((S_pg0, S_pg1, S_pg2, S_pg3), dim=1)
        S_pg = S_pg.sum(2)
        conloss1 = self.compute_contrastive_loss1(S_pa, S_pg)
        if self.training:
            S_pa = S_pa + S_pg
        else:
            S_pa = self.mixcr * S_pa + (1-self.mixcr) * S_pg

        # A_b_p = self.att.new_full((shape[0], self.dim_att), fill_value=1)
        S_ppA = torch.einsum('ki, bi->bik', self.att, S_pa)
        S_ppA = torch.sum(S_ppA, dim=1)  # [bk] <== [bik]

        if self.is_bias:
            self.vec_bias = self.mask_bias * self.bias
            S_ppA = S_ppA + self.vec_bias

        G_p, G_pg = self.GNet(Fs)
        G_paa = torch.cat((G_p[:, indicator == 0], G_p[:, indicator == 1], G_p[:, indicator == 2],
                           G_p[:, indicator == 3]), dim=1)
        G_pa = G_paa.sum(2)
        G_pag = torch.cat((G_pg[:, indicator == 0], G_pg[:, indicator == 1], G_pg[:, indicator == 2],
                           G_pg[:, indicator == 3]), dim=1)
        G_pag = G_pag.sum(2)
        conloss2 = self.compute_contrastive_loss1(G_pa, G_pag)
        if self.training:
            G_pa = G_pa + G_pag
        else:
            G_pa = self.mixcr * G_pa + (1-self.mixcr) * G_pag

        S_pp = torch.einsum('ki, bi->bik', self.att, G_pa)
        S_pp = torch.sum(S_pp, dim=1)  # [bk] <== [bik]

        # S_pp = (S_pp + S_ppA) / 2

        # augment prediction scores by adding a margin of 1 to unseen classes and -1 to seen classes
        if self.is_bias:
            self.vec_bias = self.mask_bias * self.bias
            S_pp = S_pp + self.vec_bias

        package1 = {'S_pp': S_pp, 'S_p': G_pa, 'S_pa': G_paa, 'Con_loss': conloss2}
        package2 = {'S_pp': S_ppA, 'S_p': S_pa, 'S_pa': S_paa, 'Con_loss': conloss1}

        return package1, package2

# %%
#