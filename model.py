import torch
import torch.nn as nn
import torch.nn.functional as F
from mirt import irt2pl
from GCCD.fusion import Fusion
import numpy as np
import pandas as pd
from gnn import GCN
from irt import irt3pl
from ContrastiveGcn import ContrastiveLearningGCN

class Net(nn.Module):
    def __init__(self, args, local_map,irf_kwargs=None):
        self.device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
        # self.gclhidden_dim = args.gclhidden_dim
        # self.gclnum_layers = args.gclnum_layers
        self.knowledge_dim = args.knowledge_n
        self.k_n=args.knowledge_n
        #self.uk_n = args.unknowledge_n
        self.exer_n = args.exer_n
        self.emb_num = args.student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.num_layers = 4
        # self.all_stu_emb2=None
        # for ncd
        self.prednet_len1, self.prednet_len2 = 512, 256

        # self.k_from_e = local_map['k_from_e'].to(self.device)
        # self.e_from_k = local_map['e_from_k'].to(self.device)
        # self.u_from_e = local_map['u_from_e'].to(self.device)
        # self.e_from_u = local_map['e_from_u'].to(self.device)

        self.coograph = local_map['coograph'].to(self.device)
        self.rcoograph = local_map['rcoograph'].to(self.device)
        # for RCD
        # self.directed_g = local_map['directed_g'].to(self.device)
        # self.undirected_g = local_map['undirected_g'].to(self.device)

        # IRT
        # self.irf_kwargs = irf_kwargs if irf_kwargs is not None else {}
        # self.value_range = args.value_range
        # self.a_range = args.a_range

        # MIRT
        # self.latent_dim=args.latent_dim

        # DINA
        # self.step = 0
        # self.max_step = 1000
        # self.max_slip = 0.4
        # self.max_guess = 0.4
        # self.hidde_dim=1

        super(Net, self).__init__()

        self.w1 = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        self.w2 = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        self.w3 = nn.Parameter(torch.tensor(0.5, requires_grad=True))

        # network structure ROW NCD
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)

        self.knowledge_emb=nn.Embedding(self.k_n,self.stu_dim)
        # self.knowledge_emb = nn.Embedding(self.emb_num, self.knowledge_dim)

        self.exercise_emb=nn.Embedding(self.exer_n,self.stu_dim)

        self.e_discrimination = nn.Embedding(self.exer_n, 1)

        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)

        # IRT
        # self.c=nn.Embedding(self.exer_n,1)

        # DINA
        # self.guess = nn.Embedding(self.exer_n, 1)
        # self.slip = nn.Embedding(self.exer_n, 1)

        # network structure

        self.k_index = torch.LongTensor(list(range(self.k_n))).to(self.device)
        self.stu_index = torch.LongTensor(list(range(self.emb_num))).to(self.device)
        self.exer_index = torch.LongTensor(list(range(self.exer_n))).to(self.device)

        
        self.ContrastiveGcnLayer = ContrastiveLearningGCN(self.coograph,self.rcoograph,self.knowledge_dim,self.knowledge_dim)

        self.disc = nn.Linear(self.knowledge_dim, 1)
        self.full_stu = nn.Linear(2 * self.knowledge_dim, 1)
        self.full_exer = nn.Linear(2 * self.knowledge_dim, 1)

        # ncd
        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.drop_2 = nn.Dropout(p=0.5)
        # for RCD
        # self.FusionLayer1 = Fusion(args, local_map)

        # self.FusionLayer2 = Fusion(args, local_map)

        # RCD
        # self.prednet_full1 = nn.Linear(2 * args.knowledge_n, args.knowledge_n, bias=False)
        # self.drop_1 = nn.Dropout(p=0.5)
        # self.prednet_full2 = nn.Linear(2 * args.knowledge_n, args.knowledge_n, bias=False)
        # self.drop_2 = nn.Dropout(p=0.5)
        # self.prednet_full3 = nn.Linear(1 * args.knowledge_n, 1)

        # irt
        # self.irtprednet_full1=nn.Linear(args.knowledge_n,1)
        # self.b_layer = nn.Linear(args.knowledge_n, 1)  # New layer for b (difficulty)
        # mirt
        # self.mirtprednet_full1=nn.Linear(args.knowledge_n,self.latent_dim)
        # self.mirtprednet_full2=nn.Linear(args.knowledge_n,1)

        # DINA
        # self.guess = nn.Embedding(self.exer_n, 1)
        # self.slip = nn.Embedding(self.exer_n, 1)
        # self.dinaprednet_full1 = nn.Linear(args.knowledge_n, self.hidde_dim)



        # initialization
        for name, param in self.named_parameters():
            # if 'weight' in name:
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_normal_(param)


    def create_gcn_layers(self, graph, num_layers):
        gcn_layers = nn.ModuleList()
        for _ in range(num_layers):
            gcn_layers.append(GCN(graph, self.knowledge_dim, self.knowledge_dim, None))
        return gcn_layers
    def forward(self, stu_id, exer_id, kn_r):


        all_stu_emb = self.student_emb(self.stu_index).to(self.device)

        exer_emb = self.exercise_emb(self.exer_index).to(self.device)

        kn_emb = self.knowledge_emb(self.k_index).to(self.device)

        kn_emb, contrastive_loss_1, contrastive_loss_2, contrastive_loss_3 = self.ContrastiveGcnLayer(kn_emb)

        # for RCD
        # Fusion layer 1
        # kn_emb1, exer_emb1, all_stu_emb1 = self.FusionLayer1(kn_emb, exer_emb, all_stu_emb)
        # Fusion layer 2
        # kn_emb2, exer_emb2, all_stu_emb2 = self.FusionLayer2(kn_emb1, exer_emb1, all_stu_emb1)

       
        # kn_emb1, contrastive_loss_1, contrastive_loss_2, contrastive_loss_3 = self.ContrastiveGcnLayer(kn_emb)

       
        # kn_emb = kn_emb2
        # all_stu_emb = all_stu_emb2
        # exer_emb = exer_emb2

        # get batch student data
        batch_stu_emb = all_stu_emb[stu_id]
        batch_stu_vector = batch_stu_emb.repeat(1, batch_stu_emb.shape[1]).reshape(batch_stu_emb.shape[0], batch_stu_emb.shape[1], batch_stu_emb.shape[1])#8 13 13

        # get batch exercise data
        batch_exer_emb = exer_emb[exer_id]
        batch_exer_vector = batch_exer_emb.repeat(1, batch_exer_emb.shape[1]).reshape(batch_exer_emb.shape[0], batch_exer_emb.shape[1], batch_exer_emb.shape[1])# 8 13 13

        # get batch knowledge concept data
        kn_vector = kn_emb.repeat(batch_stu_emb.shape[0], 1).reshape(batch_stu_emb.shape[0], kn_emb.shape[0],kn_emb.shape[1])


        # RCD
        # kn_vector = kn_emb.repeat(batch_stu_emb.shape[0], 1).reshape(batch_stu_emb.shape[0], kn_emb.shape[0], kn_emb.shape[1])



        # Cognitive diagnosis

        

        # NCD
        proficiency = torch.sigmoid(self.full_stu(torch.cat((batch_stu_vector, kn_vector), dim=2))).squeeze(2)
        k_difficulty = torch.sigmoid(self.full_exer(torch.cat((batch_exer_vector, kn_vector), dim=2))).squeeze(2)
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10

        #
        # # NCD prednet
        e_discrimination = torch.clamp(e_discrimination, min=-1e5, max=1e5)
        proficiency = torch.clamp(proficiency, min=-1e5, max=1e5)
        k_difficulty = torch.clamp(k_difficulty, min=-1e5, max=1e5)
        kn_r = torch.clamp(kn_r, min=-1e5, max=1e5)

        input_x = e_discrimination * (proficiency - k_difficulty) * kn_r
        input_x = torch.relu(self.prednet_full1(input_x))
        input_x = torch.relu(self.prednet_full2(input_x))
        output = torch.sigmoid(self.prednet_full3(input_x))


        # RCD
        # preference = torch.sigmoid(self.prednet_full1(torch.cat((batch_stu_vector, kn_vector), dim=2)))
        # diff = torch.sigmoid(self.prednet_full2(torch.cat((batch_exer_vector, kn_vector), dim=2)))
        # o = torch.sigmoid(self.prednet_full3(preference - diff))

        # sum_out = torch.sum(o * kn_r.unsqueeze(2), dim = 1)
        # count_of_concept = torch.sum(kn_r, dim = 1).unsqueeze(1)
        # output = sum_out / count_of_concept


        # IRT
        '''
        stu_emb = torch.sigmoid(batch_stu_emb)
        theta = self.irtprednet_full1(stu_emb).squeeze(-1)  # Use the existing layer for theta
        # k_difficulty = torch.sigmoid(batch_exer_emb)
        k_difficulty = torch.sigmoid(self.full_exer(torch.cat((batch_exer_vector, kn_vector), dim=2))).squeeze(2)
        b = self.b_layer(k_difficulty).squeeze(-1)  # Use the new b_layer for difficulty

        e_discrimination = self.e_discrimination(exer_id)
        a = F.softplus(e_discrimination).squeeze(-1)  # Use softplus to ensure positivity for a

        c = torch.sigmoid(self.c(exer_id)).squeeze(-1)  # Only apply sigmoid once to c

        if self.value_range is not None:
            theta = self.value_range * (theta - 0.5)  # Adjust theta range
            b = self.value_range * (b - 0.5)  # Adjust b range
        if self.a_range is not None:
            a = self.a_range * torch.sigmoid(a)  # Scale a by a_range

        # NaN check
        if torch.isnan(theta).any() or torch.isnan(a).any() or torch.isnan(b).any():
            raise ValueError('NaN detected in theta, a, or b. Check value_range or a_range.')

        # Calculate the IRT output using the irf function
        output = self.irf(theta, a, b, c, **self.irf_kwargs)
        output = output.unsqueeze(1)
        '''




        # MIRT
        '''
        stu_emb = torch.sigmoid(torch.matmul(batch_stu_emb, kn_emb.T))
        theta=self.mirtprednet_full1(stu_emb)
        theta=torch.squeeze(theta,dim=-1)
        # #
        k_difficulty = torch.sigmoid(torch.matmul(batch_exer_emb, kn_emb.T))
        a=self.mirtprednet_full1(k_difficulty)
        a=torch.squeeze(a,dim=-1)
        # #
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        b=torch.squeeze(e_discrimination,dim=-1)
        # #
        output=self.irf(theta,a,b,**self.irf_kwargs)
        output = output.unsqueeze(1)
        '''


        # DINA
        '''
        # stu_emb = torch.sigmoid(self.full_stu(torch.cat((batch_stu_vector, kn_vector), dim=2))).squeeze(2)
        stu_emb = torch.sigmoid(batch_stu_emb)
        theta = self.dinaprednet_full1(stu_emb)
        slip = torch.squeeze(torch.sigmoid(self.slip(exer_id)) * self.max_slip)
        guess = torch.squeeze(torch.sigmoid(self.guess(exer_id)) * self.max_guess)
        if self.training:
            n = torch.sum(kn_r * (torch.sigmoid(theta) - 0.5), dim=1)
            t, self.step = max((np.sin(2 * np.pi * self.step / self.max_step) + 1) / 2 * 100,
                               1e-6), self.step + 1 if self.step < self.max_step else 0
            output=torch.sum(
                torch.stack([1 - slip, guess]).T * torch.softmax(torch.stack([n, torch.zeros_like(n)]).T / t, dim=-1),
                dim=1
            )
        else:
            n = torch.prod(kn_r * (theta >= 0) + (1 - kn_r), dim=1)
            output=(1 - slip) ** n * guess ** (1 - n)
        output = output.unsqueeze(1)
        '''

        return output, contrastive_loss_1, contrastive_loss_2, contrastive_loss_3

    def get_stu_know(self,stu_id):
        stat_emb=torch.sigmoid(self.all_stu_emb2[stu_id])
        return stat_emb.data
    def get_exer(self,exer_id):
        stat_emb=torch.sigmoid(self.exer_emb2[exer_id])
        return stat_emb.data


    #irt
    '''
    @classmethod
    def irf(cls, theta, a, b, c, **kwargs):
        return irt3pl(theta, a, b, c, F=torch, **kwargs)
    '''
    # mirt

    @classmethod
    def irf(cls, theta, a, b, **kwargs):
        return irt2pl(theta, a, b, F=torch)


    def apply_clipper(self):

        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)




class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)
