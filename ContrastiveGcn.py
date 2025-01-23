import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
import dgl


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        x = torch.mm(adj, x)
        x = self.linear(x)
        return F.relu(x)


class SelfAttentionFusion(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionFusion, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)

        
        attn_weights = self.softmax(torch.bmm(q.unsqueeze(1), k.unsqueeze(2)))
        attn_weights = attn_weights.squeeze(-1)

        
        fusion = attn_weights * v + (1 - attn_weights) * x1

        return F.relu(fusion)


class ContrastiveLearningGCN(nn.Module):
    def __init__(self, coograph, rcoograph, knowledge_dim, hidden_dim):
        super(ContrastiveLearningGCN, self).__init__()

        
        self.coograph_adj = coograph.adjacency_matrix().to_dense().to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.rcoograph_adj = rcoograph.adjacency_matrix().to_dense().to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        
        self.gcn_con_1 = GCNLayer(knowledge_dim, hidden_dim)
        self.gcn_rcon_1 = GCNLayer(knowledge_dim, hidden_dim)
        self.gcn_con_2 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn_rcon_2 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn_con_3 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn_rcon_3 = GCNLayer(hidden_dim, hidden_dim)

        
        self.fusion_layer_1 = SelfAttentionFusion(hidden_dim)
        self.fusion_layer_2 = SelfAttentionFusion(hidden_dim)
        self.fusion_layer_3 = SelfAttentionFusion(hidden_dim)

    def forward(self, x):
        
        z_con_1 = self.gcn_con_1(x, self.coograph_adj)
        z_rcon_1 = self.gcn_rcon_1(x, self.rcoograph_adj)

        
        contrastive_loss_1 = self.info_nce_loss(z_con_1, z_rcon_1)
        enhanced_con_1 = self.enhance_with_contrastive_loss(z_con_1, z_rcon_1, contrastive_loss_1)
        enhanced_rcon_1 = self.enhance_with_contrastive_loss(z_rcon_1, z_con_1, contrastive_loss_1)

        
        z_con_2 = self.gcn_con_2(enhanced_con_1, self.coograph_adj)
        z_rcon_2 = self.gcn_rcon_2(enhanced_rcon_1, self.rcoograph_adj)

       
        z_con_2 = self.fusion_layer_1(enhanced_con_1, z_con_2)
        z_rcon_2 = self.fusion_layer_1(enhanced_rcon_1, z_rcon_2)

       
        contrastive_loss_2 = self.info_nce_loss(z_con_2, z_rcon_2)
        enhanced_con_2 = self.enhance_with_contrastive_loss(z_con_2, z_rcon_2, contrastive_loss_2)
        enhanced_rcon_2 = self.enhance_with_contrastive_loss(z_rcon_2, z_con_2, contrastive_loss_2)

        
        z_con_3 = self.gcn_con_3(enhanced_con_2, self.coograph_adj)
        z_rcon_3 = self.gcn_rcon_3(enhanced_rcon_2, self.rcoograph_adj)

        
        z_con_3 = self.fusion_layer_2(enhanced_con_2, z_con_3)
        z_rcon_3 = self.fusion_layer_2(enhanced_rcon_2, z_rcon_3)

        
        contrastive_loss_3 = self.info_nce_loss(z_con_3, z_rcon_3)
        enhanced_con_3 = self.enhance_with_contrastive_loss(z_con_3, z_rcon_3, contrastive_loss_3)
        enhanced_rcon_3 = self.enhance_with_contrastive_loss(z_rcon_3, z_con_3, contrastive_loss_3)

        
        enhanced_con_3 = self.fusion_layer_3(z_con_3, enhanced_con_3)
        enhanced_rcon_3 = self.fusion_layer_3(z_rcon_3, enhanced_rcon_3)

        
        enhanced_con_3 = normalize(enhanced_con_3, dim=1)
        enhanced_rcon_3 = normalize(enhanced_rcon_3, dim=1)

        
        return enhanced_con_3, contrastive_loss_1, contrastive_loss_2, contrastive_loss_3

    def info_nce_loss(self, z1, z2, temperature=0.5):
        
        z1 = normalize(z1, dim=1)
        z2 = normalize(z2, dim=1)

        
        positive_score = torch.sum(z1 * z2, dim=1)

        
        negative_score = torch.mm(z1, z2.T)
        negative_score.fill_diagonal_(0)  

        
        logits = torch.cat([positive_score.unsqueeze(1), negative_score], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)  

        
        loss = F.cross_entropy(logits / temperature, labels)
        return loss

    def enhance_with_contrastive_loss(self, z1, z2, contrastive_loss):
        
        enhanced = z1 * (1 - contrastive_loss) + z2 * contrastive_loss
        return enhanced
