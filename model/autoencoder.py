from matplotlib.backend_bases import ToolContainerBase
from .gat import GAT
from utils.utils import create_norm
from functools import partial
from itertools import chain
from .loss_func import sce_loss
import torch
import torch.nn as nn
import dgl
import random


def build_model(args):
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    negative_slope = args.negative_slope
    mask_rate = args.mask_rate
    alpha_l = args.alpha_l
    n_dim = args.n_dim
    e_dim = args.e_dim

    model = GMAEModel(
        n_dim=n_dim,
        e_dim=e_dim,
        hidden_dim=num_hidden,
        n_layers=num_layers,
        n_heads=4,
        activation="prelu",
        feat_drop=0.1,
        negative_slope=negative_slope,
        residual=True,
        mask_rate=mask_rate,
        norm='BatchNorm',
        loss_fn='sce',
        alpha_l=alpha_l
    )
    return model


class GMAEModel(nn.Module):
    def __init__(self, n_dim, e_dim, hidden_dim, n_layers, n_heads, activation,
                 feat_drop, negative_slope, residual, norm, mask_rate=0.5, loss_fn="sce", alpha_l=2):
        super(GMAEModel, self).__init__()
        self._mask_rate = mask_rate
        self._output_hidden_size = hidden_dim
        self.recon_loss = nn.BCELoss(reduction='mean')
        self.supervised_loss = nn.CrossEntropyLoss()  # 有监督损失函数
        self.contrastive_loss = nn.CosineEmbeddingLoss()  # 对比损失函数
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant_(m.bias, 0)

        self.edge_recon_fc = nn.Sequential(
            nn.Linear(hidden_dim * n_layers * 2, hidden_dim),
            nn.LeakyReLU(negative_slope),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.edge_recon_fc.apply(init_weights)

        assert hidden_dim % n_heads == 0
        enc_num_hidden = hidden_dim // n_heads
        enc_nhead = n_heads

        dec_in_dim = hidden_dim
        dec_num_hidden = hidden_dim

        # 分类器，用于有监督学习
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * n_layers, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # 假设有两个类别
        )
        self.classifier.apply(init_weights)

        # build encoder
        self.encoder = GAT(
            n_dim=n_dim,
            e_dim=e_dim,
            hidden_dim=enc_num_hidden,
            out_dim=enc_num_hidden,
            n_layers=n_layers,
            n_heads=enc_nhead,
            n_heads_out=enc_nhead,
            concat_out=True,
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=0.0,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=True,
        )

        # build decoder for attribute prediction
        self.decoder = GAT(
            n_dim=dec_in_dim,
            e_dim=e_dim,
            hidden_dim=dec_num_hidden,
            out_dim=n_dim,
            n_layers=1,
            n_heads=n_heads,
            n_heads_out=1,
            concat_out=True,
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=0.0,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=False,
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, n_dim))
        self.encoder_to_decoder = nn.Linear(dec_in_dim * n_layers, dec_in_dim, bias=False)

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def encoding_mask_noise(self, g, mask_rate=0.3):
        new_g = g.clone()
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=g.device)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        new_g.ndata["attr"][mask_nodes] = self.enc_mask_token

        return new_g, (mask_nodes, keep_nodes)

    def forward1(self, g):
        loss = self.compute_loss(g)
        return loss

    def forward(self, g, label,dif_g):
        loss = self.compute_loss(g,label,dif_g)
        return loss
    def compute_loss(self, g , label=None,diffent_g=None):
        # Feature Reconstruction
        pre_use_g, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, self._mask_rate)
        pre_use_x = pre_use_g.ndata['attr'].to(pre_use_g.device)
        use_g = pre_use_g
        enc_rep, all_hidden = self.encoder(use_g, pre_use_x, return_hidden=True)
        enc_rep = torch.cat(all_hidden, dim=1)
        rep = self.encoder_to_decoder(enc_rep)

        recon = self.decoder(pre_use_g, rep)
        x_init = g.ndata['attr'][mask_nodes]
        x_rec = recon[mask_nodes]
        recon_rec_val = self.criterion(x_rec, x_init)
        
        
        # Structural Reconstruction
        threshold = min(10000, g.num_nodes())

        negative_edge_pairs = dgl.sampling.global_uniform_negative_sampling(g, threshold)
        positive_edge_pairs = random.sample(range(g.number_of_edges()), threshold)
        positive_edge_pairs = (g.edges()[0][positive_edge_pairs], g.edges()[1][positive_edge_pairs])
        sample_src = enc_rep[torch.cat([positive_edge_pairs[0], negative_edge_pairs[0]])].to(g.device)
        sample_dst = enc_rep[torch.cat([positive_edge_pairs[1], negative_edge_pairs[1]])].to(g.device)
        y_pred = self.edge_recon_fc(torch.cat([sample_src, sample_dst], dim=-1)).squeeze(-1)
        y = torch.cat([torch.ones(len(positive_edge_pairs[0])), torch.zeros(len(negative_edge_pairs[0]))]).to(
            g.device)
        
        recon_loss_val = self.recon_loss(y_pred, y)
        
        supervised_loss = torch.tensor(0.0).to(g.device)
        #有监督学习部分
        if label is not None:
            label = torch.tensor([label], dtype=torch.long).to(g.device)
            # 扩充 label 为与 enc_rep 批量大小一致的 Tensor
            label = label.expand(enc_rep.size(0))
            # 计算分类损失
            supervised_loss = self.supervised_loss(self.classifier(enc_rep), label)
        
        # 紧凑损失部分
        compactness_loss_val =  torch.tensor(0.0).to(g.device)
        if False:
            # 计算良性样本之间的平均距离
            benign_centroid = torch.mean(enc_rep, dim=0, keepdim=True)
            compactness_loss_val = torch.mean(torch.norm(benign_embed - benign_centroid, p=2, dim=1))
        
        # 对比学习部分
        contrastive_loss_val =  torch.tensor(0.0).to(g.device)
        if diffent_g is not None:
            # 对恶意样本图进行编码
            diffent_g_enc_rep, diffent_g_enc_hidden = self.encoder(diffent_g, diffent_g.ndata['attr'], return_hidden=True)
            # 如果 encoder 返回的是一个元组或列表，将它们在特征维度上进行拼接

            diffent_g_enc_rep = torch.cat(diffent_g_enc_hidden, dim=1)
 
            
            # 获取恶意样本和正常样本的嵌入
            malicious_embed = diffent_g_enc_rep
            benign_embed = enc_rep
            # 确保恶意样本和正常样本的嵌入维度一致
            num_malicious = malicious_embed.size(0)
            num_benign = benign_embed.size(0)
            
            # 随机选择1024个节点嵌入进行对比学习
            num_malicious = malicious_embed.size(0)
            num_benign = benign_embed.size(0)
            
            # 随机选择1024个恶意样本
            malicious_indices = torch.randperm(num_malicious)[:1024].to(g.device)
            malicious_embed_sampled = malicious_embed[malicious_indices]
            
            # 随机选择1024个良性样本
            benign_indices = torch.randperm(num_benign)[:1024].to(g.device)
            benign_embed_sampled = benign_embed[benign_indices]
            
            # 创建对比损失的目标标签（-1 表示不同类别）
            contrastive_labels = torch.full((1024,), -1, dtype=torch.long).to(g.device)
            
            # 计算对比损失
            contrastive_loss_val += self.contrastive_loss(malicious_embed_sampled, benign_embed_sampled, contrastive_labels)
            
        loss = recon_rec_val + recon_loss_val + supervised_loss + contrastive_loss_val + compactness_loss_val
        print(f"recon_rec_val: {recon_rec_val.item()}, recon_loss_val: {recon_loss_val.item()}, supervised_loss: {supervised_loss.item()}, contrastive_loss:{contrastive_loss_val.item()}\n")
        return loss
        


        

    def embed(self, g):
        x = g.ndata['attr'].to(g.device)
        rep = self.encoder(g, x)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
