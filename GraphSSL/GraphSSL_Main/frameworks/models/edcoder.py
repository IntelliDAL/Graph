from typing import Optional
from itertools import chain
from functools import partial
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from .gin import GIN
from .gat import GAT
from .gcn import GCN
from .dot_gat import DotGAT
from frameworks.utils import create_norm
from .simple_align import Simplealign as spp
from .simple_align import Reversealign as rpp
from .structure_encode import GraphStructureEncoder as GSE

from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "dotgat":
        mod = DotGAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gcn":
        mod = GCN(
            in_dim=in_dim, 
            num_hidden=num_hidden, 
            out_dim=out_dim, 
            num_layers=num_layers, 
            dropout=dropout, 
            activation=activation, 
            residual=residual, 
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding")
        )
    elif m_type == "mlp":
        # * just for decoder 
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError
    
    return mod


class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,
            encoder_type: str = "gat",
            decoder_type: str = "gat",
            loss_fn: str = "sce",
            drop_edge_rate: float = 0.0,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
            fea_len: int = 271,
            num_classes: int = 2
         ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate

        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden
        
        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden 

        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        # build decoder for attribute prediction
        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            num_layers=1,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        self.weight_add = nn.Linear(num_hidden * 2, num_hidden, bias=False)

        self.align_size = 16
        self.Simplealign = spp(fea_len, self.align_size, 0.2)
        self.Reversealign = rpp(num_hidden, self.align_size, 0.2)
        self.GraphStruEnc = GSE(self.align_size, num_hidden)

        # * setup loss function
        # self.criterion = nn.MSELoss()
        self.criterion = partial(sce_loss, alpha=alpha_l)

        self.classify = nn.Sequential(
            nn.Linear(num_hidden, 800, bias=True),
            # nn.Dropout(0.5),
            nn.LeakyReLU(0.33),
            nn.Linear(800, 64, bias=True),
            # nn.Dropout(0.5),
            nn.LeakyReLU(0.33)
        )

        self.classify_1 = nn.Linear(64, num_classes, bias=True) # tran=False
        self.classify_2 = nn.Linear(64, num_classes, bias=True) # tran=True

        self.pooler = AvgPooling()
        # self.pooler = MaxPooling()
        # self.pooler = SumPooling()


    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion
    
    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()

        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] = out_x[token_nodes] + self.enc_mask_token
        use_g = g.clone()

        # use_g.ndata['attr'] = out_x

        return use_g, out_x, (mask_nodes, keep_nodes)
    
    def noise_aug(self, g, x):
        noise_rate = 0.3
        num_nodes = g.num_nodes()
        num_noise_nodes = int(noise_rate * num_nodes)

        noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

        aug_x = x.clone()
        aug_x[noise_to_be_chosen] = 0.0
        use_g = g.clone()

        return use_g, aug_x

    def masked_node_delete_edge(self, graph, unbatch, mask_nodes):
        n_node = graph.num_nodes()
        
        src = graph.edges()[0].clone()
        dst = graph.edges()[1].clone()

        node_split = graph.batch_num_nodes()
        edge_split = graph.batch_num_edges()

        after_mask_src = src.clone()
        after_mask_dst = dst.clone()
        after_mask_edge_split = edge_split.clone()
        save_mask_edge_split = edge_split.clone()
        save_mask_src = src.clone()
        save_mask_dst = dst.clone()

        masked_edge_TF = torch.isin(after_mask_src, mask_nodes) & torch.isin(after_mask_dst, mask_nodes)

        batch_sum = 0
        for i in range(0, len(unbatch)):
            after_mask_edge_split[i] = torch.sum(masked_edge_TF[int(batch_sum):int(batch_sum + edge_split[i])])
            save_mask_edge_split[i] = edge_split[i] - after_mask_edge_split[i]
            batch_sum = batch_sum + edge_split[i]

        masked_edge_TF = masked_edge_TF==1
        save_masked_edge_TF = ~ masked_edge_TF

        # after mask
        after_mask_src = after_mask_src[masked_edge_TF]
        after_mask_dst = after_mask_dst[masked_edge_TF]

        after_mask_g = dgl.graph((after_mask_src, after_mask_dst), num_nodes=n_node)

        # save mask
        save_mask_src = save_mask_src[save_masked_edge_TF]
        save_mask_dst = save_mask_dst[save_masked_edge_TF]

        save_mask_g = dgl.graph((save_mask_src, save_mask_dst), num_nodes=n_node)

        return after_mask_g, save_mask_g, node_split, after_mask_edge_split, save_mask_edge_split


    def forward(self, g, unbatch_g, x, upstream=True, tran=False):
        if upstream:
            aug_g, aug_x = g, x

            pack_data, adj_fea, adj_batch = self.mask_attr_encoder(g, unbatch_g, x)
            _, aug_adj_fea, adj_aug_batch = self.mask_attr_encoder(aug_g, unbatch_g, aug_x)

            rec_loss, x_rec_con, x_init_con = self.mask_attr_decoder(pack_data, adj_fea)

            return rec_loss, adj_batch, adj_aug_batch, x_rec_con, x_init_con

        else:
            return self.embed_new(g, x, tran=tran)


    def embed_new(self, g, x, tran=False):
        rep = self.encoder(g, x)

        align_adj, _ = self.Simplealign(g, x)
        align_adj_temp = align_adj.unsqueeze(0)
        align_adj_batch = align_adj_temp[:, 0:self.align_size, 0:self.align_size]
        for i in range(self.align_size, align_adj.shape[0], self.align_size):
            align_adj_batch = torch.cat((align_adj_batch, align_adj_temp[:, i:i + self.align_size, i:i + self.align_size]), dim=0)
        adj_fea = self.GraphStruEnc(align_adj_batch)

        adj_fea = adj_fea.permute(0, 2, 1, 3)
        adj_fea = torch.reshape(adj_fea, (adj_fea.shape[0] * adj_fea.shape[1], adj_fea.shape[2]))

        g_clone = g.clone()
        g_clone.ndata['attr'] = rep
        adj_fea = self.Reversealign(g_clone, rep, adj_fea)
        rep = rep + adj_fea

        rep = self.encoder_to_decoder(rep)
        rep = self.pooler(g, rep)

        pred = self.classify(rep)

        if tran is False:
            pred = self.classify_1(pred)
        else:
            pred = self.classify_2(pred)

        pred = torch.nn.functional.softmax(pred, dim=-1)

        return pred

    def mask_attr_encoder(self, g, unbatch_g, x):

        pre_use_g, use_x, (mask_nodes, _) = self.encoding_mask_noise(g, x, self._mask_rate)
        save_masked_edge_g = None
        use_g = pre_use_g
        g_maskded = pre_use_g

        align_adj, _ = self.Simplealign(g_maskded, x)
        align_adj_temp = align_adj.unsqueeze(0)
        align_adj_batch = align_adj_temp[:, 0:self.align_size, 0:self.align_size]
        for i in range(self.align_size, align_adj.shape[0], self.align_size):
            align_adj_batch = torch.cat((align_adj_batch, align_adj_temp[:, i:i + self.align_size, i:i + self.align_size]), dim=0)
        adj_fea = self.GraphStruEnc(align_adj_batch)

        adj_fea = adj_fea.permute(0, 2, 1, 3)
        adj_fea = torch.reshape(adj_fea, (adj_fea.shape[0] * adj_fea.shape[1], adj_fea.shape[2]))

        align_adj_batch = torch.sum(align_adj_batch, dim=1)

        return (use_g, x, use_x, pre_use_g, mask_nodes, save_masked_edge_g), adj_fea, align_adj_batch


    def mask_attr_decoder(self, pack_data, adj_fea):
        (use_g, x, use_x, pre_use_g, mask_nodes, save_masked_edge_g) = pack_data

        node_rep, _ = self.encoder(use_g, use_x, strufea=None, return_hidden=True)

        node_rep[mask_nodes] = 0

        use_g_clone = use_g.clone()
        use_g_clone.ndata['attr'] = node_rep
        adj_fea = self.Reversealign(use_g_clone, node_rep, adj_fea)

        rep = node_rep + adj_fea
        rep = self.encoder_to_decoder(rep)

        recon = self.decoder(pre_use_g, rep, strufea=None, return_hidden=False)
        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]
        rec_loss = self.criterion(x_rec, x_init)

        adj_rec = torch.mm(recon, recon.permute(1, 0))
        adj_rec[mask_nodes, mask_nodes] = 1
        adj_rec[~mask_nodes, ~mask_nodes] = 0

        use_g_adj = dgl.remove_self_loop(use_g)
        use_g_adj = use_g_adj.adjacency_matrix(transpose=True, ctx=use_g_adj.device)
        use_g_adj = use_g_adj.to_dense()
        rec_stru_loss = self.criterion(adj_rec, use_g_adj)


        x_rec_con, x_init_con = AvgPooling()(pre_use_g, recon), AvgPooling()(pre_use_g, x)

        return rec_loss + rec_stru_loss, x_rec_con, x_init_con

    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
