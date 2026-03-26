import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_adjs(n_text_embeds):
    # n_text_embeds: (bs, n, hdim)
    n_text_embeds = n_text_embeds / n_text_embeds.norm(dim=-1, keepdim=True)
    sims = torch.bmm(n_text_embeds, n_text_embeds.permute(0, 2, 1))  # (bs, n, n)
    sims = torch.sigmoid(sims)
    adjs = torch.where(sims>0.5, torch.ones_like(sims), torch.zeros_like(sims))
    return adjs  # (bs, n, n)


def normalized_adjs(adjs, norm_type='sym'):
    # adjs: (bs, n, n)
    deg = adjs.sum(dim=-1)  # (bs, n)

    if norm_type == 'sym':
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == torch.inf] = 0
        deg_inv_sqrt[deg_inv_sqrt == torch.nan] = 0
        idxs = torch.arange(adjs.shape[1])
        deg_norm = torch.zeros_like(adjs)
        deg_norm[:, idxs, idxs] = deg_inv_sqrt
        return torch.bmm(torch.bmm(deg_norm, adjs), deg_norm)

    elif norm_type == 'rw':
        deg_inv = torch.pow(deg, -1)
        deg_inv[deg_inv == torch.inf] = 0
        deg_inv[deg_inv == torch.nan] = 0
        idxs = torch.arange(adjs.shape[1])
        deg_norm = torch.zeros_like(adjs)
        deg_norm[:, idxs, idxs] = deg_inv
        return torch.bmm(deg_norm, adjs)

    else:
        raise NotImplementedError


class GraphAttention(nn.Module):
    def __init__(self, in_channels, out_channels, nheads=1, leaky_slope=0.1, concat=True, v2=False):
        super(GraphAttention, self).__init__()

        self.out_channels = out_channels
        self.nheads = nheads
        self.concat = concat
        self.leaky_slope = leaky_slope
        self.v2 = v2

        self.lin1 = nn.Linear(in_channels, out_channels*nheads, bias=False)
        self.att_l = nn.Linear(out_channels, 1, bias=False)
        self.att_r = nn.Linear(out_channels, 1, bias=False)

    def reset_parameters(self):
        with torch.no_grad():
            self.lin1.reset_parameters()
            self.att_l.reset_parameters()
            self.att_r.reset_parameters()

    def forward(self, xs, adjs=None, return_atts=False):
        # xs: (bs, n, indim)
        # adjs: (bs, n, n)
        bs, n, _ = xs.shape
        hs = self.lin1(xs).reshape(bs, n, self.nheads, -1).permute(0, 2, 1, 3)  # (bs, nh, n, outdim)

        if not self.v2:  # GAT
            ei = self.att_l(hs).squeeze(-1)  # (bs, nh, n)
            ej = self.att_r(hs).squeeze(-1)  # (bs, nh, n)
            eij = ei.unsqueeze(-1) + ej.unsqueeze(-2)  # (bs, nh, n, n)
            eij = F.leaky_relu(eij, self.leaky_slope)  # (bs, nh, n, n)
        else:  # GATv2
            hs_leakyrelu = F.leaky_relu(hs, self.leaky_slope)
            ei = self.att_l(hs_leakyrelu).squeeze(-1)  # (bs, nh, n)
            ej = self.att_r(hs_leakyrelu).squeeze(-1)  # (bs, nh, n)
            eij = ei.unsqueeze(-1) + ej.unsqueeze(-2)  # (bs, nh, n, n)

        if adjs is not None:
            adjs_tmp = adjs.unsqueeze(1).expand(-1, self.nheads, -1, -1)  # (bs, nh, n, n)
            eij = eij.masked_fill(adjs_tmp==0, -1e18)
            atts = torch.softmax(eij, dim=-1)  # (bs, nh, n, n)
            atts = adjs.unsqueeze(1) * atts  # (bs, nh, n, n)
        else:
            atts = torch.softmax(eij, dim=-1)  # (bs, nh, n, n)
        hs = torch.einsum('bhij,bhjd->bhid', [atts, hs])  # (bs, nh, n, outdim)

        hs = hs.permute(0, 2, 1, 3)  # (bs, n, nh, outdim)
        if self.concat:
            hs = hs.reshape(bs, n, -1)  # (bs, n, nh*outdim)
        else:
            hs = hs.mean(dim=-2)  # (bs, n, outdim)

        if return_atts:
            return hs, (eij, atts)
        return hs


class GAT(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, config):
        super(GAT, self).__init__()

        self.num_layers = num_layers
        self.dropout = config.gnn_dropout
        self.nheads = config.gnn_nheads
        self.leaky_slope = config.gnn_leaky_slope
        self.v2 = config.gnn_v2

        assert out_channels % self.nheads == 0, "out_channels must be divisible by nheads"
        hdim = out_channels // self.nheads

        layers = [GraphAttention(in_channels, hdim, self.nheads, self.leaky_slope, False if self.num_layers == 1 else True, self.v2)]
        for _ in range(self.num_layers - 2):
            layers.append(GraphAttention(hdim*self.nheads, hdim, self.nheads, self.leaky_slope, True, self.v2))
        layers.append(GraphAttention(hdim*self.nheads, out_channels, self.nheads, self.leaky_slope, False, self.v2))
        self.layers = nn.ModuleList(layers)

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for l in range(self.num_layers):
                self.layers[l].reset_parameters()

    def forward(self, xs, adjs=None, return_atts=False):
        # xs: (bs, n, indim)
        # adjs: (bs, n, n)
        for l in range(self.num_layers - 1):
            xs = self.layers[l](xs, adjs)
            xs = F.relu(xs)
            xs = F.dropout(xs, self.dropout, self.training)
        # (bs, n, outdim), (bs, nh, n, n)
        xs, atts = self.layers[-1](xs, adjs, return_atts=True)

        if return_atts:
            return xs, atts
        return xs


class RelationGraphAttention(nn.Module):
    def __init__(self, in_channels, out_channels, nheads=1, nrels=3, leaky_slope=0.1, concat=True):
        super(RelationGraphAttention, self).__init__()

        self.out_channels = out_channels
        self.nheads = nheads
        self.nrels = nrels
        self.leaky_slope = leaky_slope
        self.concat = concat

        lin_rels = []
        att_l_rels = []
        att_r_rels = []
        for _ in range(self.nrels):
            lin_rels.append(nn.Linear(in_channels, nheads*out_channels, bias=False))
            att_l_rels.append(nn.Linear(out_channels, 1, bias=False))
            att_r_rels.append(nn.Linear(out_channels, 1, bias=False))
        self.lin_rels = nn.ModuleList(lin_rels)
        self.att_l_rels = nn.ModuleList(att_l_rels)
        self.att_r_rels = nn.ModuleList(att_r_rels)
        self.lin_out = nn.Linear(in_channels, out_channels, bias=False)

    def reset_parameters(self):
        with torch.no_grad():
            for r in range(self.nrels):
                self.lin_rels[r].reset_parameters()
                self.att_l_rels[r].reset_parameters()
                self.att_r_rels[r].reset_parameters()
            self.lin_out.reset_parameters()

    def forward(self, xs, adjs_list=None, return_atts=False):
        # xs: (bs, n, indim)
        # adjs_list: list of (bs, n, n)
        bs, n, _ = xs.shape
        nh = self.nheads
        assert len(adjs_list) == self.nrels, "adjs_list must be a list of length nrels"

        eij_list = []
        out = 0
        for r in range(self.nrels):
            hs = self.lin_rels[r](xs).reshape(bs, n, nh, -1).permute(0, 2, 1, 3)  # (bs, nh, n, outdim)

            ei = self.att_l_rels[r](hs).squeeze(-1)  # (bs, nh, n)
            ej = self.att_r_rels[r](hs).squeeze(-1)  # (bs, nh, n)
            eij = ei.view(bs, nh, n, 1) + ej.view(bs, nh, 1, n)  # (bs, nh, n, n)
            eij = F.leaky_relu(eij, self.leaky_slope)  # (bs, nh, n, n)
            if adjs_list[r] is not None:
                adjs = adjs_list[r].view(bs, 1, n, n)
                eij = eij.masked_fill(adjs==0, -1e18)
                atts = torch.softmax(eij, dim=-1)  # (bs, nh, n, n)
                atts = adjs * atts  # (bs, nh, n, n)
                eij_list.append(eij)
            else:
                atts = torch.softmax(eij, dim=-1)
                eij_list = [eij]  # (bs, nh, n, n)

            out = out + torch.einsum('bhij,bhjd->bhid', [atts, hs])  # (bs, nh, n, outdim)

        out = out + self.lin_out(xs).view(bs, 1, n, -1)  # (bs, nh, n, outdim)
        out = out.permute(0, 2, 1, 3)  # (bs, n, nh, outdim)
        if self.concat:
            out = out.reshape(bs, n, -1)  # (bs, n, nh*outdim)
        else:
            out = out.mean(dim=-2)  # (bs, n, outdim)

        if return_atts:
            return out, eij_list
        return out


class RGAT(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, config):
        super(RGAT, self).__init__()

        self.num_layers = num_layers
        self.dropout = config.gnn_dropout
        self.nheads = config.gnn_nheads
        self.nrels = config.gnn_nrels
        self.leaky_slope = config.gnn_leaky_slope

        assert out_channels % self.nheads == 0, "out_channels must be divisible by nheads"
        hdim = out_channels // self.nheads

        layers = [RelationGraphAttention(in_channels, hdim, self.nheads, self.nrels, self.leaky_slope, False if self.num_layers == 1 else True)]
        for _ in range(self.num_layers - 2):
            layers.append(RelationGraphAttention(hdim*self.nheads, hdim, self.nheads, self.nrels, self.leaky_slope, True))
        layers.append(RelationGraphAttention(hdim*self.nheads, out_channels, self.nheads, self.nrels, self.leaky_slope, False))
        self.layers = nn.ModuleList(layers)

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for l in range(self.num_layers):
                self.layers[l].reset_parameters()

    def forward(self, xs, adjs_list=None, return_atts=False):
        # xs: (bs, n, indim)
        # adjs_list: list of (bs, n, n)
        if isinstance(adjs_list, torch.Tensor):
            adjs_list = [adjs_list]
        for l in range(self.num_layers - 1):
            xs = self.layers[l](xs, adjs_list)
            xs = F.relu(xs)
            xs = F.dropout(xs, self.dropout, self.training)
        # (bs, n, outdim), (bs, nh, n, n)
        xs, atts_list = self.layers[-1](xs, adjs_list, return_atts=True)

        if return_atts:
            return xs, atts_list
        return xs

