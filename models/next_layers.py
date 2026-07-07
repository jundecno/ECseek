from models.modules import *
from torch import nn
from torch_geometric.nn.aggr import AttentionalAggregation


class PocEnc(nn.Module):

    def __init__(
        self,
        enz_node_dim,
        enz_edge_dim,
        hidden_dim=64,
        edge_dim=256,
        out_dim=512,
        num_layers=3,
        heads=4,
        activation="silu",
        norm="layer",
        dropout=0.1,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        # input, embedding use normlization
        self.plm_proj = Dense(enz_node_dim[0], hidden_dim)
        self.sym_proj = nn.Embedding(enz_node_dim[1], hidden_dim)  # no sym features, use learnable embedding
        self.phy_proj = Dense(enz_node_dim[2], hidden_dim // 4, activation, norm, dropout=dropout, pre_norm=False)
        self.fuse_proj = Dense(hidden_dim, hidden_dim, activation, norm, dropout=dropout)

        self.edge_proj = Dense(enz_edge_dim, edge_dim, activation, norm, dropout=dropout, pre_norm=False)
        self.cls_proj = Dense(enz_node_dim[0], hidden_dim, activation, norm, dropout=dropout, pre_norm=False)

        # graph conv layers
        self.poc_layers = nn.ModuleList(
            [GraphEncoder(hidden_dim, edge_dim, heads, activation, norm, dropout, last_layer=(i == num_layers - 1)) for i in range(num_layers)]
        )
        self.out_proj = MLP([hidden_dim, hidden_dim * 2, out_dim], activation, norm, dropout=dropout)

    def forward(self, poc_graph):  # graph
        # poc_graph = Data(x=plm_x, sym_x=sym_x, phy_x=phy_x, edge_index=edge_index, edge_attr=edge_attr, cls=cls_feat)
        # plm_x:960; sym_x:0;  phy_x:29; edge_attr:82; cls_feat:960
        # 4+1+1
        # unpack x, edge_index, edge_attr, batch, cls
        edge_index, batch = poc_graph.edge_index, poc_graph.batch
        # x proj
        plm_x = self.plm_proj(poc_graph.x)  # [n, hidden_dim//2]
        sym_x = self.sym_proj(poc_graph.sym_x)  # [n, hidden_dim//4]
        phy_x = self.phy_proj(poc_graph.phy_x)  # [n, hidden_dim//4]
        x = self.fuse_proj(torch.cat([plm_x, sym_x, phy_x], dim=-1))  # [n, hidden_dim]
        # other proj
        edge_attr = self.edge_proj(poc_graph.edge_attr)  # [e, edge_dim]
        cls = self.cls_proj(poc_graph.cls)  # [b, hidden_dim]
        # graph conv layers
        for poc_layer in self.poc_layers:
            x, edge_attr, cls = poc_layer(x, edge_index, edge_attr, batch, cls)
        # out
        return self.out_proj(cls)  # [b, out_dim]


class RXNEnc(nn.Module):

    def __init__(
        self,
        rxn_node_dim,
        rxn_edge_dim,
        hidden_dim=64,
        edge_dim=256,
        out_dim=512,
        num_layers=3,
        heads=4,
        activation="silu",
        norm="layer",
        dropout=0.1,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        # input, embedding use normlization
        self.plm_proj = Dense(rxn_node_dim[0], hidden_dim, activation, norm, dropout=dropout, pre_norm=False)
        self.sym_proj = nn.Embedding(rxn_node_dim[1], hidden_dim)  # no sym features, use learnable embedding
        self.edge_proj = Dense(rxn_edge_dim, edge_dim, activation, norm, dropout=dropout, pre_norm=False)  # one-hot edge features, 不使用norm
        self.drfp_proj = Dense(2048, hidden_dim, activation, norm, dropout=dropout, pre_norm=False)
        self.rxnfp_proj = Dense(256, hidden_dim, activation, norm, dropout=dropout, pre_norm=False)
        self.mol_proj = Dense(rxn_node_dim[0], hidden_dim, activation, norm, dropout=dropout, pre_norm=False)
        # graph conv layers
        self.rxn_layers = nn.ModuleList(
            [GraphEncoder(hidden_dim, edge_dim, heads, activation, norm, dropout, last_layer=i == num_layers - 1) for i in range(num_layers)]
        )
        self.pool = AttentionalAggregation(gate_nn=Linear(hidden_dim, 1))
        self.out_proj = MLP([hidden_dim, hidden_dim * 2, out_dim], activation, norm, dropout=dropout)

    def forward(self, rxn_graph):
        # rxn_graph_data = Data(x=plm_x, sym_x=sym_x, edge_index=edge_index, edge_attr=edge_attr, drfp=drfp_feat, rxnfp=rxnfp_feat, mol_cls=mol_cls)
        # x:512;sym_x:0, edge_attr:22; drfp:2048; rxnfp:256; mol_cls:512
        # 2+3+1
        # unpack
        edge_index, batch = rxn_graph.edge_index, rxn_graph.batch
        # proj
        x = self.plm_proj(rxn_graph.x) + self.sym_proj(rxn_graph.sym_x)  # [n, hidden_dim]
        # other proj
        edge_attr = self.edge_proj(rxn_graph.edge_attr)  # [e, edge_dim]
        cls = self.drfp_proj(rxn_graph.drfp) + self.rxnfp_proj(rxn_graph.rxnfp)
        # graph conv layers
        for rxn_layer in self.rxn_layers:
            x, edge_attr, cls = rxn_layer(x, edge_index, edge_attr, batch, cls)
        # pool
        mol_cls = self.mol_proj(rxn_graph.mol_cls)
        mol = self.pool(mol_cls, rxn_graph.mol_cls_batch)
        # out
        return self.out_proj(mol + cls)  # [b, out_dim]


#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
############################# Ranking Model #############################
class Ranking(nn.Module):
    def __init__(self, hidden_dim=64, out_dim=1, activation="silu", norm="layer", dropout=0.1, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_proj = MLP([hidden_dim * 2, hidden_dim, out_dim], activation, norm, dropout=dropout)

    def forward(self, poc_emb, rxn_emb):
        # poc_emb: [b, hidden_dim], rxn_emb: [b, hidden_dim]
        x = torch.cat([poc_emb, rxn_emb], dim=-1)  # [b, hidden_dim*2]
        return self.out_proj(x)  # [b, out_dim]
