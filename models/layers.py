import rootutils

root_path = str(rootutils.setup_root(__file__, indicator=".root", pythonpath=True))
from models.modules import *
from utils import *
from models.dataloader import RHEADataset, DataLoader
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, SAGPooling, global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation

class PocEnc(nn.Module):

    def __init__(self, hidden_dim=64, node_dim=1324, out_dim=512, num_layers=3, heads=4, activation="silu", norm="layer", dropout=0.1, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        hidden_dims = [int(x) for x in np.linspace(hidden_dim, out_dim, num_layers + 1)]
        # input
        self.node_dense = Dense(node_dim, hidden_dim, activation, dropout=dropout)  # 不使用norm
        self.edge_dense = Dense(434, hidden_dim, activation, dropout=dropout)  # 不使用norm

        self.GAT_conv = nn.ModuleList()
        self.pool = nn.ModuleList()
        self.graph_mlp = nn.ModuleList()
        self.relu = nn.ModuleList()
        for i in range(num_layers):
            self.GAT_conv.append(
                GATv2Conv(hidden_dims[i], hidden_dims[i + 1], heads, edge_dim=hidden_dim, concat=False, dropout=dropout, add_self_loops=False)
            )
            self.graph_mlp.append(Dense(hidden_dims[i + 1], hidden_dims[-1], activation, norm, dropout))
            if i < (num_layers - 1):
                self.relu.append(nn.LeakyReLU(1e-2))
                self.pool.append(SAGPooling(in_channels=hidden_dims[i + 1], ratio=0.5))
        self.fc_mlp = Dense(hidden_dims[-1], out_dim, activation, dropout=dropout)

    def forward(self, poc_graph):  # graph
        # x, edge_index, edge_attr
        x, edge_index, edge_attr, batch = poc_graph.x, poc_graph.edge_index, poc_graph.edge_attr, poc_graph.batch
        x = self.node_dense(x)
        edge_attr = self.edge_dense(edge_attr)
        graph_out = 0.0
        # graph conv layers
        for i in range(len(self.GAT_conv)):
            x = self.GAT_conv[i](x, edge_index, edge_attr)
            pooled_x = global_mean_pool(x, batch)
            graph_out = graph_out + self.graph_mlp[i](pooled_x)

            if i < (len(self.GAT_conv) - 1):
                x = self.relu[i](x)
                x, edge_index, edge_attr, batch, _, _ = self.pool[i](x, edge_index, edge_attr, batch)
        # output
        graph_out = self.fc_mlp(graph_out)
        return graph_out  # [x, hidden_dim]


class RXNEnc(nn.Module):

    def __init__(self, hidden_dim=64, out_dim=512, num_layers=3, heads=4, activation="silu", norm="layer", dropout=0.1, **kwargs):
        super().__init__()

        hidden_dims = [int(x) for x in np.linspace(hidden_dim, out_dim, num_layers + 1)]

        self.atom_embed = nn.Embedding(119, hidden_dim)
        self.edge_dense = Dense(18, hidden_dim, dropout=dropout)  # 不使用norm
        self.drfp_dense = Dense(2048, out_dim)  # 不使用norm
        self.seq_dense = Dense(768, hidden_dims[-1], activation, norm, dropout)  # 不使用norm
        self.seq_pool = AttentionalAggregation(gate_nn=Dense(hidden_dims[-1], out_dim, activation, norm, dropout))

        self.GAT_conv = nn.ModuleList()
        self.pool = nn.ModuleList()
        self.graph_mlp = nn.ModuleList()
        self.relu = nn.ModuleList()
        for i in range(num_layers):
            self.GAT_conv.append(
                GATv2Conv(hidden_dims[i], hidden_dims[i + 1], heads, edge_dim=hidden_dim, concat=False, dropout=dropout, add_self_loops=False)
            )
            self.graph_mlp.append(Dense(hidden_dims[i + 1], hidden_dims[-1], activation, norm, dropout))
            if i < (num_layers - 1):
                self.relu.append(nn.LeakyReLU(1e-2))
                self.pool.append(SAGPooling(in_channels=hidden_dims[i + 1], ratio=0.5))

        self.fc_mlp = Dense(hidden_dims[-1], out_dim, activation, dropout=dropout)

    def forward(self, rxn_graph):
        # x edge_index edge_attr drfp seq
        x, edge_index, edge_attr, drfp, seq, batch = rxn_graph.x, rxn_graph.edge_index, rxn_graph.edge_attr, rxn_graph.drfp, rxn_graph.seq, rxn_graph.batch
        x = self.atom_embed(x)  # [n, hidden_dim]
        edge_attr = self.edge_dense(edge_attr)  # [e, hidden_dim]
        graph_out = 0.0

        for i in range(len(self.GAT_conv)):
            x = self.GAT_conv[i](x, edge_index, edge_attr)
            pooled_x = global_mean_pool(x, batch)
            graph_out = graph_out + self.graph_mlp[i](pooled_x)

            if i < (len(self.GAT_conv) - 1):
                x = self.relu[i](x)
                x, edge_index, edge_attr, batch, _, _ = self.pool[i](x, edge_index, edge_attr, batch)

        seq = self.seq_dense(seq)  # [n, hidden_dim]
        seq_out = self.seq_pool(seq, rxn_graph.seq_batch)  # [b, hidden_dim]
        drfp_out = self.drfp_dense(drfp)  # [b, hidden_dim]
        out = graph_out + seq_out + drfp_out
        out = self.fc_mlp(out)
        return out  # [x, hidden_dim]


def test_step():
    # 初始化
    from torch.optim.adamw import AdamW
    poc_model = PocEnc()
    rxn_model = RXNEnc()
    rxn_model.train()
    optimizer = AdamW(rxn_model.parameters(), lr=1e-4)
    criterion = nn.TripletMarginLoss(margin=0.5, p=2)
    feat_dir = f"{root_path}/data/features"
    pairs = pkl_load(f"{root_path}/data/training/rhea_valid_pair.pkl")
    dataset = RHEADataset(feat_dir, pairs)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, follow_batch=["seq"])

    for batch in dataloader:
        optimizer.zero_grad()
        poc, pos_rxn, neg_pos = batch
        poc_out = poc_model(poc)
        pos_out = rxn_model(pos_rxn)
        neg_out = rxn_model(neg_pos)
        loss = criterion(poc_out, pos_out, neg_out)
        loss.backward()
        # --- 🔍 核心功能：查找未使用的参数 ---
        print("------- 🕵️ 正在侦探未使用的参数 -------")
        unused_count = 0
        for name, param in rxn_model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    print(f"🔴 [未使用] {name} -> 梯度为 None")
                    unused_count += 1

        if unused_count == 0:
            print("\n🎉 完美！所有参数都参与了计算。")
        else:
            print(f"\n⚠️ 警告：发现了 {unused_count} 个未使用的参数。这就是 DDP 报错的原因。")
        break


if __name__ == "__main__":
    # Example usage
    test_step()
    # data = pkl_load("/data/zzjun/PLA/datasets/_feat/pdbbind-2020/10gs.pkl")
    # print(data)
