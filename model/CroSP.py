import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss import crossview_contrastive_Loss, loss_metric

def gcn_norm(edge_index, add_self_loops=True):
    adj_t = edge_index.to_dense()
    if add_self_loops:
        adj_t = adj_t+torch.eye(*adj_t.shape, device=adj_t.device)
    deg = adj_t.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0.)

    adj_t.mul_(deg_inv_sqrt.view(-1, 1))
    adj_t.mul_(deg_inv_sqrt.view(1, -1))
    edge_index = adj_t.to_sparse()
    return edge_index, None

class GCNConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 cached: bool = True, add_self_loops: bool = False,
                 bias: bool = True, **kwargs):
        super(GCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.cached = cached
        self.add_self_loops = add_self_loops

        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, edge_index):
        if self.cached:
            if not hasattr(self, "cached_adj"):
                edge_index, edge_weight = gcn_norm(
                    edge_index, self.add_self_loops)
                self.register_buffer("cached_adj", edge_index)
            edge_index = self.cached_adj
        else:
            edge_index, _ = gcn_norm(edge_index, self.add_self_loops)
        x = torch.matmul(x, self.weight)
        out = edge_index@x
        if self.bias is not None:
            out += self.bias
        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class GraphAutoencoder(nn.Module):
    def __init__(self,  input_dim,  latent_dim,config):
        super(GraphAutoencoder, self).__init__()
        self.encoder = GCNEncoder(input_dim, latent_dim,config)
        self.decoder = GCNDecoder(latent_dim, input_dim)
        self.dc = InnerProductDecoder()

    def forward(self, x, adj):
        # 编码
        z = self.encoder(x, adj)
        # 解码
        x_hat = self.decoder(z, adj)
        return z, x_hat


class GCNEncoder(nn.Module):
    def __init__(self,  input_dim, embedding_dim,config, num_layers=4):
        super(GCNEncoder, self).__init__()
        self.d_prior = embedding_dim[0] // int(config.d_prior)
        self.enc_fc1 = nn.Linear(input_dim, self.d_prior, bias=False)
        self.enc_fc2 = nn.Linear(input_dim, embedding_dim[0] - self.d_prior, bias=False)
        in_channels_list = [embedding_dim[0], embedding_dim[1], embedding_dim[2], embedding_dim[3]]
        self.gnn_layers = nn.ModuleList([GCNConv(in_channels=in_channels_list[i], out_channels=embedding_dim[i+1]) for i in range(num_layers)])

    def forward(self, embedding, edge):
        if not hasattr(self, "edge_index"):
            edge_index = torch.sparse_coo_tensor(*edge)
            self.register_buffer("edge_index", edge_index)
        edge_index = self.edge_index
        embedding = torch.cat([self.enc_fc1(embedding), self.enc_fc2(embedding)], -1)
        for i, layer in enumerate(self.gnn_layers):
            out = layer(embedding, edge_index)
            embedding = out
        # m0=nn.Softmax(dim=1)
        # embedding=m0(embedding)
        return embedding



class GCNDecoder(nn.Module):
    def __init__(self,  embedding_dim, output_dim, padding=True, num_layers=1):
        super(GCNDecoder, self).__init__()
        # 确定输入通道数和输出通道数的列表
        in_channels_list = [embedding_dim[4], embedding_dim[3], embedding_dim[2], embedding_dim[1]]
        out_channels_list = [embedding_dim[3], embedding_dim[2], embedding_dim[1], embedding_dim[0]]
        self.gnn_layers = nn.ModuleList([GCNConv(in_channels=in_channels_list[i], out_channels=out_channels_list[i]) for i in range(num_layers)])
        self.init_transform = nn.Linear(out_channels_list[num_layers-1], output_dim, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)

    def forward(self, z, edge):
        if not hasattr(self, "edge_index"):
            edge_index = torch.sparse_coo_tensor(*edge)
            self.register_buffer("edge_index", edge_index)
        edge_index = self.edge_index
        for i, layer in enumerate(self.gnn_layers):
            out = layer(z, edge_index)
            z = out
        z = self.init_transform(z)
        return z

class AdaptiveFusionLayer(nn.Module):
    def __init__(self):
        super(AdaptiveFusionLayer, self).__init__()
        self.gamma = nn.Parameter(torch.tensor(0.6))

    def forward(self, m1, m2):
        fused_representation = self.gamma * m1 + (1 - self.gamma) * m2
        return fused_representation

class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.act = act

    def forward(self, z):
        adj = self.act(torch.mm(z, z.t()))
        return adj


class Completer():

    def __init__(self, config, input_size):
        """Constructor.
        """
        self._config = config
        self.embedding_dim = list(map(int, config.emb_dim))
        self.spa_encoder = GraphAutoencoder(input_size, self.embedding_dim, config)
        self.img_encoder = GraphAutoencoder(input_size,  self.embedding_dim, config)
        self.fusion_model = AdaptiveFusionLayer()

    def to_device(self, device):
        """ to cuda if gpu is used """
        self.spa_encoder.to(device)
        self.img_encoder.to(device)
        self.fusion_model.to(device)

    def train(self, config, logger, x1_train, x2_train, x1_edge, x2_edge, metric_graph, optimizer):
        """Training the model.
            Args:
              logger: print the information.
              config : config of train
              x1_train: data of view 1
              x2_train: data of view 2
              x1_edge: graph of view 1
              x2_edge: graph of view 2
              metric_graph: for metric_loss
              optimizer: adam is used in our experiments
            Returns:
        """

        edge_index = torch.sparse_coo_tensor(*x1_edge)
        dense_tensor = edge_index.to_dense()
        edge_index2 = torch.sparse_coo_tensor(*x2_edge)
        dense_tensor2 = edge_index2.to_dense()
        loss_all_l, loss_rec1_l, loss_rec2_l, loss_cl_l, loss_metric_l = [], [], [], [], []
        for epoch in range(int(config.epoch)):
            z_1, x_hat1 = self.spa_encoder(x1_train, x1_edge)
            z_2, x_hat2 = self.img_encoder(x2_train, x2_edge)

            # Cross-view Contrastive_Loss
            cl_loss_inter = crossview_contrastive_Loss(z_1, z_2)

            # Metric Loss
            if config.data_type == 'Slide_seqV2' or config.data_type == 'Stereo_seq':
                pass
            else:
                metric1 = loss_metric(z_1, metric_graph)
                metric2 = loss_metric(z_2, metric_graph)
                metric = metric1 + metric2

            z_f = self.fusion_model(z_1, z_2)

            z_f_1 = self.spa_encoder.decoder(z_f, x1_edge)
            z_f_2 = self.img_encoder.decoder(z_f, x2_edge)
            z_hat1_edge = self.spa_encoder.dc(z_f)
            z_hat2_edge = self.img_encoder.dc(z_f)

            feature_loss1_z = F.mse_loss(z_f_1, x1_train)
            feature_loss2_z = F.mse_loss(z_f_2, x2_train)
            structure_loss1_z = F.mse_loss(z_hat1_edge, dense_tensor)
            structure_loss2_z = F.mse_loss(z_hat2_edge, dense_tensor2)
            recon1 = feature_loss1_z+feature_loss2_z
            recon2 = structure_loss1_z + structure_loss2_z

            if config.data_type == 'Slide_seqV2' or config.data_type == 'Stereo_seq':
                loss = cl_loss_inter * config.lambda1 + (recon1+recon2) * config.lambda2
            else:
                loss = cl_loss_inter * config.lambda1 + (recon1+recon2) * config.lambda2 + metric*config.lambda3

            optimizer.zero_grad()
            gradient_clipping = 5
            torch.nn.utils.clip_grad_norm_(self.spa_encoder.parameters(), gradient_clipping)
            torch.nn.utils.clip_grad_norm_(self.img_encoder.parameters(), gradient_clipping)
            torch.nn.utils.clip_grad_norm_(self.fusion_model.parameters(), gradient_clipping)
            loss.backward()
            optimizer.step()

            loss_all = loss.item()
            loss_rec1 = recon1.item()
            loss_rec2 = recon2.item()
            loss_cl = cl_loss_inter.item()

            loss_all_l.append(loss_all)
            loss_rec1_l.append(loss_rec1)
            loss_rec2_l.append(loss_rec2)
            loss_cl_l.append(loss_cl)

            if (epoch + 1) % config.print_num == 0:
                output = "Epoch : {:.0f}/{:.0f} ===> Reconstruction1 loss = {:.4f}===>Reconstruction2 loss = {:.4f}===> Contrastive loss = {:.4f} ===> Loss = {:.4f}" \
                    .format((epoch + 1), config.epoch, loss_rec1, loss_rec2, loss_cl, loss_all)
                logger.info("\033[2;29m" + output + "\033[0m")
        return z_f, x_hat1, x_hat2, z_f_1








