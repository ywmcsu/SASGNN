import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import squeeze
from models.encoder import Residual_connetion,LayerNorm,res_encoder,res_decoder
from models.RevIN import RevIN
from models.graphadj import adj_emb
from models.TCN import TCN
from models.SkipGRU import RecurrentSkipLayer
import torch



class GatingMechanism(nn.Module):
    def __init__(self, input_dim):
        super(GatingMechanism, self).__init__()
        self.gate = nn.Linear(3*input_dim, 3*input_dim)

    def forward(self, x1, x2, x3):
        concatenated = torch.cat((x1, x2, x3), dim=-1)
        gate_values = torch.sigmoid(self.gate(concatenated))
        gate_x1, gate_x2, gate_x3 = torch.chunk(gate_values, 3, dim=-1)
        gated_output = gate_x1 * x1 + gate_x2 * x2 + gate_x3 * x3
        return gated_output


class BlockLayer(nn.Module):
    def __init__(self, args, bias=True):
        super(BlockLayer, self).__init__()
        self.unit = args.node_cnt
        self.horizon = args.horizon
        self.time_step = args.window_size
        self.label_len = args.labellen
        self.hidden_dim = args.d_model
        self.res_hidden=args.d_model
        self.encoder_num=args.e_layers
        self.decoder_num=args.d_layers
        self.feature_dim= 4
        self.freq=4
        self.feature_encode_dim=2
        self.decode_dim = args.c_out
        self.timeDecoderHidden = args.d_ff
        self.dropout = args.dropout
        self.res_dim = self.time_step + (self.time_step + self.horizon) * self.feature_encode_dim
        self.forecast = nn.Linear(self.time_step*2, self.horizon)
        self.forecast_result = nn.Linear(self.time_step , self.horizon)
        self.drop = nn.Dropout(0.5)
        self.activation = nn.ReLU()
        self.GLUs = nn.ModuleList()
        self.num_channels = [64, 128, 256, 512]
        self.TCN= TCN(self.time_step, self.time_step, self.num_channels, 3, 0.4)
        self.gating_mechanism = GatingMechanism(self.time_step)
        self.skipgru = RecurrentSkipLayer(self.unit, self.time_step, self.time_step//2)
        self.residual_proj = nn.Linear(self.time_step, self.horizon)
        self.sepc_ln = nn.Linear(self.time_step, self.time_step)
        self.feature_encoder = Residual_connetion(self.feature_dim, self.res_hidden, self.feature_encode_dim,self.dropout, bias=True)
        self.res_encoder = res_encoder(self.res_dim, self.res_hidden, self.hidden_dim, self.encoder_num,self.dropout, bias=True)
        self.res_decoder = res_decoder(self.hidden_dim, self.res_hidden, self.decoder_num,self.decode_dim,self.horizon,self.dropout, bias=True)
        self.timeDecoder = Residual_connetion(self.decode_dim + self.feature_encode_dim, self.timeDecoderHidden, 1, self.dropout, bias=True)
        self.gconv_weights = nn.Parameter(torch.Tensor(4, self.unit, self.unit))
        nn.init.xavier_uniform_(self.gconv_weights)
        self.weight = nn.Parameter(
            torch.Tensor(1, self.unit, 1,self.time_step))
        nn.init.kaiming_normal_(self.weight)

    def spectral_conv(self, input):
        batch_size, time_step ,node_cnt= input.size()
        input = input.view(batch_size, -1, node_cnt, time_step)
        ffted = torch.rfft(input, 1, onesided=False)
        real = ffted[..., 0].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
        img = ffted[..., 1].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
        real = self.TCN(real)
        img = self.TCN(img)
        real = real.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
        img = img.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
        time_step_as_inner = torch.cat([real.unsqueeze(-1), img.unsqueeze(-1)], dim=-1)
        iffted = torch.irfft(time_step_as_inner, 1, onesided=False)
        iffted = self.sepc_ln(iffted.reshape(batch_size, node_cnt, -1)).contiguous()
        return iffted

    def weighted_graph_convolution(self, x, cheb_polynomials, num_layers):
        x = x.permute(0, 2, 1)  # (B, T, N) -> (B, N, T)
        B, N, T = x.size()

        for _ in range(num_layers):
            output = torch.zeros(B, N, T).to(x.device)
            for k in range(4):
                T_k = cheb_polynomials[k]  # (N, N)
                weighted_conv = torch.einsum('ij,bjt->bit', T_k, x)
                output += torch.einsum('ij,bti->bjt', self.gconv_weights[k], weighted_conv.transpose(1, 2))
            x = output

        return output

    def residual_connect(self,x, batch_y_mark):
        skip_residual_con_list=[]
        for feature in range(x.shape[-1]):
            x_enc = x[:, :, feature]
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
            feature = self.feature_encoder(batch_y_mark)
            res_hidden = self.res_encoder(torch.cat([x_enc, feature.reshape(feature.shape[0], -1)], dim=-1))
            decoded = self.res_decoder(res_hidden).reshape(res_hidden.shape[0], self.horizon, self.decode_dim)
            skip_residual_con_feature= self.timeDecoder(torch.cat([feature[:, self.time_step:], decoded], dim=-1)).squeeze(
                -1) + self.residual_proj(x_enc)
            skip_residual_con_feature= skip_residual_con_feature * (stdev[:, 0].unsqueeze(1).repeat(1, self.horizon))
            skip_residual_con_feature = skip_residual_con_feature + (means[:, 0].unsqueeze(1).repeat(1, self.horizon))
            skip_residual_con_list.append(skip_residual_con_feature)
        skip_residual_con = torch.stack(skip_residual_con_list, dim=-1)
        return skip_residual_con

    def forward(self, x, batch_y_mark, mul_L):
        spec_x = self.spectral_conv(x)
        ispec_x = torch.sum(torch.matmul(spec_x.unsqueeze(3), self.weight), dim=2)
        igfted = self.weighted_graph_convolution(x, mul_L, 4)
        forecast_s = self.skipgru(x)
        skip_residual_con = self.residual_connect(x, batch_y_mark).permute(0, 2, 1)
        ft = torch.cat((ispec_x, igfted), dim=-1)
        forecast = self.forecast(ft)+forecast_s + skip_residual_con
        return forecast


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.unit = args.node_cnt
        self.pri_adj = nn.Parameter(torch.zeros(size=(self.unit, self.unit)))
        nn.init.kaiming_normal_(self.pri_adj, mode='fan_in', nonlinearity='leaky_relu')
        self.time_step = args.window_size
        self.horizon = args.horizon
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.block = BlockLayer(args, bias=True)
        self.layer1 = nn.Linear(512, self.unit, bias=True).to(self.device)
        nn.init.kaiming_normal_(self.layer1.weight, mode='fan_in', nonlinearity='leaky_relu')
        self.x_info= adj_emb(self.time_step)
        self.use_norm = args.use_nnorm
        if self.use_norm == True:
            self.nnorm = RevIN(self.unit)

    def get_laplacian(self, graph, normalize):
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L

    def cheb_polynomial(self, laplacian):
        N = laplacian.size(0)
        laplacian = laplacian.unsqueeze(0)
        first_laplacian = torch.eye(N, device=laplacian.device, dtype=torch.float).unsqueeze(0)
        second_laplacian = laplacian
        third_laplacian = (2 * torch.matmul(laplacian, second_laplacian)) - first_laplacian
        forth_laplacian = 2 * torch.matmul(laplacian, third_laplacian) - second_laplacian
        multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=0)
        return multi_order_laplacian

    def prob_adj(self, x):
        enc_out = self.x_info(x)[0]
        xp = self.layer1(enc_out)
        attention = torch.mean(xp, dim=0)
        attention = torch.abs(attention + self.pri_adj)
        degree = torch.sum(attention, dim=1)
        degree_l = torch.diag(degree)
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))
        laplacian = torch.matmul(diagonal_degree_hat,
                                 torch.matmul(degree_l - attention, diagonal_degree_hat))
        mul_L = self.cheb_polynomial(laplacian)
        return mul_L

    def forward(self, x, x_mark_enc, x_dec, batch_y_mark, mask=None):
        mul_L= self.prob_adj(x)
        batch_y_mark = torch.cat([x_mark_enc, batch_y_mark[:, -self.horizon:, :]], dim=1)
        result = self.block(x, batch_y_mark, mul_L).permute(0, 2, 1)
        return result


