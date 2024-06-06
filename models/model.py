import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import squeeze
from models.encoder import Residual_connetion,LayerNorm,res_encoder,res_decoder
from models.RevIN import RevIN
from models.graphadj import adj_emb

class GLU(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(GLU, self).__init__()
        self.linear_left = nn.Linear(input_channel, output_channel)
        self.linear_right = nn.Linear(input_channel, output_channel)

    def forward(self, x):
        return torch.mul(self.linear_left(x), torch.sigmoid(self.linear_right(x)))


class StockBlockLayer(nn.Module):
    def __init__(self, args, time_step, unit, multi_layer, stack_cnt=1, use_nnorm=True, bias=True):
        super(StockBlockLayer, self).__init__()
        self.prop_feature = []
        self.k_k = 3
        self.gpr_weights = nn.Parameter(torch.zeros(self.k_k))
        self.gpr_weights.data[2] = 1.0
        self.time_step = time_step
        self.unit = unit
        self.horizon = args.horizon
        self.seq_len = args.window_size
        self.label_len = args.labellen
        self.hidden_dim=args.d_model
        self.res_hidden=args.d_model
        self.encoder_num=args.e_layers
        self.decoder_num=args.d_layers
        self.stack_cnt = stack_cnt
        self.feature_dim= 4
        self.freq=4
        self.feature_encode_dim=2
        self.decode_dim = args.c_out
        self.timeDecoderHidden = args.d_ff
        self.dropout = args.dropout
        self.res_dim = self.seq_len + (self.seq_len + self.horizon) * self.feature_encode_dim
        self.multi = multi_layer
        self.weight = nn.Parameter(
            torch.Tensor(1, 3 + 1, 1, self.time_step * self.multi,
                         self.multi * self.time_step))
        nn.init.kaiming_normal_(self.weight)
        self.forecast = nn.Linear(self.time_step * self.multi, self.time_step * self.multi)
        self.forecast_result = nn.Linear(self.time_step * self.multi, self.horizon)
        self.relu = nn.ReLU()
        self.GLUs = nn.ModuleList()
        self.output_channel = 4 * self.multi
        self.skip = time_step
        self.skipgru = nn.GRU(self.skip, self.time_step*10, batch_first=True, num_layers=2)
        self.slinear = nn.Linear(self.time_step*10, self.time_step)
        self.gru = nn.LSTM(self.time_step, self.time_step*4, batch_first=True, num_layers=2)
        self.glinear = nn.Linear(self.time_step*4, self.time_step)
        self.residual_proj = nn.Linear(self.seq_len, self.horizon, bias=bias)
        self.use_nnorm=use_nnorm
        self.feature_encoder = Residual_connetion(self.feature_dim, self.res_hidden, self.feature_encode_dim,self.dropout, bias)
        self.res_encoder = res_encoder(self.res_dim, self.res_hidden, self.hidden_dim, self.encoder_num,self.dropout, bias)
        self.res_decoder = res_decoder(self.hidden_dim, self.res_hidden, self.decoder_num,self.decode_dim,self.horizon,self.dropout, bias)
        self.timeDecoder = Residual_connetion(self.decode_dim + self.feature_encode_dim, self.timeDecoderHidden, 1, self.dropout, bias)
        if self.use_nnorm == True:
            self.nnorm= RevIN(self.unit)
        for i in range(3):
            if i == 0:
                self.GLUs.append(GLU(self.time_step * 4, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * 4, self.time_step * self.output_channel))
            elif i == 1:
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
            else:
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
    def spectral_conv(self, input):
        batch_size, k, input_channel, node_cnt, time_step = input.size()
        input = input.view(batch_size, -1, node_cnt, time_step)
        ffted = torch.rfft(input, 1, onesided=False)
        real = ffted[..., 0].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
        img = ffted[..., 1].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
        for i in range(3):
            real = self.GLUs[i * 2](real)
            img = self.GLUs[2 * i + 1](img)
        real = real.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
        img = img.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
        time_step_as_inner = torch.cat([real.unsqueeze(-1), img.unsqueeze(-1)], dim=-1)
        iffted = torch.irfft(time_step_as_inner, 1, onesided=False)
        return iffted

    def res_connect(self,x_r, batch_y_mark):
        skip_residual_con_list=[]
        for feature in range(x_r.shape[-1]):
            x_enc = x_r[:, :, feature]
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
            feature = self.feature_encoder(batch_y_mark)
            res_hidden = self.res_encoder(torch.cat([x_enc, feature.reshape(feature.shape[0], -1)], dim=-1))
            decoded = self.res_decoder(res_hidden).reshape(res_hidden.shape[0], self.horizon, self.decode_dim)
            skip_residual_con_feature= self.timeDecoder(torch.cat([feature[:, self.seq_len:], decoded], dim=-1)).squeeze(
                -1) + self.residual_proj(x_enc)
            skip_residual_con_feature= skip_residual_con_feature * (stdev[:, 0].unsqueeze(1).repeat(1, self.horizon))
            skip_residual_con_feature = skip_residual_con_feature + (means[:, 0].unsqueeze(1).repeat(1, self.horizon))
            skip_residual_con_list.append(skip_residual_con_feature)
        skip_residual_con = torch.stack(skip_residual_con_list, dim=-1)
        return skip_residual_con

    def forward(self, x, x_r,batch_y_mark,mul_L, attention,norm_mean=None, norm_stdev=None):
        hw = x.squeeze(1)
        hwo, _ = self.gru(hw)
        s, _ = self.skipgru(x.squeeze(1)[:, :, -self.skip:].contiguous())
        s = self.slinear(s)
        gfted = torch.matmul(mul_L.unsqueeze(1), x.unsqueeze(1))
        igfted = torch.sum(torch.matmul(self.spectral_conv(gfted).unsqueeze(2), self.weight), dim=1)
        skip_residual_con = self.res_connect(x_r, batch_y_mark).permute(0, 2, 1)
        forecast_source = torch.sigmoid(self.forecast(igfted).squeeze(1))
        forecast = self.forecast_result(forecast_source) + skip_residual_con
        return forecast


class Model(nn.Module):
    def __init__(self, args, units, stack_cnt, time_step, multi_layer, horizon=1, dropout_rate=0.5, leaky_rate=0.2,
                 device='cpu',use_nnorm=True,bias=True):
        super(Model, self).__init__()
        self.pri_adj = nn.Parameter(torch.zeros(size=(units, units)))
        nn.init.kaiming_normal_(self.pri_adj, mode='fan_in', nonlinearity='leaky_relu')

        self.unit = units
        self.stack_cnt = stack_cnt
        self.alpha = leaky_rate
        self.time_step = time_step
        self.horizon = args.horizon
        self.weight_key = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
        self.GRU = nn.GRU(self.time_step, self.unit)
        self.multi_layer = multi_layer
        self.stock_block = nn.ModuleList()
        self.stock_block.extend(
            [StockBlockLayer(args,self.time_step, self.unit, self.multi_layer, stack_cnt=i, use_nnorm= use_nnorm) for i in range(self.stack_cnt)])
        self.fc = nn.Sequential(
            nn.Linear(int(self.time_step), int(self.time_step)),
            nn.LeakyReLU(),
            nn.Linear(int(self.time_step), self.horizon),
        )
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.x_info= adj_emb(time_step)
        self.to(device)
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
        N = laplacian.size(0)  # [N, N]
        laplacian = laplacian.unsqueeze(0)
        first_laplacian = torch.zeros([1, N, N], device=laplacian.device, dtype=torch.float)
        second_laplacian = laplacian
        third_laplacian = (2 * torch.matmul(laplacian, second_laplacian)) - first_laplacian
        forth_laplacian = 2 * torch.matmul(laplacian, third_laplacian) - second_laplacian
        multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=0)
        return multi_order_laplacian
    def prob_adj(self, x):
        if self.use_norm == True:
            x ,norm_mean, norm_stdev = self.nnorm(x, 'norm')
            enc_out = self.x_info(x[0])[0]
        else:
            enc_out = self.x_info(x)[0]
        inl = enc_out.shape[1]
        self.stack_cnt
        layer1 = nn.Linear(inl, self.unit, bias=True).to(device=0)
        nn.init.kaiming_normal_(layer1.weight, mode='fan_in',nonlinearity='leaky_relu')
        layer2 = nn.Linear(512, self.unit, bias=True).to(device=0)
        nn.init.kaiming_normal_(layer2.weight, mode='fan_in', nonlinearity='leaky_relu')
        enc_out1 = enc_out.permute(0, 2, 1)
        enc_out1 = layer1(enc_out1).permute(0, 2, 1)
        xp = layer2(enc_out1)
        attention = torch.mean(xp, dim=0)
        attention = attention + self.pri_adj
        attention = torch.abs(attention)
        degree = torch.sum(attention, dim=1)
        degree_l = torch.diag(degree)
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))
        laplacian = torch.matmul(diagonal_degree_hat,
                                 torch.matmul(degree_l - attention, diagonal_degree_hat))
        mul_L = self.cheb_polynomial(laplacian)
        if self.use_norm == True:
            return mul_L, attention,norm_mean,norm_stdev
        else:
            return mul_L, attention

    def self_graph_attention(self, input):

        input = input.permute(0, 2, 1).contiguous()
        bat, N, fea = input.size()
        key = torch.matmul(input, self.weight_key)
        query = torch.matmul(input, self.weight_query)
        data = key.repeat(1, 1, N).view(bat, N * N, 1) + query.repeat(1, N, 1)
        data = data.squeeze(2)
        data = data.view(bat, N, -1)
        data = self.leakyrelu(data)

        attention = F.softmax(data, dim=2)
        attention = self.dropout(attention)
        return attention

    def graph_fft(self, input, eigenvectors):
        return torch.matmul(eigenvectors, input)

    def forward(self, x, x_mark_enc, x_dec, batch_y_mark, mask=None):
        if self.use_norm==True:
            mul_L, attention,norm_mean,norm_stdev = self.prob_adj(x)
        else:
            mul_L, attention = self.prob_adj(x)
        x_g = x.unsqueeze(1).permute(0, 1, 3, 2).contiguous()
        batch_y_mark = torch.cat([x_mark_enc, batch_y_mark[:, -self.horizon:, :]], dim=1)
        if self.use_norm==True:
            result = self.stock_block[0](x_g, x, batch_y_mark, mul_L, attention, norm_mean, norm_stdev).permute(0, 2, 1)
        else:
            result = self.stock_block[0](x_g, x, batch_y_mark, mul_L, attention).permute(0, 2, 1)
        if self.use_norm == True:
            result = self.nnorm(result, 'denorm', norm_mean, norm_stdev)[0]
        return result, attention

