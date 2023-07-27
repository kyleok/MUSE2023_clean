import torch.nn as nn
from config import ACTIVATION_FUNCTIONS, device
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch
from config import ACTIVATION_FUNCTIONS
from sub_modules import GatedResidualNetwork, VariableSelectionNetwork, GateAddNormNetwork, \
    InterpretableMultiHeadAttention
import math


class RNN(nn.Module):
    def __init__(self, d_in, d_out, n_layers=1, bi=True, dropout=0.2, n_to_1=False):
        super(RNN, self).__init__()
        self.rnn = nn.GRU(input_size=d_in, hidden_size=d_out, bidirectional=bi, num_layers=n_layers, dropout=dropout)
        self.n_layers = n_layers
        self.d_out = d_out
        self.n_directions = 2 if bi else 1
        self.n_to_1 = n_to_1

    def forward(self, x, x_len):
        x_packed = pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=False)
        rnn_enc = self.rnn(x_packed)

        if self.n_to_1:
            # hiddenstates, h_n, only last layer
            return last_item_from_packed(rnn_enc[0], x_len)
            # batch_size = x.shape[0]
            # h_n = h_n.view(self.n_layers, self.n_directions, batch_size, self.d_out) # (NL, ND, BS, dim)
            # last_layer = h_n[-1].permute(1,0,2) # (BS, ND, dim)
            # x_out = last_layer.reshape(batch_size, self.n_directions * self.d_out) # (BS, ND*dim)

        else:
            x_out = rnn_enc[0]
            x_out = pad_packed_sequence(x_out, total_length=x.size(1), batch_first=True)[0]

        return x_out


# https://discuss.pytorch.org/t/get-each-sequences-last-item-from-packed-sequence/41118/7
def last_item_from_packed(packed, lengths):
    sum_batch_sizes = torch.cat((
        torch.zeros(2, dtype=torch.int64),
        torch.cumsum(packed.batch_sizes, 0)
    )).to(device)
    sorted_lengths = lengths[packed.sorted_indices].to(device)
    last_seq_idxs = sum_batch_sizes[sorted_lengths] + torch.arange(lengths.size(0)).to(device)
    last_seq_items = packed.data[last_seq_idxs]
    last_seq_items = last_seq_items[packed.unsorted_indices]
    return last_seq_items


class OutLayer(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout=.0, bias=.0):
        super(OutLayer, self).__init__()
        self.fc_1 = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(True), nn.Dropout(dropout))
        self.fc_2 = nn.Linear(d_hidden, d_out)
        nn.init.constant_(self.fc_2.bias.data, bias)

    def forward(self, x):
        y = self.fc_2(self.fc_1(x))
        return y


class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
        self.params = params

        self.inp = nn.Linear(params.d_in, params.model_dim, bias=False)

        self.encoder = RNN(params.model_dim, params.model_dim, n_layers=params.rnn_n_layers, bi=params.rnn_bi,
                           dropout=params.rnn_dropout, n_to_1=params.n_to_1)

        d_rnn_out = params.model_dim * 2 if params.rnn_bi and params.rnn_n_layers > 0 else params.model_dim
        self.out = OutLayer(d_rnn_out, params.d_fc_out, params.n_targets, dropout=params.linear_dropout)
        self.final_activation = ACTIVATION_FUNCTIONS[params.task]()

    def forward(self, x, x_len):
        x = self.inp(x)
        x = self.encoder(x, x_len)
        y = self.out(x)
        activation = self.final_activation(y)
        return activation

    def set_n_to_1(self, n_to_1):
        self.encoder.n_to_1 = n_to_1


class TFT_encoder(nn.Module):
    def __init__(self, params):
        super(TFT_encoder, self).__init__()
        self.hparams = params

        print("params.d_in", params.d_in)  # 입력차원 faus =20 , egemaps = 88
        print("params.d_rnn(model_dim)", params.model_dim)  # rnn hidden layer의 차원
        print("params.rnn_n_layers", params.rnn_n_layers)  # rnn의 수
        print("params.rnn_dropout", params.rnn_dropout)  # drop out rate

        self.inp = nn.Linear(params.d_in, params.model_dim, bias=False)

        if params.rnn_n_layers > 0:
            self.rnn = RNN(params.model_dim, params.model_dim, n_layers=params.rnn_n_layers, bi=params.rnn_bi,
                           dropout=params.rnn_dropout, n_to_1=params.n_to_1)

        # code.interact(local=dict(globals(), **locals()))
        d_rnn_out = params.model_dim * 2 if params.rnn_bi and params.rnn_n_layers > 0 else params.model_dim

        self.regular_var_embeddings = nn.ModuleList([nn.Linear(1, params.model_dim) for i in range(params.d_in)])

        self.vsn = VariableSelectionNetwork(hidden_layer_size=params.model_dim,
                                            input_size=params.d_in * params.model_dim,
                                            output_size=params.d_in,
                                            dropout_rate=params.rnn_dropout)

        self.state_h_grn = GatedResidualNetwork(params.model_dim, dropout_rate=params.rnn_dropout,
                                                output_size=params.model_dim)
        self.state_c_grn = GatedResidualNetwork(params.model_dim, dropout_rate=params.rnn_dropout,
                                                output_size=params.model_dim)

        self.post_seq_encoder_gate_add_norm = GateAddNormNetwork(params.model_dim,
                                                                 params.model_dim,
                                                                 params.rnn_dropout,
                                                                 activation=None)

        self.self_attn_layer = InterpretableMultiHeadAttention(n_head=params.rnn_n_layers,
                                                               d_model=params.model_dim,
                                                               dropout=params.rnn_dropout)

        self.lstm = nn.LSTM(input_size=params.model_dim, hidden_size=params.model_dim, batch_first=True)

        self.final_activation = ACTIVATION_FUNCTIONS[params.task]()
        ## Initializing remaining weights
        self.out = OutLayer(d_rnn_out, params.d_fc_out, params.n_targets, dropout=params.linear_dropout)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if ('lstm' in name and 'ih' in name) and 'bias' not in name:
                # print(name)
                # print(p.shape)
                torch.nn.init.xavier_uniform_(p)
            #                torch.nn.init.kaiming_normal_(p, a=0, mode='fan_in', nonlinearity='sigmoid')
            elif ('lstm' in name and 'hh' in name) and 'bias' not in name:

                torch.nn.init.orthogonal_(p)

            elif 'lstm' in name and 'bias' in name:
                # print(name)
                # print(p.shape)
                torch.nn.init.zeros_(p)

    def get_tft_embeddings(self, regular_inputs):
        # Static input

        # static_regular_inputs = [self.regular_var_embeddings[i](regular_inputs[:, 0, i:i + 1])
        #                        for i in range(self.hparams.d_in)]

        static_regular_inputs = [self.regular_var_embeddings[i](regular_inputs[:, :, i:i + 1]) for i in
                                 range(self.hparams.d_in)]
        static_regular_inputs = torch.stack(static_regular_inputs, axis=-1)  # .transpose(-2,-1)
        return static_regular_inputs

    def forward(self, x, x_len, x_static=None):
        # code.interact(local=dict(globals(), **locals()))
        x = self.get_tft_embeddings(x)
        sparse_features, sparse_weights = self.vsn(x)

        state_h = self.state_h_grn(sparse_features).mean(axis=1)  # .sum(axis =1)
        state_c = self.state_c_grn(sparse_features).mean(axis=1)  # .sum(axis =1)
        # state_h 219(batch)*16(hidden)이어야 한다. -> forward에 static추가하고 이걸로 static variable state를 만들어야겠다.
        output_lstm, (state_h, state_c) = self.lstm(sparse_features, (state_h.unsqueeze(0), state_c.unsqueeze(0)))

        enriched = self.post_seq_encoder_gate_add_norm(output_lstm, sparse_features)
        x, self_att = self.self_attn_layer(enriched, enriched, enriched)  # , mask = self.get_decoder_mask(enriched))
        # -> self_att -> torch.Size([219, 200, 2, 200]) 왜 두개지?
        x = self.out(x)

        return self.final_activation(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TFModel(nn.Module):
    def __init__(self, params):
        super(TFModel, self).__init__()
        self.params = params
        self.device = torch.device("cuda")
        d_model = params.model_dim
        d_rnn_out = params.model_dim * 2 if params.rnn_bi and params.rnn_n_layers > 0 else params.model_dim

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=params.rnn_n_layers,
                                                        dropout=params.rnn_dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=int(params.rnn_n_layers / 2))
        self.pos_encoder = PositionalEncoding(params.model_dim, params.rnn_dropout)

        self.encoder = nn.Sequential(
            nn.Linear(params.d_in, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        self.out = OutLayer(d_rnn_out, params.d_fc_out, params.n_targets, dropout=params.linear_dropout)
        self.final_activation = ACTIVATION_FUNCTIONS[params.task]()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_len):
        # code.interact(local=dict(globals(), **locals()))
        srcmask = self.generate_square_subsequent_mask(src.shape[1]).to(self.device)
        src = self.encoder(src)
        # src = self.pos_encoder(src)#old version
        src = self.pos_encoder(src.transpose(0, 1))  # transpose before pos_encoder
        # output = self.transformer_encoder(src.transpose(0,1), srcmask).transpose(0,1)#original
        output = self.transformer_encoder(src, srcmask).transpose(0, 1)  # original
        # output = self.transformer_encoder(src).transpose(0,1)
        output = self.out(output)
        output = self.final_activation(output)

        # code.interact(local=dict(globals(), **locals()))
        return output
