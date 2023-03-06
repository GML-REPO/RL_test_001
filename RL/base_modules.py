import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


### base module
def SinosoidalPositionEmbedding(n_seq, emb_size):
    def cal_angle(position, _i):
        return position / np.power(10000, 2*(_i//2) / emb_size)
    def angle_vector(pos):
        return [cal_angle(pos, _i) for _i in range(emb_size)]
    
    table = np.array([angle_vector(_i) for _i in range(n_seq)])
    table[:,0::2] = np.sin(table[:,0::2])
    table[:,1::2] = np.cos(table[:,1::2])
    return table

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        """
        sin, cos encoding 구현
        parameter
        - d_model : model의 차원
        - max_len : 최대 seaquence 길이
        - device : cuda or cpu
        """
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False
        
        pos = torch.arange(0, max_len, device =device)
        pos = pos.float().unsqueeze(dim=1)
        
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        
        self.encoding[:, ::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
    def forward(self, x):
        sh = x.size() 
        return self.encoding[:sh[1], :]
    
class ResidualLayer(nn.Module):
    def __init__(self, layer, with_normN=0) -> None:
        super().__init__()
        self.layer = layer
        # self.norm = nn.LayerNorm()
    def forward(self, x):
        return x + self.layer(x)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, n_head=1) -> None:
        super().__init__()
        self.layer_Q = nn.Linear(emb_dim, emb_dim)
        self.layer_K = nn.Linear(emb_dim, emb_dim)
        self.layer_V = nn.Linear(emb_dim, emb_dim)
        self.scale = np.sqrt(emb_dim)
        self.out = nn.Linear(emb_dim, emb_dim)

        assert emb_dim % n_head == 0, f'n_head is not available:{n_head} with {emb_dim}'
        self.dim_per_head = emb_dim // n_head
        self.n_head = n_head

    def forward(self, a, b=None):
        if b is None: b = a
        Q = self.layer_Q(a)
        K = self.layer_K(b)
        V = self.layer_V(b)
        if len(a.size()) == 2:
            Q = Q.unsqueeze(1)
        if len(b.size()) == 2:
            K = K.unsqueeze(1)
            V = V.unsqueeze(1)
        batch_size, seq_len_a, _ = Q.size()
        _, seq_len_b, _ = V.size()

        Q = Q.reshape([batch_size, seq_len_a, self.n_head, self.dim_per_head])
        K = K.reshape([batch_size, seq_len_b, self.n_head, self.dim_per_head])
        V = V.reshape([batch_size, seq_len_b, self.n_head, self.dim_per_head])

        Q = torch.einsum('N A H D -> N H A D', Q)
        K = torch.einsum('N B H D -> N H B D', K)
        V = torch.einsum('N B H D -> N H B D', V)

        score = torch.einsum('N H A D, N H B D -> N H A B', Q, K) / self.scale
        score = torch.softmax(score, dim=3)

        result = torch.einsum('N H A B, N H B D -> N H A D', score, V)
        result = torch.einsum('N H A D -> N A H D', result)
        result = result.reshape([batch_size, seq_len_a, self.n_head*self.dim_per_head]) 
        # if len(a.size()) == 2:
        #     result = result.squeeze(1)
        result = self.out(result)
        return result

class AF_block(nn.Module):
    def __init__(self, _in, _out) -> None:
        super().__init__()

        _out2 = _out // 2

        self.layer0 = nn.Conv1d(_in, _out2, 1)
        self.layer1 = nn.Conv1d(_in, _out2, 3, 1, 1, 1)
        self.layer2 = nn.Conv1d(_in, _out2, 3, 1, 3, 3)
        self.layer3 = nn.Conv1d(_in, _out2, 3, 1, 5, 5)

        self.merge = nn.Conv1d(_out2, _out, 1)

        self.attention = nn.Sequential(
            nn.Linear(_out2*4, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        r0 = F.gelu(self.layer0(x))
        r1 = F.gelu(self.layer1(x))
        r2 = F.gelu(self.layer2(x))
        r3 = F.gelu(self.layer3(x))

        r = torch.cat([r0, r1, r2, r3], dim=1)

        mean_r = torch.einsum('NCD->NC', r)
        attention = self.attention(mean_r)
        
        r = torch.stack([r0, r1, r2, r3], dim=1)

        attentioned = torch.einsum('NACD,NA->NCD', r, attention)
        merged = F.gelu(self.merge(attentioned))
        return merged
    
class TCN_block(nn.Module):
    def __init__(self, state_shape, ch=[64,128,256], with_pool=False, use_AF=False) -> None:
        super().__init__()
        # shape : [N,T,C]
        self.state_shape = state_shape
        self.layers = []
        
        self.layers.append(nn.Conv1d(state_shape[0], ch[0], 1))
        self.layers.append(nn.GELU())

        for i in range(len(ch)-1):
            _in = ch[i]
            _out = ch[i+1]
            if use_AF: 
                layer = AF_block(_in, _in)
            else:
                layer = nn.Conv1d(_in, _out, 3)
            self.layers.append(layer)
            if with_pool: 
                pool = nn.Conv1d(_in, _out, 4, 2)
                act = nn.GELU()
                self.layers.append(pool)
                self.layers.append(act)

        self.layers = nn.Sequential(*self.layers)
        self.vector_size = 1
        self._build_test()

    @property
    def device(self):
        return next(self.parameters()).device
    
    def _build_test(self):
        with torch.no_grad():
            x = torch.zeros([1,*self.state_shape]).to(device=self.device)
            x = self.layers(x)
            x = x.view(1,-1)
            self.vector_size = x.size()[1]

    def forward(self, x:torch.Tensor):
        x = self.layers(x)
        x = x.flatten(start_dim=1)
        return x
    
#############################################
#############################################

class MLP(nn.Module):
    def __init__(self, input_dim, hidden=[256,256], out_feat=10, inter_act=nn.GELU()) -> None:
        super().__init__()
        self.ch = [input_dim] + hidden + [out_feat]
        self.layers = []

        for i in range(len(self.ch)-1):
            act = inter_act if i < (len(self.ch)-2) else nn.Identity()
            # Elayer = blk(self.ch[i], self.ch[i+1], _out_act=act)
            layer = nn.Linear(self.ch[i], self.ch[i+1])
            self.layers.append(layer)
            self.layers.append(act)
        
        self.layers = nn.Sequential(*self.layers)

    def forward(self, state):
        out_features = self.layers(state)
        return out_features

class MLP_TCN(nn.Module):
    def __init__(self, input_dim, hidden=[256,256], out_feat=10, inter_act=nn.GELU()) -> None:
        super().__init__()

        self.TCN = TCN_block(input_dim)
        self.ch = [self.TCN.vector_size] + hidden + [out_feat]
        self.layers = []

        for i in range(len(self.ch)-1):
            act = inter_act if i < (len(self.ch)-2) else nn.Identity()
            # Elayer = blk(self.ch[i], self.ch[i+1], _out_act=act)
            layer = nn.Linear(self.ch[i], self.ch[i+1])
            self.layers.append(layer)
            self.layers.append(act)
        
        self.layers = nn.Sequential(*self.layers)

    def forward(self, state):
        out_features = self.TCN(state)
        out_features = self.layers(out_features)
        return out_features

class AMLP(nn.Module):
    def __init__(self, input_dim, hidden=[256,256], out_feat=10, n_head=1, inter_act=nn.GELU()) -> None:
        super().__init__()
        self.ch = [input_dim] + hidden# + [out_feat]

        # self.root = nn.Sequential(
        #     ResidualLayer(MultiHeadAttention(input_dim))
        # )

        self.layers = []
        for i in range(len(self.ch)-1):
            act = inter_act if i < (len(self.ch)-2) else nn.Identity()
            # Elayer = blk(self.ch[i], self.ch[i+1], _out_act=act)
            layer = nn.Linear(self.ch[i], self.ch[i+1])
            self.layers.append(layer)
            self.layers.append(act)
        self.layers = nn.Sequential(*self.layers)

        self.exit = nn.Sequential(
            inter_act,
            ResidualLayer(MultiHeadAttention(self.ch[-1], n_head=n_head)),
            nn.Linear(self.ch[-1], out_feat)
        )
        
    def forward(self, state):
        # out_features = self.layers(state)
        # out_features = self.root(state)
        out_features = self.layers(state)
        out_features = self.exit(out_features)
        return out_features
    
class cMLP(nn.Module):
    def __init__(self, input_dim, addtional_dim, hidden=[256,256], out_feat=10, inter_act=nn.GELU()) -> None:
        super().__init__()
        self.ch = [input_dim] + hidden + [out_feat]
        self.layers = []

        for i in range(len(self.ch)-1):
            act = inter_act if i < (len(self.ch)-2) else nn.Identity()
            # Elayer = blk(self.ch[i], self.ch[i+1], _out_act=act)
            layer = nn.Linear(self.ch[i], self.ch[i+1])
            self.layers.append(layer)
            self.layers.append(act)
        
        self.layers = nn.Sequential(*self.layers)

        self.add_cond = nn.Sequential(
            nn.Linear(out_feat+addtional_dim, out_feat),
            inter_act,
            nn.Linear(out_feat, out_feat),
        )

    def forward(self, x, additional_x):
        emb = self.layers(x)
        emb = torch.cat([emb, additional_x], dim=1)
        emb = self.add_cond(emb)
        return emb
    