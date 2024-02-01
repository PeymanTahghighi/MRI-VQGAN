import math
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

class GPTConfig:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs) -> None:
        self.vocab_size = vocab_size;
        self.block_size = block_size;

        for k,v in kwargs.items():
            setattr(self, k, v);


class SelfAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.n_head = config.n_head;
        self.key = nn.Linear(config.n_embd, config.n_embd);
        self.query = nn.Linear(config.n_embd, config.n_embd);
        self.value = nn.Linear(config.n_embd, config.n_embd);


        self.attn_drop = nn.Dropout(config.attn_pdrop);
        self.resid_drop = nn.Dropout(config.resid_pdrop);

        self.proj = nn.Linear(config.n_embd, config.n_embd);
    
        triangle_mask =  torch.tril(torch.ones(
            (config.vocab_size, config.vocab_size)
        ));

        self.register_buffer('triangle_mask', triangle_mask);


    def forward(self, x):
        B,T,C = x.size();
        k = self.key(x).view(B,T,self.n_head, C//self.n_head).permute(0,2,1,3); #(b, nh, T, hs);
        q = self.query(x).view(B,T,self.n_head, C//self.n_head).permute(0,2,1,3); #(b, nh, T, hs);
        v = self.value(x).view(B,T,self.n_head, C//self.n_head).permute(0,2,1,3); #(b, nh, T, hs);

        attn = k@q.transpose(-2,-1);
        attn = attn * (C**(-0.5));
        attn = attn.masked_fill(self.triangle_mask[:T, :T] == 0, float('-inf'));
        attn = torch.softmax(attn, -1);
        attn = self.attn_drop(attn);
        out = attn@v;
        out = out.transpose(1,2).contiguous().view(B,T,C);
        out = self.resid_drop(self.proj(out));
        return out;



class Block(nn.Module):
    def __init__(self, config) -> None:
        super().__init__();
        self.ln1 = nn.LayerNorm(config.n_embd);
        self.ln2 = nn.LayerNorm(config.n_embd);
        self.attn = SelfAttention(config);
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd*4),
            nn.GELU(),
            nn.Linear(4*config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop)
        )

    def forward(self, x):
        attn = self.attn(self.ln1(x));
        x = x + attn;
        out = x + self.mlp(self.ln2(x));
        return out;


class GPT(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 block_size, 
                 n_layer = 12, 
                 n_head = 8, 
                 n_embd = 256,
                 embd_pdrop = 0,
                 resid_pdrop = 0,
                 attn_pdrop = 0,
                 n_unmasked = 0
                 ):
        super().__init__()
        config = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                           n_unmasked=n_unmasked)
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd);
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))  
        self.drop = nn.Dropout(config.embd_pdrop);

        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)]);
        self.ln_f = nn.LayerNorm(config.n_embd);
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias= False);
        self.block_size = config.block_size;
        self.apply(self.init_weights);
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.02, std = 0.02);
            if m.bias is not None:
                m.bias.data.zero_();
        if isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight.data, mean = 0.0, std = 0.02);
        if isinstance(m, nn.LayerNorm):
            m.bias.data.zero_();
            m.weight.data.fill_(1);


    def forward(self, x):
        embd = self.token_embedding(x);
        pos_embd = self.pos_emb[:,:embd.shape[1],:];
        x = self.drop(embd + pos_embd);
        x = self.blocks(x);
        x = self.ln_f(x);
        logits = self.head(x);
        return logits;


def test():
    x = np.array([np.random.randint(0,64) for _ in range(32)]);
    x = torch.from_numpy(x);
    gpt = GPT(64,32);
    out = gpt(x.unsqueeze(dim = 0));
    print(out.shape);

if __name__ == "__main":
    test();
