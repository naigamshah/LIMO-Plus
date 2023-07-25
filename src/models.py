import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from src.constants import *
import functools
import math

def _cosine_decay_warmup(iteration, warmup_iterations, total_iterations, min_mult):
    """
    Linear warmup from 0 --> 1.0, then decay using cosine decay to 0.1
    """
    if iteration <= warmup_iterations:
        multiplier = iteration / warmup_iterations
    else:
        if iteration >= total_iterations:
            multiplier = min_mult
        else:
            multiplier = (iteration - warmup_iterations) / (total_iterations - warmup_iterations)
            multiplier = max(min_mult, 0.5 * (1 + math.cos(math.pi * multiplier)))
    return multiplier


def CosineAnnealingLRWarmup(optimizer, T_max, T_warmup, min_mult=0.01):
    _decay_func = functools.partial(
        _cosine_decay_warmup,
        warmup_iterations=T_warmup, total_iterations=T_max, min_mult=min_mult
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _decay_func)
    return scheduler

class VAE(pl.LightningModule):
    def __init__(self, max_len, vocab_len, latent_dim, embedding_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.vocab_len = vocab_len
        self.embedding_dim = embedding_dim
        self.init_modules()  
    
    def init_modules(self):
        self.embedding = nn.Embedding(self.vocab_len, self.embedding_dim, padding_idx=0)
        self.encoder = nn.Sequential(nn.Linear(self.max_len * self.embedding_dim, 2000),
                                     nn.ReLU(),
                                     nn.Linear(2000, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, self.latent_dim * 2))
        self.decoder = nn.Sequential(nn.Linear(self.latent_dim, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, 2000),
                                     nn.ReLU(),
                                     nn.Linear(2000,self. max_len * self.vocab_len))

    def encode(self, x):
        x = self.encoder(self.embedding(x).view((len(x), -1))).view((-1, 2, self.latent_dim))
        mu, log_var = x[:, 0, :], x[:, 1, :]
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std, mu, log_var
    
    def decode(self, x):
        return F.log_softmax(self.decoder(x).view((-1, self.max_len, self.vocab_len)), dim=2).view((-1, self.max_len * self.vocab_len))
    
    def forward(self, x):
        z, mu, log_var = self.encode(x)
        return self.decode(z), z, mu, log_var
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        return {'optimizer': optimizer}
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        scheduler = CosineAnnealingLRWarmup(
            optimizer, T_max=50000, T_warmup=100, min_mult=0.01)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    
    def loss_function(self, pred, target, mu, log_var, batch_size, p):
        nll = F.nll_loss(pred, target, ignore_index=0)
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / (batch_size * pred.shape[1])
        return nll + p * kld, nll, kld
    
    def training_step(self, train_batch, batch_idx):
        out, z, mu, log_var = self(**train_batch)
        p = 0.1 * (min((self.global_step % 1000) / 1000, 0.5)*2) # 0.01 #min(self.current_epoch/10, 0.1)  #0.1
        loss, nll, kld = self.loss_function(out.reshape((-1, self.vocab_len)), train_batch["x"].flatten(), mu, log_var, len(train_batch["x"]), p)
        self.log('train_loss', loss)
        self.log('train_nll', nll)
        self.log('train_kld', kld)
        return loss
        
    def validation_step(self, val_batch, batch_idx):
        out, z, mu, log_var = self(**val_batch)
        p = 0.1 * (min((self.global_step % 1000) / 1000, 0.5)*2) # 0.01 #min(self.current_epoch/10, 0.1)  #0.1
        loss, nll, kld = self.loss_function(out.reshape((-1, self.vocab_len)), val_batch["x"].flatten(), mu, log_var, len(val_batch["x"]), p)
        self.log('val_loss', loss)
        self.log('val_nll', nll)
        self.log('val_kld', kld)
        self.log('val_mu', torch.mean(mu))
        self.log('val_logvar', torch.mean(log_var))
        return loss

class cVAE(VAE):
    def __init__(self, max_len, vocab_len, latent_dim, embedding_dim):
        super().__init__(max_len, vocab_len, latent_dim, embedding_dim)
    
    def init_modules(self):
        self.embedding = nn.Embedding(self.vocab_len, self.embedding_dim, padding_idx=0)
        self.inp_encoder = nn.Linear(self.max_len*self.embedding_dim, 2000)
        self.encoder = nn.Sequential(#nn.Linear((max_len+2) * embedding_dim, 2000),
                                     nn.Linear(2000 + 2*self.embedding_dim, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, self.latent_dim * 2))
        
        self.decoder = nn.Sequential(nn.Linear(self.latent_dim + 2*self.embedding_dim, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, 2000),
                                     nn.ReLU(),
                                     nn.Linear(2000, self.max_len * self.vocab_len))
        
        self.cond_emb = nn.ModuleDict(dict(
            sa=nn.Sequential(
                nn.Linear(1, self.embedding_dim)
                ),
            qed=nn.Sequential(
                nn.Linear(1, self.embedding_dim)
                ),
        ))

    def encode(self, x, sa, qed):
        x = self.inp_encoder(self.embedding(x).view((len(x), -1)))
        enc_inp_emb = torch.cat([
                self.cond_emb.qed(qed), 
                self.cond_emb.sa(sa),  
                x],
            dim=1
        )
        x = self.encoder(enc_inp_emb).view((-1, 2, self.latent_dim))
        mu, log_var = x[:, 0, :], x[:, 1, :]
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std, mu, log_var
    
    def decode(self, z, sa=None, qed=None):
        if sa is None:
            sa = (torch.ones(z.shape[0], 1, device=z.device) * SA_TARGET -  SA_MEAN) / SA_STD 
        if qed is None:
            qed = (torch.ones(z.shape[0], 1, device=z.device) * QED_TARGET - QED_MEAN) / QED_STD
        dec_inp_emb = torch.cat([
                self.cond_emb.qed(qed), 
                self.cond_emb.sa(sa),  
                z],
            dim=1
        )
        return F.log_softmax(self.decoder(dec_inp_emb).view((-1, self.max_len, self.vocab_len)), dim=2).view((-1, self.max_len * self.vocab_len))
    
    def forward(self, x, sa, qed):
        z, mu, log_var = self.encode(x, sa, qed)
        return self.decode(z, sa, qed), z, mu, log_var

class cVAEFormer(cVAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def init_modules(self):
        self.inp_emb = nn.Embedding(self.vocab_len, self.embedding_dim)
        self.lm_head = nn.Linear(self.embedding_dim, self.vocab_len, bias=False)
        self.inp_emb.weight = self.lm_head.weight # weight tying
        self.time_emb = nn.Parameter(torch.zeros(1, self.max_len, self.embedding_dim))

        self.z_emb = nn.Parameter(torch.zeros(1, self.embedding_dim))
        self.reparametrize = nn.Linear(self.embedding_dim, self.embedding_dim*2)
        self.cond_emb = nn.ModuleDict(dict(
            sa=nn.Sequential(
                nn.Linear(1, self.embedding_dim)
                ),
            qed=nn.Sequential(
                nn.Linear(1, self.embedding_dim)
                )
        ))

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.embedding_dim, 
                nhead=8,
                dim_feedforward=self.embedding_dim*4,
                activation="gelu",
                batch_first=True,
                norm_first=True), 
            num_layers=4
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=self.embedding_dim,
                nhead=8,
                dim_feedforward=self.embedding_dim*4,
                activation='gelu',
                batch_first=True,
                norm_first=True),# important for stability
            num_layers=4
        )

        sz = self.max_len + 1 # start token
        self.dec_start_token = nn.Parameter(torch.zeros(1, self.embedding_dim))
        self.tgt_time_emb = nn.Parameter(torch.zeros(1, self.max_len, self.embedding_dim))
        decoder_mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        self.decoder_mask = decoder_mask.float().masked_fill(decoder_mask == 0, float('-inf')).masked_fill(decoder_mask == 1, float(0.0))

    def encode(self, x, sa, qed):
        src_key_mask = torch.stack([row != 0 for row in x], dim=0).bool()
        src_key_mask = torch.cat([torch.ones(x.size(0), 3, dtype=bool,device=x.device),src_key_mask],dim=1)
        x = self.inp_emb(x) + self.time_emb
        enc_inp_emb = torch.cat([
                self.z_emb.unsqueeze(0).expand(x.size(0),-1,-1),  
                self.cond_emb.qed(qed).unsqueeze(1), 
                self.cond_emb.sa(sa).unsqueeze(1),
                x],
            dim=1
        )

        x = self.encoder(enc_inp_emb, src_key_padding_mask=src_key_mask)
        x = self.reparametrize(x[:, 0])
        mu, log_var = x[:, :self.embedding_dim], x[:, self.embedding_dim:]
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std, mu, log_var
    
    def decode(self, z, sa=None, qed=None, true_targets=None):
        if sa is None:
            sa = (torch.ones(z.shape[0], 1, device=z.device) * SA_TARGET -  SA_MEAN) / SA_STD 
        if qed is None:
            qed = (torch.ones(z.shape[0], 1, device=z.device) * QED_TARGET - QED_MEAN) / QED_STD
        dec_inp_emb = torch.cat([
                z.unsqueeze(1),
                self.cond_emb.qed(qed).unsqueeze(1), 
                self.cond_emb.sa(sa).unsqueeze(1)],
            dim=1
        )
        if self.training:
            # teacher forcing
            tgt = self.inp_emb(true_targets[:, :-1]) + self.time_emb[:, :-1]
            tgt = torch.cat([self.dec_start_token.expand(z.size(0), -1,-1), tgt], dim=1)
            x = self.decoder(
                tgt=tgt, 
                memory=dec_inp_emb, 
                tgt_mask=self.decoder_mask[:-1,:-1].to(self.device))
            x = self.lm_head(x)
            x[:,:,0] = float('-inf')
        else:
            dec_tokens = [self.dec_start_token.expand(z.size(0), -1).unsqueeze(1)] 
            dec_outputs = []
            assert len(self.decoder_mask.shape) == 2
            for i in range(self.max_len):
                x = self.decoder(
                    tgt=torch.cat(dec_tokens, dim=1), 
                    memory=dec_inp_emb, 
                    tgt_mask=self.decoder_mask[:i+1, :i+1].to(self.device))
                dec_logits = self.lm_head(x[:,-1,:])
                dec_logits[:,0] = float('-inf')
                dec_outputs.append(dec_logits)
                next_token = self.inp_emb(torch.argmax(dec_logits, dim=1)).unsqueeze(1) + self.time_emb[:, i, :]
                dec_tokens.append(next_token)
            x = torch.stack(dec_outputs, dim=1)
        
        return F.log_softmax(x, dim=-1).view((-1, self.max_len * self.vocab_len))
    
    def forward(self, x, sa, qed):
        z, mu, log_var = self.encode(x, sa, qed)
        return self.decode(z, sa, qed, x), z, mu, log_var
    
    
    
class PropertyPredictor(pl.LightningModule):
    def __init__(self, in_dim, learning_rate=0.001):
        super(PropertyPredictor, self).__init__()
        self.learning_rate = learning_rate
        self.fc = nn.Sequential(nn.Linear(in_dim, 1000),
                                nn.ReLU(),
                                nn.Linear(1000, 1000),
                                nn.ReLU(),
                                nn.Linear(1000, 1))
        
    def forward(self, x):
        return self.fc(x)
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def loss_function(self, pred, real):
        return F.mse_loss(pred, real)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss_function(out, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss_function(out, y)
        self.log('val_loss', loss)
        return loss