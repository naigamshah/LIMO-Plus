import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F


class VAE(pl.LightningModule):
    def __init__(self, max_len, vocab_len, latent_dim, embedding_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.vocab_len = vocab_len
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_len, embedding_dim, padding_idx=0)
        self.encoder = nn.Sequential(nn.Linear(max_len * embedding_dim, 2000),
                                     nn.ReLU(),
                                     nn.Linear(2000, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, latent_dim * 2))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, 2000),
                                     nn.ReLU(),
                                     nn.Linear(2000, max_len * vocab_len))
        
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
        optimizer = optim.AdamW(self.parameters(), lr=0.0001, weight_decay=0.1)
        return {'optimizer': optimizer}
    
    def loss_function(self, pred, target, mu, log_var, batch_size, p):
        nll = F.nll_loss(pred, target)
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / (batch_size * pred.shape[1])
        return nll + p * kld, nll, kld
    
    def training_step(self, train_batch, batch_idx):
        out, z, mu, log_var = self(**train_batch)
        p = 0.01 * (min((self.global_step % 1000) / 1000, 0.5)*2) # 0.01 #min(self.current_epoch/10, 0.1)  #0.1
        loss, nll, kld = self.loss_function(out.reshape((-1, self.vocab_len)), train_batch["x"].flatten(), mu, log_var, len(train_batch), p)
        self.log('train_loss', loss)
        self.log('train_nll', nll)
        self.log('train_kld', kld)
        return loss
        
    def validation_step(self, val_batch, batch_idx):
        out, z, mu, log_var = self(**val_batch)
        p = 0.01 * (min((self.global_step % 1000) / 1000, 0.5)*2) # 0.01 #min(self.current_epoch/10, 0.1)  #0.1
        loss, nll, kld = self.loss_function(out.reshape((-1, self.vocab_len)), val_batch["x"].flatten(), mu, log_var, len(val_batch), p)
        self.log('val_loss', loss)
        self.log('val_nll', nll)
        self.log('val_kld', kld)
        self.log('val_mu', torch.mean(mu))
        self.log('val_logvar', torch.mean(log_var))
        return loss

class cVAE(VAE):
    def __init__(self, max_len, vocab_len, latent_dim, embedding_dim):
        super().__init__(max_len, vocab_len, latent_dim, embedding_dim)

        self.embedding = nn.Embedding(vocab_len, embedding_dim, padding_idx=0)
        self.encoder = nn.Sequential(nn.Linear((max_len+2) * embedding_dim, 2000),
                                     nn.ReLU(),
                                     nn.Linear(2000, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, latent_dim * 2))
        
        self.decoder = nn.Sequential(nn.Linear(latent_dim + 2*embedding_dim, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, 2000),
                                     nn.ReLU(),
                                     nn.Linear(2000, max_len * vocab_len))
        
        self.cond_emb = nn.ModuleDict(dict(
            sa=nn.Sequential(
                nn.Linear(1, embedding_dim*2), 
                nn.ReLU(), 
                nn.Linear(embedding_dim*2, embedding_dim)
                ),
            qed=nn.Sequential(
                nn.Linear(1, embedding_dim*2), 
                nn.ReLU(), 
                nn.Linear(embedding_dim*2, embedding_dim)
                ),
        ))

    def encode(self, x, sa, qed):
        inp_emb = torch.cat([
                self.cond_emb.qed(qed), 
                self.cond_emb.sa(sa),  
                self.embedding(x).view((len(x), -1))],
            dim=1
        )
        x = self.encoder(inp_emb).view((-1, 2, self.latent_dim))
        mu, log_var = x[:, 0, :], x[:, 1, :]
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std, mu, log_var
    
    def decode(self, z, sa=None, qed=None):
        if sa is None:
            sa = torch.ones(z.shape[0], 1, device=z.device) * 0.2
        if qed is None:
            qed = torch.ones(z.shape[0], 1, device=z.device) * 0.8
        inp_emb = torch.cat([
                self.cond_emb.qed(qed), 
                self.cond_emb.sa(sa),  
                z],
            dim=1
        )
        return F.log_softmax(self.decoder(inp_emb).view((-1, self.max_len, self.vocab_len)), dim=2).view((-1, self.max_len * self.vocab_len))
    
    def forward(self, x, sa, qed):
        z, mu, log_var = self.encode(x, sa, qed)
        return self.decode(z, sa, qed), z, mu, log_var

class cTransformerVAE(cVAE):
    def __init__(self, max_len, vocab_len, latent_dim, embedding_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.vocab_len = vocab_len

        self.inp_emb = nn.Embedding(vocab_len, embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_len, bias=False)
        self.inp_emb.weight = self.lm_head.weight # weight tying
        self.time_emb = nn.Embedding(max_len, embedding_dim)

        self.z_emb = torch.zeros(1, embedding_dim)
        self.cond_emb = nn.ModuleDict(dict(
            sa=nn.Sequential(
                nn.Linear(1, embedding_dim), 
                nn.ReLU(), 
                nn.Linear(embedding_dim, embedding_dim*4),
                nn.ReLU(), 
                nn.Linear(embedding_dim*4, embedding_dim)
                ),
            qed=nn.Sequential(
                nn.Linear(1, embedding_dim), 
                nn.ReLU(), 
                nn.Linear(embedding_dim, embedding_dim*4),
                nn.ReLU(), 
                nn.Linear(embedding_dim*4, embedding_dim)
                )
        ))

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embedding_dim, 
                nhead=8,
                dim_feedforward=embedding_dim*4,
                activation="gelu",
                batch_first=True,
                norm_first=True), 
            num_layers=4
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=embedding_dim,
                nhead=8,
                dim_feedforward=embedding_dim*4,
                activation='gelu',
                batch_first=True,
                norm_first=True),# important for stability
            num_layers=4
        )
        sz = self.max_len
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        self.register_buffer("mask", mask)

    def encode(self, x):
        raise NotImplementedError()
        x = self.encoder(self.embedding(x).view((len(x), -1))).view((-1, 2, self.latent_dim))
        mu, log_var = x[:, 0, :], x[:, 1, :]
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std, mu, log_var
    
    def decode(self, x):
        raise NotImplementedError()
        return F.log_softmax(self.decoder(x).view((-1, self.max_len, self.vocab_len)), dim=2).view((-1, self.max_len * self.vocab_len))
    
    
    
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
        return optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.1)
    
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