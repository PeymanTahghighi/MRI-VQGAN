import torch
import torch.nn as nn
import torch.nn.functional as F
from mingpt import GPT
from vqgan import VQGAN

class VQGANTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.sos_token = args.sos_token;

        self.vqgan = self.load_vqgan(args);

        self.transformer_config = {
            'vocab_size': args.num_codebook_vectors,
            'block_size': 512,
            'n_layer':24,
            'n_head':16,
            'n_embd' : 1024
        }

        self.transformer = GPT(**self.transformer_config);

        self.pkeep = args.pkeep;

    @staticmethod
    def load_vqgan(args):
        model = VQGAN(args);
        model.load_checkpoint(args.checkpoint_path);
        model = model.eval();
        return model;

    @torch.no_grad
    def encode_to_z(self, x):
        quant_z, indices, _ = self.vqgan.encode(x);
        indices = indices.view(quant_z.shape[0], -1);
        return quant_z, indices;
    
    @torch.no_grad
    def z_to_image(self, indices, p1 = 16, p2 = 16):
        ix_to_vectors = self.vqgan.codebook.embedding(indices).reshape(indices.shape[0], p1, p2, 256);
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2);
        image = self.vqgan.decode(ix_to_vectors);
        return image.cpu().detach()[0].permute(1,2,0);

    def forward(self, x):
        _,indices = self.encode_to_z(x);

        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token;
        sos_tokens = sos_tokens.long().to('cuda');

        mask = torch.bernoulli(self.pkeep * torch.ones(indices.shape, device=indices.device));
        mask = mask.round().to(dtype=torch.int64);
        random_indices = torch.randint_like(indices, self.transformer_config['vocab_size']);
        new_indices = mask * indices + (1 - mask) * random_indices;
        new_indices = torch.cat((sos_tokens, new_indices), dim=1);

        logits = self.transformer(new_indices[:,:-1]);
        return logits, indices;

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, top_k=100):
        self.transformer.eval()
        x = torch.cat((c, x), dim=1)
        for k in range(steps):
            logits = self.transformer(x)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)

            probs = F.softmax(logits, dim=-1)

            ix = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, ix), dim=1)

        x = x[:, c.shape[1]:]
        self.transformer.train()
        return x

    @torch.no_grad()
    def log_images(self, x):
        log = dict()

        _, indices = self.encode_to_z(x)
        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to("cuda")

        # start_indices = indices[:, :indices.shape[1] // 2]
        # sample_indices = self.sample(start_indices, sos_tokens, steps=indices.shape[1] - start_indices.shape[1])
        # half_sample = self.z_to_image(sample_indices)

        start_indices = indices[:, :0]
        sample_indices = self.sample(start_indices, sos_tokens, steps=indices.shape[1])
        full_sample = self.z_to_image(sample_indices)

        x_rec = self.z_to_image(indices)

        log["input"] = x
        log["rec"] = x_rec
        log["full_sample"] = full_sample
        a = x_rec[None,...];
        return log, torch.concat((x.detach().permute(0,2,3,1).cpu(), x_rec[None,...],  full_sample[None,...]))

