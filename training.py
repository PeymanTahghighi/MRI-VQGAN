import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from discriminator import Discriminator
from lpips import LPIPS
from vqgan import VQGAN
from utils import  weights_init
from data_utils import cache_dataset, MRIDataset, load_data


class TrainVQGAN:
    def __init__(self, args):
        self.args = args;
        self.vqgan = VQGAN(args).to(device=args.device)
        
        self.discriminator = Discriminator(args).to(device=args.device)
        self.discriminator.apply(weights_init)
        self.perceptual_loss = LPIPS().eval().to(device=args.device)
        self.opt_vq, self.opt_disc = self.configure_optimizers()

        self.prepare_training()

        self.train_dataset, self.test_dataset = load_data(args)
        self.best_loss = 100;

    def configure_optimizers(self):
        lr = self.args.learning_rate
        opt_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters()) +
            list(self.vqgan.decoder.parameters()) +
            list(self.vqgan.codebook.parameters()) +
            list(self.vqgan.quant_conv.parameters()) +
            list(self.vqgan.post_quant_conv.parameters()),
            lr=lr, eps=1e-08, betas=(self.args.beta1, self.args.beta2)
        )
        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=lr, eps=1e-08, betas=(self.args.beta1, self.args.beta2))

        return opt_vq, opt_disc

    @staticmethod
    def prepare_training():
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    def set_to_train(self):
        self.vqgan.train();
        self.discriminator.train();
    
    def set_to_eval(self):
        self.vqgan.eval();
        self.discriminator.eval();
    
    def train(self):
        for epoch in range(self.args.epochs):
            self.train_step(epoch);
            valid_loss = self.valid_step(epoch);
            if valid_loss < self.best_loss:
                print('new best model found!');
                torch.save(self.vqgan.state_dict(), os.path.join("checkpoints", f"vqgan_epoch.pt"))
    
    def valid_step(self, epoch):
        self.set_to_eval();
        total_loss = [];
        with torch.no_grad():
            steps_per_epoch = len(self.test_dataset)
            with tqdm(range(len(self.test_dataset))) as pbar:
                for i, imgs in zip(pbar, self.test_dataset):
                    imgs = imgs.to(device=self.args.device)
                    decoded_images, _, q_loss = self.vqgan(imgs)
                    disc_real = self.discriminator(imgs)
                    disc_fake = self.discriminator(decoded_images)

                    disc_factor = self.vqgan.adopt_weight(self.args.disc_factor, epoch*steps_per_epoch+i, threshold=args.disc_start)

                    perceptual_loss = self.perceptual_loss(imgs, decoded_images)
                    rec_loss = torch.abs(imgs - decoded_images)
                    perceptual_rec_loss = self.args.perceptual_loss_factor * perceptual_loss + self.args.rec_loss_factor * rec_loss
                    perceptual_rec_loss = perceptual_rec_loss.mean()
                    g_loss = -torch.mean(disc_fake)


                    vq_loss = perceptual_rec_loss + q_loss + disc_factor * g_loss

                    d_loss_real = torch.mean(F.relu(1. - disc_real))
                    d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                    gan_loss = disc_factor * 0.5*(d_loss_real + d_loss_fake)

                    pbar.set_postfix(
                    VQ_Loss=np.round(vq_loss.cpu().detach().numpy().item(), 5),
                    GAN_Loss=np.round(gan_loss.cpu().detach().numpy().item(), 3)
                    )

                    total_loss.append(vq_loss.item() + gan_loss.item());
                    if i % 10 == 0:
                        unnormalized = decoded_images.detach().cpu() * torch.from_numpy(np.array([0.229, 0.224, 0.225]))[None, :, None, None] + torch.from_numpy(np.array([0.485, 0.456, 0.406]))[None, :, None, None];
                        imgs = imgs.detach().cpu() * torch.from_numpy(np.array([0.229, 0.224, 0.225]))[None, :, None, None] + torch.from_numpy(np.array([0.485, 0.456, 0.406]))[None, :, None, None];

                        real_fake_images = torch.cat((imgs[:4], unnormalized[:4]))
                        vutils.save_image(real_fake_images, os.path.join("results", f"{epoch}_{i}.jpg"), nrow=4)
        return np.mean(total_loss);

    def train_step(self, epoch):
        self.set_to_train();

        steps_per_epoch = len(self.train_dataset)
        with tqdm(range(len(self.train_dataset))) as pbar:
            for i, imgs in zip(pbar, self.train_dataset):
                imgs = imgs.to(device=self.args.device)
                decoded_images, _, q_loss = self.vqgan(imgs)

                disc_real = self.discriminator(imgs)
                disc_fake = self.discriminator(decoded_images)

                disc_factor = self.vqgan.adopt_weight(self.args.disc_factor, epoch*steps_per_epoch+i, threshold=self.args.disc_start)

                perceptual_loss = self.perceptual_loss(imgs, decoded_images)
                rec_loss = torch.abs(imgs - decoded_images)
                perceptual_rec_loss = self.args.perceptual_loss_factor * perceptual_loss + self.args.rec_loss_factor * rec_loss
                perceptual_rec_loss = perceptual_rec_loss.mean()
                g_loss = -torch.mean(disc_fake)

                λ = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
                vq_loss = perceptual_rec_loss + q_loss + disc_factor * λ * g_loss

                d_loss_real = torch.mean(F.relu(1. - disc_real))
                d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                gan_loss = disc_factor * 0.5*(d_loss_real + d_loss_fake)

                self.opt_vq.zero_grad()
                vq_loss.backward(retain_graph = True)

                self.opt_disc.zero_grad()
                gan_loss.backward()

                self.opt_vq.step()
                self.opt_disc.step()

                

                pbar.set_postfix(
                    VQ_Loss=np.round(vq_loss.cpu().detach().numpy().item(), 5),
                    GAN_Loss=np.round(gan_loss.cpu().detach().numpy().item(), 3)
                )
                pbar.update(0)
           


if __name__ == '__main__':
    
    #cache_dataset('C:\\PhD\\Thesis\\MRI Project\\SSLMRI\\miccai-processed')
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=512, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=2048, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='cache', help='Path to data (default: cache)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=6, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')
    parser.add_argument('--num-workers', type=float, default=0, help='Data loader num workers.')

    args = parser.parse_args()
    #args.dataset_path = r"cache"

    train_vqgan = TrainVQGAN(args)
    train_vqgan.train();


