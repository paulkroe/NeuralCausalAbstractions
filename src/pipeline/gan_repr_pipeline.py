import torch as T
import torch.nn as nn

from src.scm.ncm.gan_ncm import GAN_NCM, Discriminator

from .gan_pipeline import GANPipeline


def log(x):
    return T.log(x + 1e-8)


class GANReprPipeline(GANPipeline):
    def __init__(self, datagen, cg, v_size, v_type, rep_v_size, rep_v_type, repr_model=None, hyperparams=None,
                 ncm_model=GAN_NCM):
        super().__init__(datagen, cg, rep_v_size, rep_v_type, repr_model=repr_model, hyperparams=hyperparams,
                         ncm_model=ncm_model)

        self.raw_v_size = v_size
        self.raw_v_type = v_type

        self.raw_disc = Discriminator(self.raw_v_size, self.raw_v_type,
                                      disc_use_sigmoid=(hyperparams.get("gan-mode", "NA") != "wgan"),
                                      hyperparams=hyperparams)

        self.rep_lr = hyperparams["rep-lr"]
        self.train_encoder = (hyperparams['repr'] != "auto_enc_notrain")
        self.classify = (hyperparams['repr'] == "auto_enc_conditional")
        self.classify_lambda = hyperparams['rep-class-lambda']

        self.recon_loss = nn.MSELoss()
        self.classify_loss = nn.BCELoss()

        if hyperparams["verbose"]:
            print("RAW DISCRIMINATOR")
            print(self.raw_disc)
            if repr_model is not None:
                print("ENCODER")
                print(self.repr_model.encoders)
                print("DECODER")
                print(self.repr_model.decoders)
                if self.classify:
                    print("PARENT HEADS")
                    print(self.repr_model.parent_heads)

    def configure_optimizers(self):
        if self.gan_mode == "wgan":
            opt_gen = T.optim.RMSprop(self.ncm.f.parameters(), lr=self.gen_lr, alpha=self.alpha)
            opt_disc = T.optim.RMSprop(self.raw_disc.parameters(), lr=self.disc_lr, alpha=self.alpha)
            opt_pu = T.optim.RMSprop(self.ncm.pu.parameters(), lr=self.gen_lr, alpha=self.alpha)
        else:
            opt_gen = T.optim.Adam(self.ncm.f.parameters(), lr=self.gen_lr)
            opt_disc = T.optim.Adam(self.raw_disc.parameters(), lr=self.disc_lr)
            opt_pu = T.optim.Adam(self.ncm.pu.parameters(), lr=self.gen_lr)

        opt_enc = T.optim.Adam(self.repr_model.encoders.parameters(), lr=self.rep_lr)
        opt_dec = T.optim.Adam(self.repr_model.decoders.parameters(), lr=self.rep_lr)
        if self.classify:
            opt_head = T.optim.Adam(self.repr_model.parent_heads.parameters(), lr=self.rep_lr)
            return opt_gen, opt_disc, opt_pu, opt_enc, opt_dec, opt_head

        return opt_gen, opt_disc, opt_pu, opt_enc, opt_dec

    def _get_loss(self, loss, out, data):
        total = 0
        for v in out:
            total += loss(out[v], data[v])
        return total

    def _step_repr_model(self, batch, batch_idx):
        label_loss = 0
        label_loss_log = 0

        if self.classify:
            out_batch, label_out, label_truth = self.repr_model(batch, classify=True)
            label_loss = self.classify_lambda * self._get_loss(self.classify_loss, label_out, label_truth)
            label_loss_log = label_loss.item()
        else:
            out_batch = self.repr_model(batch)

        recon_loss = self._get_loss(self.recon_loss, out_batch, batch)
        recon_loss_log = recon_loss.item()
        return label_loss, recon_loss, label_loss_log, recon_loss_log

    def _step_discriminator(self, ncm_batch, real_batch, decode):
        if decode:
            ncm_batch = self.repr_model.decode(ncm_batch)
            ncm_disc_real_out = self.raw_disc(real_batch)
            ncm_disc_fake_out = self.raw_disc(ncm_batch)
        else:
            real_batch = self.repr_model.encode(real_batch)
            ncm_disc_real_out = self.disc(real_batch)
            ncm_disc_fake_out = self.disc(ncm_batch)

        D_loss = self._get_D_loss(ncm_disc_real_out, ncm_disc_fake_out)

        grad_penalty = 0
        grad_penalty_log = 0
        if self.gan_mode == "wgangp":
            grad_penalty = self._get_gradient_penalty(real_batch, ncm_batch)
            self.log('grad_penalty', grad_penalty, prog_bar=True)
            grad_penalty_log = grad_penalty.item()

        return D_loss, grad_penalty, D_loss.item(), grad_penalty_log

    def _step_generator(self, ncm_batch, decode):
        if decode:
            ncm_batch = self.repr_model.decode(ncm_batch)
            ncm_disc_fake_out = self.raw_disc(ncm_batch)
        else:
            ncm_disc_fake_out = self.disc(ncm_batch)

        G_loss = self._get_G_loss(ncm_disc_fake_out)
        return G_loss, G_loss.item()

    def training_step(self, batch, batch_idx):
        if self.classify:
            G_opt, D_opt, PU_opt, enc_opt, dec_opt, class_opt = self.optimizers()
        else:
            G_opt, D_opt, PU_opt, enc_opt, dec_opt = self.optimizers()
            class_opt = None

        ncm_n = self.ncm_batch_size

        # Train Discriminator
        total_d_loss = 0
        grad_penalty_log = 0
        for d_iter in range(self.d_iters):
            G_opt.zero_grad()
            D_opt.zero_grad()
            ncm_batch = self.ncm(ncm_n)
            real_batch = {k: v[d_iter * self.cut_batch_size:(d_iter + 1) * self.cut_batch_size].float()
                          for (k, v) in batch.items()}

            D_loss_1, grad_penalty_1, D_loss_log_1, grad_penalty_log_1 = self._step_discriminator(ncm_batch, real_batch,
                                                                                          decode=True)
            D_loss_2, grad_penalty_2, D_loss_log_2, grad_penalty_log_2 = self._step_discriminator(ncm_batch, real_batch,
                                                                                          decode=False)
            grad_penalty_log = grad_penalty_log_1 + grad_penalty_log_2
            total_d_loss += D_loss_log_1 + D_loss_log_2 + grad_penalty_log
            self.manual_backward(D_loss_1 + D_loss_2 + grad_penalty_1 + grad_penalty_2)

            if ((self.d_iters * batch_idx + d_iter + 1) % self.grad_acc) == 0:
                D_opt.step()

            if self.gan_mode == "wgan":
                for p in self.disc.parameters():
                    p.data.clamp_(-self.grad_clamp, self.grad_clamp)
                for p in self.raw_disc.parameters():
                    p.data.clamp_(-self.grad_clamp, self.grad_clamp)

        # Train Representation Model
        enc_opt.zero_grad()
        dec_opt.zero_grad()
        if self.classify:
            class_opt.zero_grad()
        label_loss, recon_loss, label_loss_log, recon_loss_log = self._step_repr_model(batch, batch_idx)
        self.manual_backward(label_loss + recon_loss)

        # Train Generator
        G_opt.zero_grad()
        PU_opt.zero_grad()
        D_opt.zero_grad()
        ncm_batch = self.ncm(ncm_n)
        G_loss_1, G_loss_log_1 = self._step_generator(ncm_batch, decode=True)
        G_loss_2, G_loss_log_2 = self._step_generator(ncm_batch, decode=False)
        G_loss_log = G_loss_log_1 + G_loss_log_2
        self.manual_backward(G_loss_1 + G_loss_2)

        if ((batch_idx + 1) % self.grad_acc) == 0:
            G_opt.step()
            PU_opt.step()
            dec_opt.step()
            if self.train_encoder:
                enc_opt.step()
            if self.classify:
                class_opt.zero_grad()

        # logging
        if (self.current_epoch + 1) % 10 == 0:
            if not self.logged:
                self.logged = True
                sample = self(n=64)
                for v in self.v_type:
                    self.img_lists[v].append(sample[v].detach().cpu())

        else:
            self.logged = False

        self.log('Rec', recon_loss_log, prog_bar=True)
        if self.classify:
            self.log('Class', label_loss_log, prog_bar=True)
        self.log('G', G_loss_log, prog_bar=True)
        self.log('D', total_d_loss, prog_bar=True)
        if self.gan_mode == "wgan_gp":
            self.log('GP', grad_penalty_log, prog_bar=True)
        self.log('train_loss', 0, prog_bar=True)
