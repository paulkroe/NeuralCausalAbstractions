import numpy as np
import pandas as pd

import torch as T
from torch.autograd import grad

from src.ds.causal_graph import CausalGraph
from src.ds.counterfactual import CTF
from src.metric.evaluation import probability_table, kl
from src.scm.ncm.gan_ncm import GAN_NCM, Discriminator
from src.scm.scm import expand_do
from src.datagen import SCMDataTypes as sdt

from .base_pipeline import BasePipeline

class EmbeddingSampler:
    def __init__(self, npz_path: str, device: T.device = None):
        """
        Loads embeddings and labels from a .npz file and prepares
        per-label index lists for uniform sampling.

        Args:
            npz_path: Path to the .npz file containing 'embeddings' and 'labels'.
            device:   Torch device for returned tensors (default: CPU).
        """
        data = np.load(npz_path)
        self.embeddings = data['embeddings']  # shape [N, D]
        self.labels     = data['labels']      # shape [N]
        self.device     = device or T.device('cpu')

        # Precompute index lists for each numeric label
        unique_labels = np.unique(self.labels)
        self.indices_by_label = {
            int(lbl): np.where(self.labels == lbl)[0]
            for lbl in unique_labels
        }
        for lbl, idxs in self.indices_by_label.items():
            if len(idxs) == 0:
                raise ValueError(f"No embeddings found for label {lbl}")

    def __call__(self, requested: list[int]) -> list[T.Tensor]:
        """
        For each numeric label in `requested`, samples one embedding
        from that label uniformly at random.

        Args:
            requested: e.g. [0, 1, 0, 1, 1, ...] where 0 and 1 correspond
                       to classes mel=0, nv=1 as in your saved file.

        Returns:
            List of torch.Tensor embeddings, one per request.
        """
        out = []
        for lbl in requested:
            lbl += 4
            if lbl not in self.indices_by_label:
                raise KeyError(f"Unknown label {lbl}")
            idxs = self.indices_by_label[lbl]
            choice = np.random.choice(idxs)
            emb = self.embeddings[choice]        # numpy array [D]
            emb_t = T.from_numpy(emb).to(self.device)
            out.append(emb_t)
        return out

sampler = EmbeddingSampler("dat/HAM10000/ham10000_embeddings.npz", 'cpu')


def log(x):
    return T.log(x + 1e-8)


class GANPipeline(BasePipeline):
    def __init__(self, datagen, cg, v_size, v_type, repr_model=None, hyperparams=None, ncm_model=GAN_NCM,
                 maximize=False):
        """
        gan-mode options: vanilla, bgan, wgan
        """
        if hyperparams is None:
            hyperparams = dict()

        self.v_size = v_size
        self.v_type = v_type
        ncm = ncm_model(cg, v_size=v_size, v_type=v_type, default_u_size=hyperparams.get('u-size', 1),
                        hyperparams=hyperparams)
        super().__init__(datagen, cg, ncm, batch_size=hyperparams.get('data-bs', 1000))
        self.automatic_optimization = False

        self.gan_mode = hyperparams.get("gan-mode", "vanilla")
        self.disc = Discriminator(v_size, v_type, disc_use_sigmoid=(hyperparams.get("gan-mode", "NA") != "wgan"),
                                  hyperparams=hyperparams)
        self.repr_model = repr_model

        self.ncm_batch_size = hyperparams.get('ncm-bs', 1000)
        self.grad_acc = hyperparams.get('grad-acc', 1)
        self.d_iters = hyperparams.get('d-iters', 1)
        self.g_iters = hyperparams.get('g-iters', 1)
        self.cut_batch_size = hyperparams.get('data-bs', 1000) // self.d_iters
        self.grad_clamp = hyperparams.get('grad-clamp', 0.01)
        self.gp_weight = hyperparams.get('gp-weight', 10.0)
        self.gp_one_side = hyperparams.get('gp-one-size', False)
        self.gen_lr = hyperparams['lr']
        self.disc_lr = hyperparams['disc-lr']
        self.alpha = hyperparams['alpha']
        self.ordered_v = cg.v

        self.img_query = hyperparams.get('img-query', False)
        self.optimize_query = hyperparams.get('identify', False)
        self.maximize = maximize
        self.max_query_iters = hyperparams['max-epochs']
        self.use_tau = hyperparams.get('use-tau', False)
        self.min_lambda = hyperparams.get('min-lambda', 0.001)
        self.max_lambda = hyperparams.get('max-lambda', 1.0)

        self.eval_samples = hyperparams.get('eval-samples', 1000)

        #self.dat_prob_table = self.datagen.get_prob_table()
        self.logged = False
        self.stored_loss = 1e8
        self.img_lists = {v: [] for v in self.v_type}

        if hyperparams["verbose"]:
            print("NCM")
            print(self.ncm.f)
            print("DISCRIMINATOR")
            print(self.disc.f_disc)

        self.wandb = hyperparams["wandb"]
        self.pretrain_d_epochs = 1

    def forward(self, n=1000, u=None, do={}, evaluating=False):
        out = self.ncm(n, u, do, evaluating=evaluating)
        if self.repr_model is None:
            return out
        else: 
            for k, v in out.items():
                out[k] = v.to(device=next(self.repr_model.decoders.parameters()).device)
            out = self.repr_model.decode(out)
            return out

    def sample_ctf(self, query: CTF, n=64, batch=None, max_iters=1000):
        out = self.ncm.sample_ctf(query, n, batch, max_iters)
        if self.repr_model is not None and type(out) != float:
            out = self.repr_model.decode(out)
        return out

    def configure_optimizers(self):
        if self.gan_mode == "wgan":
            opt_gen = T.optim.RMSprop(self.ncm.f.parameters(), lr=self.gen_lr, alpha=self.alpha)
            opt_disc = T.optim.RMSprop(self.disc.parameters(), lr=self.disc_lr, alpha=self.alpha)
            opt_pu = T.optim.RMSprop(self.ncm.pu.parameters(), lr=self.gen_lr, alpha=self.alpha)
        else:
            opt_gen = T.optim.Adam(self.ncm.f.parameters(), lr=self.gen_lr)
            opt_disc = T.optim.Adam(self.disc.parameters(), lr=self.disc_lr)
            opt_pu = T.optim.Adam(self.ncm.pu.parameters(), lr=self.gen_lr)
        return opt_gen, opt_disc, opt_pu

    def _get_D_loss(self, real_out, fake_out):
        if self.gan_mode == "wgan" or self.gan_mode == "wgangp":
            E_real = T.mean(real_out)
            E_fake = T.mean(fake_out)
            if self.wandb:
                import wandb
                wandb.log({
                    "E_real": E_real,
                    "E_fake": E_fake
                })
            return -(E_real - E_fake)
        else:
            return -T.mean(log(real_out) + log(1 - fake_out))

    def _get_G_loss(self, fake_out):
        if self.gan_mode == "bgan":
            return 0.5 * T.mean((log(fake_out) - log(1 - fake_out)) ** 2)
        elif self.gan_mode == "wgan" or self.gan_mode == "wgangp":
            return -T.mean(fake_out)
        else:
            return -T.mean(log(fake_out))

    def _get_gradient_penalty(self, real_data, fake_data):
        interpolated_data = dict()
        alpha = T.rand(self.ncm_batch_size, 1, device=self.device, requires_grad=True)
        for V in real_data:
            if self.v_type[V] is not sdt.IMAGE:
                v_alpha = alpha.expand_as(real_data[V])
            else:
                v_alpha = alpha[:, :, None, None].expand_as(real_data[V])
            interpolated_data[V] = v_alpha * real_data[V].detach() + (1 - v_alpha) * fake_data[V].detach()

        interpolated_out, inp_set = self.disc(interpolated_data, include_inp=True)
        gradients_norm = 0
        for inp in inp_set:
            if inp is not None:
                gradients = grad(outputs=interpolated_out, inputs=inp,
                                 grad_outputs=T.ones(interpolated_out.size(), device=self.device),
                                 create_graph=True, retain_graph=True)[0]
                gradients = gradients.view(self.ncm_batch_size, -1)
                gradients_norm += T.sum(gradients ** 2, dim=1)
        gradients_norm = T.sqrt(gradients_norm + 1e-12)
        if self.gp_one_side:
            return self.gp_weight * (T.relu(gradients_norm - self.grad_clamp) ** 2).mean()
        return self.gp_weight * ((gradients_norm - self.grad_clamp) ** 2).mean()

    
    # def on_train_start(self) -> None:
    #     # called once before epoch 0
    #     # freeze generator + PU for pre-training
    #     for p in self.ncm.f.parameters():
    #         p.requires_grad = False
    #     for p in self.ncm.pu.parameters():
    #         p.requires_grad = False
    #     self._pretraining = True
    #     print(f"[GANPipeline] Pre-training D for {self.pretrain_d_epochs} epochs → G frozen")

    # def on_train_epoch_start(self) -> None:
    #     # unfreeze generator when we hit the cutoff epoch
    #     if self.current_epoch == self.pretrain_d_epochs:
    #         for p in self.ncm.f.parameters():
    #             p.requires_grad = True
    #         for p in self.ncm.pu.parameters():
    #             p.requires_grad = True
    #         self._pretraining = False
    #         print("[GANPipeline] Finished D pre-training → G unfrozen")
    
    
    def training_step(self, batch, batch_idx):
        # assume batch is a dict of 1-D torch tensors
        Y   = batch['Y']   # shape (m,)
        EMB = batch['EMB']
        X   = batch['X']
        SES = batch['SES']

        # print(SES.shape)

        # # mask for EMB=1 & X=1 & SES=1
        # mask = (EMB == 1) & (X == 1) & (SES == 1)

        # # numerator: those also with Y=1
        # num = ((Y == 1) & mask).sum().float()

        # # denominator: all with EMB=0, X=0, SES=0
        # den = mask.sum().float()

        # # estimate P(Y=1 | EMB=0, X=0, SES=0)
        # estimate = num / den if den > 0 else T.tensor(float('nan'))

        # print(f"Estimated P(Y=1 | EMB=0, X=0, SES=0) = {estimate.item():.3f}")

        # assert 0

        G_opt, D_opt, PU_opt = self.optimizers()
        ncm_n = self.ncm_batch_size

        G_opt.zero_grad()
        PU_opt.zero_grad()

        # Train Discriminator
        total_d_loss = 0
        for d_iter in range(self.d_iters):
            D_opt.zero_grad()
            ncm_batch = self.ncm(ncm_n)
            real_batch = {k: v[d_iter * self.cut_batch_size:(d_iter + 1) * self.cut_batch_size].float()
                          for (k, v) in batch.items()}
            if self.repr_model is not None:
                real_batch = self.repr_model.encode(real_batch)
                real_batch = {k: v.detach() for (k, v) in real_batch.items()}

            ncm_disc_real_out = self.disc(real_batch)
            ncm_disc_fake_out = self.disc(ncm_batch)
            D_loss = self._get_D_loss(ncm_disc_real_out, ncm_disc_fake_out)

            if self.gan_mode == "wgangp":
                grad_penalty = self._get_gradient_penalty(real_batch, ncm_batch)
                self.log('grad_penalty', grad_penalty, prog_bar=True)
                D_loss += grad_penalty

            total_d_loss += D_loss.item()
            self.manual_backward(D_loss)

            if ((self.d_iters * batch_idx + d_iter + 1) % self.grad_acc) == 0:
                D_opt.step()

            if self.gan_mode == "wgan":
                for p in self.disc.parameters():
                    p.data.clamp_(-self.grad_clamp, self.grad_clamp)

            self.ncm.f.zero_grad()
            self.disc.zero_grad()
            self.ncm.pu.zero_grad()

        # Train Generator
        g_loss_record = 0
        ncm_batch = self.ncm(ncm_n)
        ncm_disc_fake_out = self.disc(ncm_batch)
        G_loss = self._get_G_loss(ncm_disc_fake_out)
        g_loss_record += G_loss.item()
        self.manual_backward(G_loss)

        # Optimize Query
        max_reg = 0
        if self.optimize_query:
            reg_ratio = min(self.current_epoch, self.max_query_iters) / self.max_query_iters
            reg_up = np.log(self.max_lambda)
            reg_low = np.log(self.min_lambda)
            max_reg = np.exp(reg_up - reg_ratio * (reg_up - reg_low))

            Q_loss = self.datagen.datagen.calculate_query(model=self.ncm, tau=self.use_tau, m=10000,
                                                  evaluating=False, maximize=self.maximize)
            Q_loss = Q_loss * max_reg
            q_loss_record = Q_loss.item()
            self.manual_backward(Q_loss)

        if ((batch_idx + 1) % self.grad_acc) == 0:
            G_opt.step()
            PU_opt.step()

        self.ncm.f.zero_grad()
        self.disc.zero_grad()
        self.ncm.pu.zero_grad()

        # logging
        if (self.current_epoch + 1) % 2 == 0 or True:
            if not self.logged:
                # self.logged = True

                if self.img_query:
                    sample = self(n=64)
                    for v in self.v_type:
                        self.img_lists[v].append(sample[v].detach().cpu())
                else:
                    q_estimate = self.datagen.datagen.calculate_query(
                        model=self.ncm, tau=self.use_tau, m=100000,
                        evaluating=True, log=self.wandb
                    )
                    q_true = self.datagen.datagen.calculate_query(
                        model=None, tau=self.use_tau, m=100000,
                        evaluating=True, log=self.wandb
                    )
                    if not isinstance(q_true, (list, tuple)):
                        q_true = [q_true]
                    if not isinstance(q_estimate, (list, tuple)):
                        q_estimate = [q_estimate]


                    errors = [abs(qe - qt) for qe, qt in zip(q_estimate, q_true)]
                    if batch_idx == 0:
                        print("\nQuery results:")
                        for i, (qt, qe, err) in enumerate(zip(q_true, q_estimate, errors), start=1):
                            print(f"  Query {i}: truth = {qt:.6f}, estimate = {qe:.6f}, error = {err:.6f}")

                    for i, (qt, qe, err) in enumerate(zip(q_true, q_estimate, errors), start=1):
                        self.log(f"q{i}_truth", qt)
                        self.log(f"q{i}_estimate", qe)
                        self.log(f"q{i}_error", err)
                        if self.wandb:
                            import wandb
                            wandb.log({
                                f"q{i}_truth": qt,
                                f"q{i}_estimate": qe,
                                f"q{i}_error": err,
                            })

                    errs = np.array(errors, dtype=float)
                    mae  = errs.mean()
                    rmse = np.sqrt((errs**2).mean())
                    maxe = errs.max()
                    if batch_idx == 0:
                        print(f"  MAE  = {mae:.6f}")
                        print(f"  RMSE = {rmse:.6f}")
                        print(f"  MaxE = {maxe:.6f}")

                    self.log("mae", mae)
                    self.log("rmse", rmse)
                    self.log("max_error", maxe)
                    if self.wandb:
                        import wandb
                        wandb.log({
                            "mae": mae,
                            "rmse": rmse,
                            "max_error": maxe,
                        })

                    # samples = self(n=10000, evaluating=True)
                    # print(probability_table(dat=samples))


                # big_samp_size = self.ncm_batch_size
                # big_sample = self(n=self.ncm_batch_size)
                # eval_samples = {k: v for (k, v) in big_sample.items() if self.v_type['k'] != sdt.IMAGE}
                # while big_samp_size < 10000:
                #     big_samp_size += self.ncm_batch_size
                #     big_sample = self(n=self.ncm_batch_size)
                #     for k in eval_samples:
                #         eval_samples[k] = T.concat((eval_samples[k], big_sample[k]), dim=0)
                #     fake_prob_table = probability_table(dat=eval_samples)
                #     self.stored_kl = kl(self.dat_prob_table, fake_prob_table)

        else:
            self.logged = False

        self.log('train_loss', self.stored_loss, prog_bar=True)
        self.log('G_loss', g_loss_record, prog_bar=True)
        self.log('D_loss', total_d_loss, prog_bar=True)
        if self.optimize_query:
            self.log('Q_loss', q_loss_record, prog_bar=True)

        if self.wandb:
            import wandb
            wandb.log({
                "train-epoch": self.current_epoch,
                "train-loss":  self.stored_loss,
                "G-loss":      g_loss_record,
                "D-loss":      total_d_loss,
                "Q-loss":      q_loss_record if self.optimize_query else None
            })


    # @T.no_grad()
    # def on_train_epoch_end(self):
    #     n = self.eval_samples

    #     # --- Intervene: SES = 1, X = 1 ---
    #     ses = T.ones((n, 1), dtype=T.float32).to(self.device)
    #     x   = T.ones((n, 1), dtype=T.float32).to(self.device)

    #     # --- Intervene: EMB ~ label 1 → internally mapped to label 5 in sampler ---
    #     emb_vectors = T.stack(sampler([1] * n), dim=0).to(self.device)  # list of Tensors → (n, D)


    #     # --- Forward pass under intervention ---
    #     data = self.forward(n=n, evaluating=True)
    #     data = self.forward(n=n, do={"EMB": emb_vectors, "SES": ses, "X": x}, evaluating=True)
        
    #     # --- Estimate Y and compare to ground truth ---
    #     ground_truth = 0.9  # adjust this if you have the correct expectation
    #     estimate = (data["Y"] > 0).float().mean(dim=0).item()
    #     error = np.abs(ground_truth - estimate)

    #     print(f"do-(SES=1,X=1,EMB~label1)-estimate: {estimate}")
    #     print(f"do-(SES=1,X=1,EMB~label1)-error: {error}")

    #     # --- Logging ---
    #     self.log("do-(SES=1,X=1,EMB~label1)-estimate", estimate)
    #     self.log("do-(SES=1,X=1,EMB~label1)-error", error)

    #     if self.wandb:
    #         wandb.log({
    #             "do-(SES=1,X=1,EMB~label1)-estimate": estimate,
    #             "do-(SES=1,X=1,EMB~label1)-error": error,
    #         })
