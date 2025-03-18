import wandb
import torch as T
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from src.scm.repr_nn.representation_nn import RepresentationalNN
from src.scm.ncm import GAN_NCM
from src.datagen.scm_datagen import SCMDataTypes as sdt

from .base_pipeline import BasePipeline
import warnings
import torchvision.transforms as transforms

def log(x):
    return T.log(x + 1e-8)

def nt_xent_loss(batch, model, temperature=0.1):
    # todo: add projection head
    """
    Computes the NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss for a batch of images.
    Improvements:
    - Uses SimCLR-style data augmentation with optimized transformations.
    - Batch-wise augmentation for efficiency.
    - Computes the similarity matrix with masking to exclude self-similarity.
    - Uses in-place normalization for efficiency.
    - Averages loss over image keys.

    Args:
        batch (dict of torch.Tensor): Dictionary of input images [batch_size, 3, 32, 32].
        model (torch.nn.Module): Encoder model mapping images to representations.
        temperature (float): Scaling parameter for contrastive loss.

    Returns:
        torch.Tensor: Averaged NT-Xent loss.
    """

    device = next(model.parameters()).device

    loss_total = 0
    num_image_keys = 0

    # Define SimCLR Augmentations
    augmentation = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0)),  # More aggressive cropping
        transforms.RandomHorizontalFlip(p=0.5),  # Standard flipping
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),  # Color distortion
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),  # Gaussian blur
    ])

    # Apply augmentation to batch: generate two different augmented views
    batch_aug1, batch_aug2 = {}, {}
    for key, imgs in batch.items():
        if model.v_type[key] == sdt.IMAGE:
            imgs = imgs.to(device)
            batch_aug1[key] = T.stack([augmentation(img) for img in imgs])
            batch_aug2[key] = T.stack([augmentation(img) for img in imgs])
        else:
            batch_aug1[key] = imgs
            batch_aug2[key] = imgs

    # Concatenate both augmented views along the batch dimension
    batch_aug = {key: T.cat([batch_aug1[key], batch_aug2[key]], dim=0).to(device) for key in batch.keys()}

    # Forward pass through the encoder model
    z = model(batch_aug, projection=True)

    for key in z.keys():
        if model.v_type[key] == sdt.IMAGE:
            num_image_keys += 1
            batch_size = z[key].shape[0] // 2  # Since we concatenated two views

            # Normalize embeddings in-place
            z[key] = T.nn.functional.normalize(z[key], dim=1)

            # Compute cosine similarity matrix
            sim_matrix = T.mm(z[key], z[key].T) / temperature  # Shape: [2*batch_size, 2*batch_size]

            # Mask self-similarity (diagonal) by setting it to a large negative value
            mask = T.eye(2 * batch_size, dtype=T.bool, device=z[key].device)
            sim_matrix.masked_fill_(mask, -float('inf'))

            # Construct labels: each sample should be closest to its corresponding augmented version
            labels = T.cat([T.arange(batch_size, 2 * batch_size),
                            T.arange(0, batch_size)]).to(z[key].device)

            # Compute NT-Xent loss
            loss_key = T.nn.functional.cross_entropy(sim_matrix, labels)
            loss_total += loss_key

    # Average loss over the image keys
    return loss_total / num_image_keys if num_image_keys > 0 else loss_total

class RepresentationalPipeline(BasePipeline):
    def __init__(self, datagen, cg, v_size, v_type, hyperparams=None, repr_model_type=RepresentationalNN):
        if hyperparams is None:
            hyperparams = dict()

        super().__init__(datagen, cg, None, batch_size=hyperparams.get('rep-bs', 1000))
        self.automatic_optimization = False

        self.datagen = datagen
        self.cg = cg

        self.v_size = v_size
        self.v_type = v_type
        self.img_size = hyperparams["img-size"]
        self.rep_size = hyperparams["rep-size"]
        self.model = repr_model_type(cg, v_size, v_type, hyperparams=hyperparams)
        self.automatic_optimization = False

        self.batch_size = hyperparams["rep-bs"]
        self.grad_acc = hyperparams["rep-grad-acc"]
        self.lr = hyperparams["rep-lr"]
        self.train = hyperparams["rep-train"]
        self.pred_parents = hyperparams['rep-pred-parents']
        self.reconstruct = hyperparams['rep-reconstruct']
        self.unsup_contrastive = hyperparams['rep-unsup-contrastive']
        self.sup_contrastive = hyperparams['rep-sup-contrastive']
        self.classify_lambda = hyperparams['rep-class-lambda']
        self.contrast_lambda = hyperparams['rep-contrast-lambda']
        self.temperature = hyperparams["rep-temperature"]
        self.wandb = hyperparams["wandb"]
        self.loss = nn.MSELoss()
        self.classify_loss = nn.BCELoss()

        if hyperparams["verbose"]:
            print("ENCODER")
            print(self.model.encoders)
            print("DECODER")
            print(self.model.decoders)
            if self.pred_parents:
                print("PARENT HEADS")
                print(self.model.parent_heads)

    def configure_optimizers(self):
        opt_enc = T.optim.Adam(self.model.encoders.parameters(), lr=self.lr)
        opt_dec = T.optim.Adam(self.model.decoders.parameters(), lr=self.lr)
        opt_head = None if not self.pred_parents else T.optim.Adam(self.model.parent_heads.parameters(), lr=self.lr)
        opt_proj = None if not self.unsup_contrastive else T.optim.Adam(self.model.proj_heads.parameters(), lr=self.lr)
 
        if self.pred_parents and self.unsup_contrastive:
            return opt_enc, opt_dec, opt_head, opt_proj
        elif self.pred_parents:
            return opt_enc, opt_dec, opt_head
        elif self.unsup_contrastive:
            return opt_enc, opt_dec, opt_proj
        else:
            return opt_enc, opt_dec

    def _get_loss(self, loss, out, data):
        total = 0
        for v in out:
            total += loss(out[v], data[v])
        return total

    def training_step(self, batch, batch_idx):

        if not self.train:
            warnings.warn("No component is training", UserWarning)

        if self.pred_parents and self.unsup_contrastive:
            opt_enc, opt_dec, opt_head, opt_proj = self.optimizers()
        elif self.pred_parents:
            opt_enc, opt_dec, opt_head = self.optimizers()
        elif self.unsup_contrastive:
            opt_enc, opt_dec, opt_proj = self.optimizers()
        else:
            opt_enc, opt_dec = self.optimizers()

        opt_enc.zero_grad()
        opt_dec.zero_grad()

        contrast_loss_items = 0

        loss_pred_parents = 0
        loss_pred_parents_log = 0
        loss_unsup_contrastive = 0
        loss_unsup_contrastive_log = 0
        loss_sup_contrastive = 0
        loss_sup_contrastive_log = 0
        loss_reconstruct = 0
        loss_reconstruct_log = 0

        # TODO: we have multiple forwardpasses in this step. might want to consolidate some of them

        if self.pred_parents:
            opt_head.zero_grad()
            out_batch, label_out, label_truth = self.model(batch, classify=True)
            loss_pred_parents = self.classify_lambda * self._get_loss(self.classify_loss, label_out, label_truth)
            loss_pred_parents_log = loss_pred_parents.item()

        if self.unsup_contrastive:
            opt_proj.zero_grad()
            loss_unsup_contrastive = nt_xent_loss(batch, self.model, self.temperature)
            loss_unsup_contrastive_log = loss_unsup_contrastive.item()

        if self.sup_contrastive:
            enc = self.model.encode(batch)
            for v in batch:
                if v in self.model.encode_v:
                    rep_v = enc[v]
                    label_rep_size = self.rep_size // len(self.cg.pa[v])
                    pa_v = [x for x in self.cg.v if x in self.cg.pa[v]]
                    for i, pa in enumerate(pa_v):
                        labels = batch[pa]
                        if labels.shape[1] == 1:
                            labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
                        else:
                            raise NotImplementedError("Higher dimensional labels not yet supported.")

                        rep_pa = rep_v[:, label_rep_size * i:label_rep_size * (i + 1)]
                        similarity_matrix = T.matmul(rep_pa, rep_pa.T) / self.temperature

                        mask = T.eye(labels.shape[0], dtype=T.bool).to(self.device)
                        labels = labels[~mask].view(labels.shape[0], -1)
                        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

                        loss_vals = T.softmax(similarity_matrix, dim=1)
                        loss_vals = T.sum(loss_vals * labels, dim=1) / (2 * T.sum(labels, dim=1) + 1e-8)
                        loss_sup_contrastive += T.mean(-log(loss_vals))
                        contrast_loss_items += 1

            loss_sup_contrastive = self.contrast_lambda * (loss_sup_contrastive / contrast_loss_items)
            loss_sup_contrastive_log = loss_sup_contrastive.item()
            out_batch = self.model.decode(enc)

        if self.reconstruct:
            out_batch = self.model(batch)
            loss_reconstruct = self._get_loss(self.loss, out_batch, batch)
            loss_reconstruct_log = loss_reconstruct.item()

        loss = loss_reconstruct + loss_pred_parents + loss_unsup_contrastive + loss_sup_contrastive
        
        self.manual_backward(loss)

        if ((batch_idx + 1) % self.grad_acc) == 0:
            if self.train:
                opt_enc.step()
            if self.pred_parents:
                opt_head.step()
            if self.sup_contrastive:
                opt_proj.step()
            if self.reconstruct or self.sup_contrastive:
                opt_dec.step()

        # logging
        self.log('train_loss', loss.item(), prog_bar=True)
        if self.reconstruct:
            self.log('reconstruction_loss', loss_reconstruct_log, prog_bar=True)
        if self.pred_parents:
            self.log('pred_parents_loss', loss_pred_parents_log, prog_bar=True)
        if self.sup_contrastive:
            self.log('sup_contrastive_loss', loss_sup_contrastive_log, prog_bar=True)
        if self.unsup_contrastive:
            self.log('unsup_contrastive_loss', loss_unsup_contrastive_log, prog_bar=True)

        if self.wandb:
            wandb.log({
                "rep-train-loss": loss.item(),
                "rep-reconstruction-loss": loss_reconstruct_log if self.reconstruct else None,
                "rep-pred-parents-loss": loss_pred_parents_log if self.pred_parents else None,
                "rep-sup-contrastive-loss": loss_sup_contrastive_log if self.sup_contrastive else None,
                "rep-unsup-contrastive-loss": loss_unsup_contrastive_log if self.unsup_contrastive else None
            })