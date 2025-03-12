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

def nt_xent_loss(batch, model, temperature=0.5):
    """
    Computes the NT-Xent loss for a batch of images.
    
    Args:
        batch (torch.Tensor): Input images of shape [batch_size, 3, 32, 32].
        model (torch.nn.Module): Encoder model that maps images to representations.
        temperature (float): Temperature parameter for scaling.
        
    Returns:
        torch.Tensor: Computed NT-Xent loss.
    """

    loss = 0

    # Define SimCLR augmentations
    augmentation = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),  # Random crop
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),  # Color distortion
    ])
    batch_aug1 = {key: [] for key in batch.keys()}
    batch_aug2 = {key: [] for key in batch.keys()}

    for key in batch.keys():
        for idx in range(batch[key].shape[0]):
            if model.v_type[key] == sdt.IMAGE:
                img = batch[key][idx]  # Shape: [3, 32, 32]
                # Apply augmentations to get two views of each image
                batch_aug1[key].append(augmentation(img))
                batch_aug2[key].append(augmentation(img))
            else:
                batch_aug1[key].append(batch[key][idx])
                batch_aug2[key].append(batch[key][idx])

    for key in batch.keys():
        batch_aug1[key] = T.stack(batch_aug1[key], dim=0)
        batch_aug2[key] = T.stack(batch_aug2[key], dim=0)

    # Concatenate both views along batch dimension
    batch_aug = {key: T.cat([batch_aug1[key], batch_aug2[key]], dim=0) for key in batch.keys()}

    # Forward pass through encoder
    z = model.encode(batch_aug)

    # TODO: this might be bad if we have multiple images of different importance
    for key in z:
        if model.v_type[key] == sdt.IMAGE:
            z[key] = T.nn.functional.normalize(z[key], dim=1) # Normalize embeddings

            # Compute similarity matrix
            sim_matrix = T.matmul(z[key], z[key].T)  # Shape: [2*batch_size, 2*batch_size]
    
            # Scale by temperature
            sim_matrix /= temperature


            batch_size = z[key].shape[0] // 2
            # Construct labels: each image should be most similar to its augmented pair
            labels = T.cat([T.arange(batch_size) for _ in range(2)]).to(batch[key].device)
            labels = labels + (labels >= batch_size).long() * batch_size  # Ensure correct alignment

            # Compute cross-entropy loss
            loss += T.nn.functional.cross_entropy(sim_matrix, labels)

    return loss

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
        self.train_encoder = (hyperparams['repr'] != "auto_enc_notrain")
        self.train_decoder = not hyperparams['rep-no-decoder']
        self.classify = (hyperparams['repr'] == "auto_enc_conditional")
        self.classify_lambda = hyperparams['rep-class-lambda']
        self.sup_contrastive = (hyperparams['repr'] == "auto_enc_sup_contrastive")
        self.unsup_contrastive = hyperparams['rep-contrastive-loss']
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
            if self.classify:
                print("PARENT HEADS")
                print(self.model.parent_heads)

    def configure_optimizers(self):
        opt_enc = T.optim.Adam(self.model.encoders.parameters(), lr=self.lr)
        opt_dec = T.optim.Adam(self.model.decoders.parameters(), lr=self.lr)
        if self.classify:
            opt_head = T.optim.Adam(self.model.parent_heads.parameters(), lr=self.lr)
            return opt_enc, opt_dec, opt_head
        return opt_enc, opt_dec

    def _get_loss(self, loss, out, data):
        total = 0
        for v in out:
            total += loss(out[v], data[v])
        return total

    def training_step(self, batch, batch_idx):

        if not (self.train_encoder or self.train_decoder or self.classify):
            warnings.warn("No component is training", UserWarning)

        if not (self.train_encoder or self.train_decoder or self.classify):
            return

        if self.classify:
            opt_enc, opt_dec, opt_head = self.optimizers()
        else:
            opt_enc, opt_dec = self.optimizers()

        opt_enc.zero_grad()
        opt_dec.zero_grad()

        label_loss = 0
        label_loss_log = 0
        contrast_loss = 0
        contrast_loss_items = 0
        contrast_loss_log = 0
 
        if self.classify:
            opt_head.zero_grad()
            out_batch, label_out, label_truth = self.model(batch, classify=True)
            label_loss = self.classify_lambda * self._get_loss(self.classify_loss, label_out, label_truth)
            label_loss_log = label_loss.item()

        if self.unsup_contrastive:
            contrast_loss = nt_xent_loss(batch, self.model, self.temperature)
            contrast_loss_log = contrast_loss.item()

        elif self.sup_contrastive:
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
                        contrast_loss += T.mean(-log(loss_vals))
                        contrast_loss_items += 1

            contrast_loss = self.contrast_lambda * (contrast_loss / contrast_loss_items)
            contrast_loss_log = contrast_loss.item()
            out_batch = self.model.decode(enc)

        else:
            out_batch = self.model(batch)

        loss = self._get_loss(self.loss, out_batch, batch)
        
        recon_loss_log = loss.item()

        loss = loss + label_loss + contrast_loss
        self.manual_backward(loss)

        if ((batch_idx + 1) % self.grad_acc) == 0:
            if self.train_encoder:
                opt_enc.step()
            if self.classify:
                opt_head.step()
            if self.train_decoder:
                opt_dec.step()

        # logging
        self.log('train_loss', loss.item(), prog_bar=True)
        self.log('recon_loss', recon_loss_log, prog_bar=True)
        if self.classify:
            self.log('label_loss', label_loss_log, prog_bar=True)
        if self.sup_contrastive or self.unsup_contrastive:
            self.log('contr_loss', contrast_loss_log, prog_bar=True)
        
        if self.wandb:
            wandb.log({
                "train_loss": loss.item(),
                "recon_loss": recon_loss_log,
                "label_loss": label_loss_log if self.classify else None,
                "contrast_loss": contrast_loss_log if (self.sup_contrastive or self.unsup_contrastive) else None,
            })

