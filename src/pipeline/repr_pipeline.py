import wandb
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from src.scm.repr_nn.representation_nn import RepresentationalNN
from src.scm.ncm import GAN_NCM
from src.datagen.scm_datagen import SCMDataTypes as sdt

from .base_pipeline import BasePipeline
import warnings
import torchvision.transforms as transforms

import numpy as np
import os

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

import warnings
from sklearn.exceptions import ConvergenceWarning

def log(x):
    return T.log(x + 1e-8)

def nt_xent_loss(batch, model, temperature=0.07):
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

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss as introduced in:
    "Supervised Contrastive Learning" (Khosla et al., NeurIPS 2020).

    Extended to support multi-dimensional labels via composite losses.
    If labels is 1D (shape [batch_size]), it behaves as before.
    If labels is 2D (shape [batch_size, label_dim]), it computes a separate
    loss for each label dimension and averages them.
    """
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels=None):
        """
        Args:
            features (Tensor): Feature representations with shape
                [batch_size, n_views, feature_dim] if multiple views per sample are provided,
                or [batch_size, feature_dim] for a single view (a singleton dimension is added).
            labels (Tensor): Ground-truth labels. For a standard setup, a 1D tensor of shape [batch_size].
                For multi-dimensional labels, a tensor of shape [batch_size, label_dim].
                These are used to determine positive pairs.

        Returns:
            Tensor: The computed loss (scalar).
        """
        device = features.device

        # Ensure features has three dimensions: [batch_size, n_views, feature_dim]
        if features.dim() < 3:
            features = features.unsqueeze(1)

        # Check if feature dimensions might be swapped.
        if features.size(1) > features.size(2):
            features = features.transpose(1, 2)

        batch_size, n_views, _ = features.shape

        # Normalize feature vectors.
        features = F.normalize(features, p=2, dim=-1)
        # Combine all views into one tensor: [batch_size * n_views, feature_dim]
        contrast_features = T.cat(T.unbind(features, dim=1), dim=0)

        # Compute pairwise dot-product similarity scaled by temperature.
        anchor_dot_contrast = T.div(
            T.matmul(contrast_features, contrast_features.T),
            self.temperature
        )

        # For numerical stability, subtract the max value in each row.
        logits_max, _ = T.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Create mask to remove self-comparisons.
        total_samples = batch_size * n_views
        # Create a mask with ones and then set the diagonal to zero.
        logits_mask = T.ones((total_samples, total_samples), device=device)
        logits_mask = logits_mask.scatter(1, T.arange(total_samples, device=device).view(-1, 1), 0)

        # Compute log probabilities.
        exp_logits = T.exp(logits) * logits_mask
        log_prob = logits - T.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Helper to compute loss given a mask.
        def compute_loss(mask):
            # Zero out self-contrast cases.
            mask = mask * logits_mask
            # Compute mean log-likelihood over positive pairs.
            mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
            loss_val = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            return loss_val.mean()

        # If labels are provided, build the mask(s) accordingly.
        if labels is not None:
            # Case 1: 1D labels (standard scenario)
            if labels.dim() == 1:
                labels = labels.repeat(1, n_views).flatten()
                mask = T.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(device)
                loss = compute_loss(mask)
            # Case 2: Multi-dimensional labels
            else:
                composite_loss = 0.0
                num_dims = labels.size(1)
                for i in range(num_dims):
                    # Extract the i-th label dimension: shape [batch_size]
                    label_i = labels[:, i]
                    # Expand to match the views.
                    label_i = label_i.repeat(1, n_views).flatten()
                    # Build mask: positive if the labels match exactly for this dimension.
                    mask_i = T.eq(label_i.unsqueeze(1), label_i.unsqueeze(0)).float().to(device)
                    composite_loss += compute_loss(mask_i)
                loss = composite_loss / num_dims
        else:
            # Unsupervised case: use identity matrix as mask.
            mask = T.eye(total_samples, device=device)
            loss = compute_loss(mask)

        return loss

def supervised_contrastive_loss(embeddings, labels, temperature=0.07):
    """
    Compute the supervised contrastive loss.

    Args:
        embeddings (Tensor): shape [N, D] where N is number of samples and D is feature dim.
        labels (Tensor): shape [N] ground truth labels.
        temperature (float): scaling factor for the cosine similarities.

    Returns:
        loss (Tensor): scalar loss.
    """
    # Normalize embeddings to unit vectors.
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Compute pairwise cosine similarity scaled by temperature.
    similarity_matrix = T.matmul(embeddings, embeddings.T) / temperature
    
    # Create a mask to remove self-similarity (diagonal elements).
    diag_mask = T.eye(similarity_matrix.size(0), device=embeddings.device, dtype=T.bool)
    similarity_matrix.masked_fill_(diag_mask, -1e9)
    
    # Compute log probabilities.
    exp_sim = T.exp(similarity_matrix)
    log_prob = similarity_matrix - T.log(exp_sim.sum(dim=1, keepdim=True))
    
    # Build a mask for positives: samples with the same label.
    labels = labels.unsqueeze(1)  # Shape: [N, 1]
    positive_mask = T.eq(labels, labels.T).float()
    positive_mask.masked_fill_(diag_mask, 0)
    
    # Compute the loss for each sample and take the mean.
    loss = -(positive_mask * log_prob).sum(dim=1) / (positive_mask.sum(dim=1) + 1e-8)
    return loss.mean()

def compute_contrastive_loss(label_out, label_truth, temperature=0.07):
    """
    Separates predictions into correct and wrong sets.
    For wrong predictions, computes the supervised contrastive loss.

    Args:
        label_out (Tensor): Tensor of shape [batch_size, num_classes] with logits or features.
        label_truth (Tensor): Tensor of shape [batch_size] with ground-truth labels.
        temperature (float): Temperature hyperparameter for the loss.

    Returns:
        contrastive_loss (Tensor or None): Loss computed on wrong predictions; None if there are none.
        correct_indices (Tensor): 1D tensor with indices of correct predictions.
        wrong_indices (Tensor): 1D tensor with indices of mispredictions.
    """
    # Compute predicted classes.
    preds = label_out.argmax(dim=1)
    correct_mask = preds.eq(label_truth)
    wrong_mask = ~correct_mask

    # Get indices for correct and wrong predictions.
    correct_indices = correct_mask.nonzero(as_tuple=True)[0]
    wrong_indices = wrong_mask.nonzero(as_tuple=True)[0]

    contrastive_loss = 0
    if wrong_indices.numel() > 0:
        # Select only the mispredicted samples.
        wrong_embeddings = label_out[wrong_indices]  # Shape: [n_wrong, D]
        wrong_labels = label_truth[wrong_indices]      # Shape: [n_wrong]
        contrastive_loss = supervised_contrastive_loss(wrong_embeddings, wrong_labels, temperature)
        
    return contrastive_loss

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
        self.pred_parents_contrast = hyperparams['rep-pred-parents-contr']
        self.reconstruct = hyperparams['rep-reconstruct']
        self.unsup_contrastive = hyperparams['rep-unsup-contrastive']
        self.sup_contrastive = hyperparams['rep-sup-contrastive']
        self.classify_lambda = hyperparams['rep-class-lambda']
        self.contrast_lambda = hyperparams['rep-contrast-lambda']
        self.temperature = hyperparams["rep-temperature"]
        self.wandb = hyperparams["wandb"]
        self.loss = nn.MSELoss()
        self.classify_loss = nn.BCELoss()
        # self.classify_loss = nn.CrossEntropyLoss()
        self.con_loss = SupConLoss(temperature=self.temperature)

        if hyperparams["verbose"]:
            print("ENCODER")
            print(self.model.encoders)
            print("DECODER")
            print(self.model.decoders)
            if self.pred_parents:
                print("PARENT HEADS")
                print(self.model.parent_heads)
            
        self.detailed_logging = hyperparams["detailed-logging"]
        
        # ignore convergence warnings from sklearn
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

    def configure_optimizers(self):
        opt_enc = T.optim.Adam(self.model.encoders.parameters(), lr=self.lr)
        opt_dec = T.optim.Adam(self.model.decoders.parameters(), lr=self.lr)
        opt_head = None if not self.pred_parents else T.optim.Adam(self.model.parent_heads.parameters(), lr=self.lr)
        opt_proj = None if not (self.unsup_contrastive or self.sup_contrastive) else T.optim.Adam(self.model.proj_heads.parameters(), lr=self.lr)
 
        if self.pred_parents and self.unsup_contrastive:
            return opt_enc, opt_dec, opt_head, opt_proj
        elif self.pred_parents and self.sup_contrastive:
            return opt_enc, opt_dec, opt_head, opt_proj
        elif self.pred_parents:
            return opt_enc, opt_dec, opt_head
        elif self.unsup_contrastive or self.sup_contrastive:
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
        if self.pred_parents and self.sup_contrastive:
            opt_enc, opt_dec, opt_head, opt_proj = self.optimizers()
        elif self.pred_parents:
            opt_enc, opt_dec, opt_head = self.optimizers()
        elif self.unsup_contrastive or self.sup_contrastive:
            opt_enc, opt_dec, opt_proj = self.optimizers()
        else:
            opt_enc, opt_dec = self.optimizers()

        opt_enc.zero_grad()
        opt_dec.zero_grad()

        loss_pred_parents = 0
        loss_pred_parents_log = 0
        loss_pred_parents_contrast_log = 0
        loss_unsup_contrastive = 0
        loss_unsup_contrastive_log = 0
        loss_sup_contrastive = 0
        loss_sup_contrastive_log = 0
        loss_reconstruct = 0
        loss_reconstruct_log = 0

        # TODO: we have multiple forward passes in this step. might want to consolidate some of them

        if self.pred_parents:
            opt_head.zero_grad()
            out_batch, label_out, label_truth = self.model(batch, classify=True)

            loss_pred_parents = self.classify_lambda * self._get_loss(self.classify_loss, label_out, label_truth)
            loss_pred_parents_log = loss_pred_parents.item()

            if self.pred_parents_contrast:

                for k in label_out.keys():
                    # TODO: need to fix this manual argmax, might be unnecessary in some cases
                    label_truth[k] = T.argmax(label_truth[k], dim=-1)
                    loss_contrastive = compute_contrastive_loss(label_out[k], label_truth[k], self.temperature)
                    loss_pred_parents_contrast_log += loss_contrastive.item()
                    loss_pred_parents += loss_contrastive

        if self.unsup_contrastive:
            opt_proj.zero_grad()
            loss_unsup_contrastive = nt_xent_loss(batch, self.model, self.temperature)
            loss_unsup_contrastive_log = loss_unsup_contrastive.item()

        if self.sup_contrastive:
            opt_proj.zero_grad()
            labels = dict()
            features = dict()
            enc_batch = self.model.forward(batch, projection=True)
            
            for v in batch:
                if v in self.model.encode_v:
                    pa_list = []
                    for x in self.cg.pa[v]:
                        if self.v_type[x] == sdt.BINARY_ONES:
                            pa_list.append(((batch[x] + 1) / 2).unsqueeze(dim=1))  # Ensure shape [batch_size, 1]
                        elif self.v_type[x] == sdt.BINARY:
                            pa_list.append(batch[x].unsqueeze(dim=1))  # Ensure shape [batch_size, 1]
                        elif self.v_type[x] == sdt.ONE_HOT:
                            pa_list.append(T.argmax(batch[x], dim=-1, keepdim=True))  # Shape [batch_size, 1]

                    features[v] = enc_batch[v]
                    labels[v] = T.cat(pa_list, dim=1)
            
            for k in features.keys():
                loss_sup_contrastive += self.con_loss(features[k], labels[k])
                loss_sup_contrastive_log = loss_sup_contrastive.item()

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
            if self.unsup_contrastive:
                opt_proj.step()
            if self.reconstruct or self.sup_contrastive:
                opt_dec.step()

        # logging
        self.log('train_loss', loss.item(), prog_bar=True, on_step=True, on_epoch=False)
        if self.reconstruct:
            self.log('reconstruction_loss', loss_reconstruct_log, prog_bar=True, on_step=True, on_epoch=False)
        if self.pred_parents:
            self.log('pred_parents_loss', loss_pred_parents_log, prog_bar=True, on_step=True, on_epoch=False)
        if self.pred_parents_contrast:
            self.log('pred_parents_contrast_loss', loss_pred_parents_contrast_log, prog_bar=True, on_step=True, on_epoch=False)
        if self.sup_contrastive:
            self.log('sup_contrastive_loss', loss_sup_contrastive_log, prog_bar=True, on_step=True, on_epoch=False)
        if self.unsup_contrastive:
            self.log('unsup_contrastive_loss', loss_unsup_contrastive_log, prog_bar=True, on_step=True, on_epoch=False)

        if self.wandb:
            wandb.log({
                "rep-train-epoch": self.current_epoch,
                "rep-train-loss": loss.item(),
                "rep-reconstruction-loss": loss_reconstruct_log if self.reconstruct else None,
                "rep-pred-parents-loss": loss_pred_parents_log if self.pred_parents else None,
                "rep-pred-parents-contrast-loss": loss_pred_parents_contrast_log if self.pred_parents_contrast else None,
                "rep-sup-contrastive-loss": loss_sup_contrastive_log if self.sup_contrastive else None,
                "rep-unsup-contrastive-loss": loss_unsup_contrastive_log if self.unsup_contrastive else None
            })

    def on_train_epoch_end(self):
        if not self.detailed_logging:
            return
        
        self.model.eval()  # Set model to evaluation mode

        # Get device
        device = next(self.model.parameters()).device

        # Dictionary to store features and labels for each encoded variable
        features_dict, labels_dict = {}, {}

        # Extract embeddings using the model
        with T.no_grad():
            for batch in self.train_dataloader():
                # Move batch to device
                for key in batch:
                    batch[key] = batch[key].to(device)

                # Get encoded representations
                enc_batch = self.model.encode(batch)

                # Process each variable in self.model.encode_v
                for v in enc_batch.keys():
                    if v in self.model.encode_v:
                        pa_list = []
                        for x in self.cg.pa[v]:
                            if self.v_type[x] == sdt.BINARY_ONES:
                                pa_list.append(((enc_batch[x] + 1) / 2).unsqueeze(dim=1))  # Ensure shape [batch_size, 1]
                            elif self.v_type[x] == sdt.BINARY:
                                pa_list.append(enc_batch[x].unsqueeze(dim=1))  # Ensure shape [batch_size, 1]
                            elif self.v_type[x] == sdt.ONE_HOT:
                                pa_list.append(T.argmax(enc_batch[x], dim=-1, keepdim=True))  # Shape [batch_size, 1]

                        # Store features and labels
                        if v not in features_dict:
                            features_dict[v] = []
                            labels_dict[v] = []

                        features_dict[v].append(enc_batch[v].cpu().numpy().astype(np.float32))
                        labels_dict[v].append(T.cat(pa_list, dim=1).cpu().numpy().astype(np.int32))

        # Train a linear probe & PCA for each component in self.model.encode_v
        for v in self.model.encode_v:
            if v not in features_dict or len(features_dict[v]) == 0:
                continue  # Skip if no data

            # Convert to numpy arrays
            features = np.vstack(features_dict[v])  # Shape: [num_samples, feature_dim]
            labels = np.vstack(labels_dict[v])      # Shape: [num_samples, label_dim]

            # **Ensure labels are 1D for Logistic Regression**
            if labels.shape[1] == 1:
                labels = labels.ravel()  # Convert to (n_samples,)

            # **Standardize Features**
            scaler = StandardScaler()
            features = scaler.fit_transform(features)

            # **Step 1: Train a Linear Probe**
            clf = LogisticRegression(max_iter=100, solver="saga")
            clf.fit(features, labels)
            pred_labels = clf.predict(features)
            linear_probe_acc = accuracy_score(labels, pred_labels)

            # **Addition: Calculate per-class accuracies**
            unique_classes = np.unique(labels)
            class_accuracies = {}
            for cls in unique_classes:
                mask = labels == cls
                class_acc = accuracy_score(labels[mask], pred_labels[mask])
                class_accuracies[int(cls)] = class_acc  # Use int as key for logging consistency

            # **Step 2: Perform PCA for Visualization**
            pca = PCA(n_components=2)
            pca_features = pca.fit_transform(features)  # Reduce from high-dimensional to 2D

            # **Step 3: Plot PCA visualization**
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1], c=labels, cmap="jet", alpha=0.5)
            plt.colorbar(scatter, label="Class Label")
            plt.title(f"PCA Visualization ({v}) - Epoch {self.current_epoch}")
            plt.xlabel("PCA Component 1")
            plt.ylabel("PCA Component 2")

            # Save figure
            pca_plot_path = f"pca_{v}_epoch_{self.current_epoch}.png"
            plt.savefig(pca_plot_path)
            plt.close()

            # **Step 4: Log to Weights & Biases**
            if self.wandb:
                metrics = {f"rep-{v}/linear_probe_acc": linear_probe_acc}
                # Log per-class accuracies
                for cls, acc in class_accuracies.items():
                    metrics[f"rep-{v}/class_{cls}_acc"] = acc
                # Log PCA visualization for each component
                metrics[f"rep-{v}/PCA Visualization"] = wandb.Image(pca_plot_path)
                wandb.log(metrics)
            # Remove the saved plot
            os.remove(pca_plot_path)

        self.model.train()  # Switch back to training mode
