import numpy as np
import random
import torch as T
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import matplotlib as mpl
import torchvision.utils as vutils
from src.datagen.scm_datagen import SCMDataGenerator
from src.datagen.scm_datagen import SCMDataTypes as sdt
from src.scm.scm import check_equal
from src.ds import CTF, CTFTerm
from src.metric.visualization import show_image_grid
import numpy as np
import wandb

def expand_do(val, n):
    return np.ones(n, dtype=int) * val

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
            if isinstance(lbl, T.Tensor):
                lbl = lbl.item()
            lbl += 4
            if lbl not in self.indices_by_label:
                raise KeyError(f"Unknown label {lbl}")
            idxs = self.indices_by_label[lbl]
            choice = np.random.choice(idxs)
            emb = self.embeddings[choice]        # numpy array [D]
            emb_t = T.from_numpy(emb).to(self.device)
            out.append(emb_t)
        return out

def sample_binary_2d(probs: list[float], num_samples: int, random_state: int = None) -> np.ndarray:
    """
    Draw samples of a 2D binary variable (X, Y) ∈ {0,1}×{0,1}.
    
    Args:
        probs: length-4 list of probabilities for [(0,0), (0,1), (1,0), (1,1)];
               sum(probs) must be 1.
        num_samples: number of i.i.d. samples to draw.
        random_state: optional seed for reproducibility.
    
    Returns:
        samples: numpy array of shape (num_samples, 2), where each row is [x, y].
    """
    probs = np.asarray(probs, dtype=float)
    if probs.shape != (4,):
        raise ValueError("`probs` must be a list/array of length 4.")
    if not np.isclose(probs.sum(), 1.0):
        raise ValueError("`probs` must sum to 1.")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    # We index categories 0→(0,0), 1→(0,1), 2→(1,0), 3→(1,1)
    categories = np.arange(4)
    draws = np.random.choice(categories, size=num_samples, p=probs)
    
    # Decode the integer code to (x,y):
    x = draws // 2         # 0 or 1
    y = draws % 2          # 0 or 1
    
    return np.stack([x, y], axis=1)


class HAM10000DataGenerator(SCMDataGenerator):
    def __init__(self, image_size=None, mode=None, evaluating=False, normalize=None, device=None):
        super().__init__(mode)
        
        self.device = device or T.device('cpu')

        self.evaluating = evaluating
        
        self.v_size = {
            'X': 2,
            'EMB': 512,
            'SES': 2,
            'Y': 2
        }
        self.v_type = {
            'X': sdt.ONE_HOT,
            'EMB': sdt.REAL,
            'SES': sdt.ONE_HOT,
            'Y': sdt.ONE_HOT
        }
        self.cg = "ham10000"

        self.sampler = EmbeddingSampler("dat/HAM10000/embeddings_labels.npz", 'cpu')

    def _sample_exogenous(self, n: int):
        """
        Draws all exogenous randomness:
         - emb_ses bits for EMB/SES
         - ses_x bits for SES/X
         - uniform u for Y
         - sampler indices for actual EMB tensor
        """
        # 00 01 10 11
        probs0 = [0.45, 0.05, 0.45, 0.05]
        probs1 = [0.45, 0.45, 0.05, 0.05]

        emb_ses = sample_binary_2d(probs0, n)    # shape (n,2)
        ses_x   = sample_binary_2d(probs1, n)    # shape (n,2)
        u       = np.random.rand(n)              # for Y thresholds

        # pre‐sample exact embeddings indices so EMB is exogenous
        sampler_idx = []
        for bit in emb_ses[:,0]:
            label = int(bit) + 4   # your labels 4/5
            idxs  = self.sampler.indices_by_label[label]
            sampler_idx.append(np.random.choice(idxs))
        sampler_idx = np.array(sampler_idx)

        return {
            'emb_ses': emb_ses,
            'ses_x':   ses_x,
            'u':       u,
            'idx':     sampler_idx
        }


    def _compute_from_exogenous(self, exog: dict, do: dict = None):
        """
        Given exogenous noise + optional do-intervention, compute:
        EMB tensor, SES one-hot, X one-hot, Y one-hot
        """
        do = do or {}

        emb_ses = exog['emb_ses']
        ses_x   = exog['ses_x']
        u       = exog['u']
        idx     = exog['idx']

        # 1) structural equations (raw bits 0/1)
        emb_bits = emb_ses[:, 0].astype(int)
        ses_bits = np.logical_or(emb_ses[:, 1], ses_x[:, 0]).astype(int)
        x_bits   = ses_x[:, 1].astype(int)

        # 2) interventions on SES/X override those bits
        if 'SES' in do:
            # do['SES'] is one-hot, pick the class index
            ses_bits = do['SES'].argmax(dim=1).cpu().numpy().astype(int)
        if 'X' in do:
            x_bits   = do['X'].argmax(dim=1).cpu().numpy().astype(int)

        # 3) intervention on EMB class
        if 'EMB' in do:
            cls_arr    = do['EMB'].squeeze().cpu().numpy().astype(int)
            emb_bits   = cls_arr
            emb_tensor = T.stack(self.sampler(cls_arr.tolist()), dim=0).to(self.device)
        else:
            emb_np     = self.sampler.embeddings[idx]   # shape (n, D)
            emb_tensor = T.from_numpy(emb_np).to(self.device)

        # 4) compute Y bits via threshold SCM
        thresh = {
            (0,0,0): 0.9,  (0,0,1): 0.85,
            (0,1,0): 0.95, (0,1,1): 0.90,
            (1,0,0): 0.10, (1,0,1): 0.7,
            (1,1,0): 0.20, (1,1,1): 0.75,
        }
        y_bits = np.array([
            int(u[i] <= thresh[(emb_bits[i], x_bits[i], ses_bits[i])])
            for i in range(len(u))
        ], dtype=int)

        # 5) one-hot encode SES, X, Y
        ses_oh = F.one_hot(T.from_numpy(ses_bits).long(), num_classes=2).float().to(self.device)
        x_oh   = F.one_hot(T.from_numpy(x_bits).long(),   num_classes=2).float().to(self.device)
        y_oh   = F.one_hot(T.from_numpy(y_bits).long(),   num_classes=2).float().to(self.device)

        return {'EMB': emb_tensor, 'SES': ses_oh, 'X': x_oh, 'Y': y_oh}

    def generate_samples(self, n: int, obs: dict = None, do: dict = None):
        """
        Unified data generator with one-hot SES/X/Y.
        Args:
            n (int)
            obs: may contain 'SES' and/or 'X' as one-hot Tensors (shape (1,2) or (n,2))
            do:  may contain 'SES','X' (one-hot) or 'EMB' (embeddings)
        """
        obs = obs or {}
        do  = do  or {}

        # 1) If obs is provided, filter exogenous noise until we have n matches
        if obs:
            if 'EMB' in obs:
                raise ValueError("Cannot observe 'EMB'.")

            max_bs     = 5 * n
            exog_match = []

            while len(exog_match) < n:
                exog  = self._sample_exogenous(max_bs)
                batch = self._compute_from_exogenous(exog, do=do)

                # Precompute obs‐class labels for this batch
                obs_labels = {}
                for key, tensor in obs.items():
                    if key not in ('SES','X'):
                        raise ValueError("Can only observe SES or X.")
                    arr = tensor.squeeze().cpu().numpy()
                    # arr shape: (1,2) or (n,2)
                    cls = np.argmax(arr, axis=1)                  # length = 1 or n
                    if cls.size == 1:
                        cls = np.repeat(cls, max_bs)              # broadcast scalar to batch
                    obs_labels[key] = cls                        # size = max_bs

                # Build mask by comparing argmax(batch[key]) to obs_labels[key]
                mask = np.ones(max_bs, dtype=bool)
                for key, labels in obs_labels.items():
                    vals = batch[key].squeeze().cpu().numpy()    # shape (max_bs,2)
                    vals_cls = np.argmax(vals, axis=1)           # shape (max_bs,)
                    mask &= (vals_cls == labels)

                idxs = np.nonzero(mask)[0]
                for i in idxs:
                    exog_match.append({k: v[i] for k, v in exog.items()})
                    if len(exog_match) == n:
                        break

                if not idxs.size:
                    raise ValueError("No samples match the given observations.")

            # rebuild exogenous array for the matched indices
            exog = {
                k: np.stack([em[k] for em in exog_match], axis=0)
                for k in exog_match[0]
            }

        else:
            # no obs filtering, sample exactly n exogenous draws
            exog = self._sample_exogenous(n)

        # 2) Compute all variables under intervention `do`
        data = self._compute_from_exogenous(exog, do=do)

        # 3) Move to CPU
        return {k: v.cpu() for k, v in data.items()}

    
    def _sample_model_with_obs_do(self, model, n, do: dict, obs: dict = None, evaluating: bool = False):
        """
        Draw n valid Y samples from the SCM under intervention `do`,
        filtering each batch by `obs`. Repeats up to 10*n times.
        Returns a tensor of shape (n,1).
        """
        max_iters = 10 * n
        matched = []
        iters = 0

        while len(matched) < n and iters < max_iters:
            # 1) generate a batch under do
            batch = model.forward(n=n, do=do, evaluating=evaluating)

            # 2) build mask for obs
            if obs:
                mask = T.ones(n, dtype=T.bool, device=self.device)
                for key, val in obs.items():
                    if key not in ('SES','X'):
                        raise ValueError(f"Cannot observe '{key}' in model.forward")
                    # compare scalar or batch: take first element of val
                    target = val.view(-1)[0]
                    mask &= (batch[key].view(-1) == target)
                idxs = mask.nonzero(as_tuple=False).view(-1)
            else:
                idxs = T.arange(n, device=self.device)

            # 3) collect matched Ys
            for i in idxs.tolist():
                matched.append(batch['Y'][i])
                if len(matched) >= n:
                    break

            iters += 1

        if len(matched) < n:
            raise RuntimeError(f"Only found {len(matched)}/{n} matches after {iters} iterations")

        # stack into shape (n,1)
        return T.stack(matched[:n], dim=0)

    def calculate_query(self, model, tau, m, evaluating, log=False):
        """
        Compute flexible causal queries under one-hot encoding.
        Returns list of P(Y=1) for each (obs,do) triple in `queries`.
        """

        m = 5000

        # 1) Build one-hot intervention vectors for SES and X
        #    class 0 → [1,0], class 1 → [0,1]
        labels0 = T.zeros(m, dtype=T.long, device=self.device)
        labels1 = T.ones(m,  dtype=T.long, device=self.device)

        ses0 = F.one_hot(labels0, num_classes=2).float()  # (m,2)
        ses1 = F.one_hot(labels1, num_classes=2).float()
        x0   = F.one_hot(labels0, num_classes=2).float()
        x1   = F.one_hot(labels1, num_classes=2).float()

        # 2) If using the model, also sample EMB embeddings here

        embs0 = labels0
        embs1 = labels1

        if model is not None:
            embs0 = T.stack(self.sampler([0]*m), dim=0).to(self.device)
            embs1 = T.stack(self.sampler([1]*m), dim=0).to(self.device)

        # 3) Declare your queries as (name, obs, do)
        queries = [
            # name,             obs dict,                 do dict
            ("P(Y=1|do-(EMB=1, X=0))", {},                       {"X": x0, "EMB": embs1}),
            ("P(Y=1|do-(EMB=1, X=1))", {},                       {"X": x1, "EMB": embs1}),
            ("P(Y=1|do-(EMB=1, SES=1))", {},                       {"SES": ses1, "EMB": embs1}),
        ]

        estimates = []
        for name, obs, do in queries:
            # 4) Sample Y under model or fallback
            if model is not None:
                y_oh = self._sample_model_with_obs_do(
                    model=model, n=m, do=do, obs=obs, evaluating=evaluating
                )  # returns (m,2) one-hot for Y
            else:
                data = self.generate_samples(m, obs=obs, do=do)
                y_oh = data["Y"].to(self.device)    # (m,2)

            # 5) P(Y=1) is the mean of the second channel
            p = float(y_oh[:,1].float().mean().item())
            estimates.append(p)

        return estimates


if __name__ == "__main__":

    mdg = HAM10000DataGenerator("sampling")

    n_samples = 4
    data = mdg.generate_samples(n_samples)
    
    print(data)