import numpy as np
import random
import torch as T
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
    def __init__(self, image_size=None, mode=None, evaluating=False, normalize=None):
        super().__init__(mode)

        print("mode: ", mode)

        self.evaluating = evaluating
        
        self.v_size = {
            'X': 1,
            'EMB': 512,
            'SES': 1,
            'Y': 1
        }
        self.v_type = {
            'X': sdt.BINARY_ONES,
            'EMB': sdt.REAL,
            'SES': sdt.BINARY_ONES,
            'Y': sdt.BINARY_ONES
        }
        self.cg = "ham10000"

        self.sampler = EmbeddingSampler("dat/HAM10000/embeddings_labels.npz", 'cpu')

    def generate_samples(self, n: int):
        """
        Returns a dict with:
        - 'EMB': Tensor of shape (n, D)
        - 'SES': Tensor of shape (n, 1) with values in {-1, 1}
        - 'X':   Tensor of shape (n, 1) with values in {-1, 1}
        - 'Y':   Tensor of shape (n, 1) with values in {-1, 1}
        """
        # 1) draw latent bits
        probs0 = [0.1, 0.2, 0.3, 0.4]
        probs1 = [0.4, 0.3, 0.2, 0.1]
        emb_ses = sample_binary_2d(probs0, n)   # shape (n,2)
        ses_x   = sample_binary_2d(probs1, n)   # shape (n,2)

        emb_bits = emb_ses[:, 0]
        ses_bits = np.logical_or(emb_ses[:, 1], ses_x[:, 0]).astype(int)
        x_bits   = ses_x[:, 1]

        # 2) compute y via thresholds and a fresh uniform u
        u = np.random.rand(n)
        thresh = {
            (0,0,0): 0.90, (0,0,1): 0.95,
            (0,1,0): 0.85, (0,1,1): 0.90,
            (1,0,0): 0.10, (1,0,1): 0.20,
            (1,1,0): 0.80, (1,1,1): 0.90,
        }
        y_bits = np.array([int(u[i] <= thresh[(int(emb_bits[i]), int(x_bits[i]), int(ses_bits[i]))])
                        for i in range(n)], dtype=int)

        # 3) map emb_bits → actual embeddings
        emb_list = self.sampler(emb_bits.tolist())      # list of n tensors [D]
        embeddings = T.stack(emb_list, dim=0)           # → (n, D)

        # 4) remap binary values: 0 → -1
        emb_bits[emb_bits == 0] = -1
        ses_bits[ses_bits == 0] = -1
        x_bits[x_bits == 0]     = -1
        y_bits[y_bits == 0]     = -1

        # 5) package everything as tensors
        ses = T.from_numpy(ses_bits).float().unsqueeze(1).to('cpu')
        x   = T.from_numpy(x_bits).float().unsqueeze(1).to('cpu')
        y   = T.from_numpy(y_bits).float().unsqueeze(1).to('cpu')

        return {
            'EMB': embeddings,
            'SES': ses,
            'X':   x,
            'Y':   y,
        }

    def calculate_query(self, model, tau, m, evaluating):
        print("TODO: implement calculate_query")
        return T.tensor([0.0])


if __name__ == "__main__":


    # Path to your saved embeddings + labels file
    npz_path = "dat/HAM10000/embeddings_labels.npz"

    # Load the data
    data = np.load(npz_path)
    labels = data["labels"]

    # Compute and print unique labels (and counts)
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Available labels and their sample counts:")
    for lbl, cnt in zip(unique_labels, counts):
        print(f"  Label {int(lbl)}: {cnt} samples")




    mdg = HAM10000DataGenerator("sampling")

    n_samples = 4
    data = mdg.generate_samples(n_samples)
    
    print(data)
