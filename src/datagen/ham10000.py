import numpy as np
import torch as T
import wandb
from src.datagen.scm_datagen import SCMDataGenerator
from src.datagen.scm_datagen import SCMDataTypes as sdt

def sample_binary_2d(probs: list[float], num_samples: int, random_state: int = None) -> np.ndarray:
    """
    Draw samples of two bits B0,B1 ∈ {0,1} with joint probs.
    Returns array shape (num_samples,2) of 0/1.
    """
    probs = np.asarray(probs, float)
    if probs.shape != (4,) or not np.isclose(probs.sum(), 1.0):
        raise ValueError("`probs` must be length-4 and sum to 1.")
    if random_state is not None:
        np.random.seed(random_state)
    cats  = np.arange(4)
    draws = np.random.choice(cats, size=num_samples, p=probs)
    return np.stack([draws // 2, draws % 2], axis=1)

class EmbeddingSampler:
    """
    Samples 512-dim embedding vectors by class (0 or 1) from a .npz.
    """
    def __init__(self, npz_path: str, device: T.device = None):
        data = np.load(npz_path)
        self.embeddings      = data['embeddings']  # [N, D]
        self.labels          = data['labels']      # [N]
        self.device          = device or T.device('cpu')
        self.indices_by_label = {
            int(lbl): np.where(self.labels == lbl)[0]
            for lbl in np.unique(self.labels)
        }

    def __call__(self, class_bits: T.Tensor) -> T.Tensor:
        """
        class_bits: Tensor of shape (n,1) with values 0 or 1 (dtype long or int)
        returns: Tensor of shape (n, D)
        """
        idxs = []
        for lbl in class_bits.view(-1).cpu().numpy().astype(int):
            choices = self.indices_by_label[lbl]
            idxs.append(np.random.choice(choices))
        emb_np = self.embeddings[idxs]     # (n, D)
        return T.from_numpy(emb_np).to(self.device)

class HAM10000DataGenerator(SCMDataGenerator):
    def __init__(self, mode=None, normalize=False, evaluating=False, device=None):
        super().__init__(mode)
        self.device  = device or T.device('cpu')
        self.v_size  = {'EMB':4, 'SES':1, 'X':1, 'Y':1}
        self.v_type  = {'EMB':sdt.REAL,
                        'SES':sdt.BINARY_ONES,
                        'X':  sdt.BINARY_ONES,
                        'Y':  sdt.BINARY_ONES}
        self.cg      = "ham10000"
        self.sampler = EmbeddingSampler("dat/HAM10000/ham10000_embeddings.npz", self.device)
        self.evaluating = evaluating

    def _sample_exogenous(self, n: int):
        emb_ses = sample_binary_2d([0.45,0.05,0.45,0.05], n)  # col0→EMB bit, col1→SES noise
        ses_x   = sample_binary_2d([0.45,0.45,0.05,0.05], n)  # col0→SES noise, col1→X bit
        u       = np.random.rand(n)                         # for Y
        # pre-sample indices for EMB embedding
        idxs    = [np.random.choice(self.sampler.indices_by_label[int(b)])
                   for b in emb_ses[:,0]]
        return {'emb_ses': emb_ses, 'ses_x': ses_x, 'u': u, 'idx': np.array(idxs)}

    def _compute_from_exogenous(self, exog: dict, do: dict = None):
        do      = do or {}
        emb_ses = exog['emb_ses']; ses_x = exog['ses_x']; u = exog['u']; idx = exog['idx']
        n       = len(u)

        # 1) raw bits in {0,1}
        emb_bits = emb_ses[:,0].astype(int)
        ses_bits = np.logical_or(emb_ses[:,1], ses_x[:,0]).astype(int)
        x_bits   = ses_x[:,1].astype(int)

        # 2) do-interventions override bits
        if 'EMB' in do:
            emb_bits = do['EMB'].squeeze().cpu().numpy().astype(int)
        if 'SES' in do:
            ses_bits = do['SES'].squeeze().cpu().numpy().astype(int)
        if 'X' in do:
            x_bits   = do['X'].squeeze().cpu().numpy().astype(int)

        # 3) get EMB tensor
        if 'EMB' in do:
            emb_tensor = self.sampler(do['EMB'])
        else:
            emb_tensor = self.sampler(T.from_numpy(emb_bits).view(n,1)).to(self.device)

        # 4) compute Y in {0,1}
        thresh = {
            (0,0,0):0,(0,0,1):0,
            (0,1,0):0,(0,1,1):1,
            (1,0,0):0,(1,0,1):1,
            (1,1,0):1,(1,1,1):1,
        }
        y_bits = np.fromiter(
            (int(u[i] <= thresh[(emb_bits[i], ses_bits[i], x_bits[i])]) for i in range(n)),
            dtype=int, count=n
        )

        # 5) remap {0,1}→{-1,+1}
        def signed(a): return np.where(a==0, -1, +1)
        ses_t = T.from_numpy(signed(ses_bits)).float().view(n,1).to(self.device)
        x_t   = T.from_numpy(signed(x_bits)) .float().view(n,1).to(self.device)
        y_t   = T.from_numpy(signed(y_bits)) .float().view(n,1).to(self.device)
        return {'EMB': emb_tensor, 'SES': ses_t, 'X': x_t, 'Y': y_t}

    def generate_samples(self, n: int, do: dict = None):
        exog = self._sample_exogenous(n)
        batch = self._compute_from_exogenous(exog, do=do)
        return {k:v.cpu() for k,v in batch.items()}

    def calculate_query(self,
                        model=None,
                        tau: float = None,
                        m: int = 5000,
                        evaluating: bool = False,
                        log: bool = False) -> dict[str,float]:
        m=2500
        """
        Returns Monte Carlo estimates for:
          • Observational P(Y=1)
          • P(Y=1|do(EMB=0)), P(Y=1|do(EMB=1))
          • P(Y=1|do(X=0)),   P(Y=1|do(X=1))
          • P(Y=1|do(SES=0)), P(Y=1|do(SES=1))
        """
        # build intervention dicts in {0,1} labels
        to_bit = lambda val: T.full((m,1), val, dtype=T.int64, device=self.device)
        emb0 = to_bit(0); emb1 = to_bit(1)
        x0   = to_bit(0); x1   = to_bit(1)
        s0   = to_bit(0); s1   = to_bit(1)

        queries = [
            # ("P(Y=1)",           None       ),
            # ("P(Y=1|do(EMB=0))", {"EMB": emb0}),
            # ("P(Y=1|do(EMB=1))", {"EMB": emb1}),

            ("P(Y=1|do(EMB=0, SES=0, X=0))",   {"X":   x0, "SES": s0, "EMB": emb0}),
            ("P(Y=1|do(EMB=0, SES=0, X=1))",   {"X":   x1, "SES": s0, "EMB": emb0}),
            ("P(Y=1|do(EMB=0, SES=1, X=0))",   {"X":   x0, "SES": s1, "EMB": emb0}),

            ("P(Y=1|do(EMB=1, SES=0, X=0))",   {"X":   x0, "SES": s0, "EMB": emb1}),

            ("P(Y=1|do(EMB=0, SES=1, X=1))",   {"X":   x1, "SES": s1, "EMB": emb0}),

            ("P(Y=1|do(EMB=1, SES=0, X=1))",   {"X":   x1, "SES": s0, "EMB": emb1}),
            ("P(Y=1|do(EMB=1, SES=1, X=0))",   {"X":   x0, "SES": s1, "EMB": emb1}),
            ("P(Y=1|do(EMB=1, SES=1, X=1))",   {"X":   x1, "SES": s1, "EMB": emb1}),

            # ("P(Y=1|do(X=1))",   {"X":   x1 }),
            # ("P(Y=1|do(SES=0))", {"SES": s0 }),
            # ("P(Y=1|do(SES=1))", {"SES": s1 }),
        ]

        results: dict[str,float] = {}
        for name, do_dict in queries:
            if model is not None:
                # Prepare a fresh do‐dict for the model
                model_do = {}
                if do_dict:
                    for var, tensor in do_dict.items():
                        if var == 'EMB':
                            # Convert class‐bits to actual embeddings
                            # sampler expects a (m,1) int Tensor in {0,1}
                            model_do['EMB'] = self.sampler(tensor)
                        else:
                            # pass through other interventions
                            model_do[var] = tensor.to(self.device)

                # batch = model.forward(n=20, evaluating=True)
                # x     = batch['X'].to(self.device)
                # y     = batch['Y'].to(self.device)
                # emb   = batch['EMB'].to(self.device)
                # print(emb)
                # print(f"X: {x.mean().item():.3f}")
                # print(f"Y: {y.mean().item():.3f}")
                # assert 0
                
                
                out = model.forward(n=m, do=model_do, evaluating=evaluating)
                y   = out['Y'].to(self.device)
            else:                
                batch = self.generate_samples(m, do=do_dict)
                y     = batch['Y'].to(self.device)

            # map {-1,+1}→{0,1} then compute mean
            p = float((y.ge(0).float().mean().item()))
            if log:
                wandb.log({name: p})
            results[name] = p

        return [v for v in results.values()]

# Example usage:
# mdg = HAM10000DataGenerator("sampling")
# joint = mdg.calculate_query(model=None, m=10000)
# print(joint)