import numpy as np
import torch as T
import wandb
from src.datagen.scm_datagen import SCMDataGenerator
from src.datagen.scm_datagen import SCMDataTypes as sdt
import itertools

def sample_binary_2d(probs: list[float], num_samples: int, random_state: int = None) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    if probs.shape != (4,) or not np.isclose(probs.sum(), 1.0):
        raise ValueError("`probs` must be length-4 and sum to 1.")
    if random_state is not None:
        np.random.seed(random_state)
    cats  = np.arange(4)
    draws = np.random.choice(cats, size=num_samples, p=probs)
    return np.stack([draws // 2, draws % 2], axis=1)

class HAM10000DataGenerator(SCMDataGenerator):
    def __init__(self, mode=None, normalize=False, evaluating=False, device=None):
        super().__init__(mode)
        self.device = device or T.device('cpu')
        self.v_size = {'X':1, 'EMB':1, 'SES':1, 'Y':1}
        self.v_type = {k: sdt.BINARY_ONES for k in self.v_size}
        self.cg = "ham10000"
        self.evaluating = evaluating

    def _sample_exogenous(self, n: int):
        probs0 = [0.45, 0.05, 0.45, 0.05]
        probs1 = [0.45, 0.45, 0.05, 0.05]
        return {
            'emb_ses': sample_binary_2d(probs0, n),
            'ses_x':   sample_binary_2d(probs1, n),
            'u':       np.random.rand(n),
        }

    def _compute_from_exogenous(self, exog: dict, do: dict = None):
        do = do or {}
        emb_ses = exog['emb_ses']; ses_x = exog['ses_x']; u = exog['u']
        n = len(u)

        # 1) structural equations (internal bits 0/1)
        emb = emb_ses[:,0].astype(int)
        ses = np.logical_or(emb_ses[:,1], ses_x[:,0]).astype(int)
        x   = ses_x[:,1].astype(int)

        # 2) do-interventions: override and map back to 0/1
        if 'EMB' in do:
            ext = do['EMB'].squeeze().cpu().numpy().astype(int)    # now -1 or +1
            emb = (ext == +1).astype(int)                         # back to 0/1
        if 'SES' in do:
            ext = do['SES'].squeeze().cpu().numpy().astype(int)
            ses = (ext == +1).astype(int)
        if 'X' in do:
            ext = do['X'].squeeze().cpu().numpy().astype(int)
            x   = (ext == +1).astype(int)

        # 3) threshold for Y (still using 0/1 keys)
        thresh = {
            # (0,0,0): 0.80, (0,0,1): 0.75,
            # (0,1,0): 0.75, (0,1,1): 0.80,
            # (1,0,0): 0.10, (1,0,1): 0.60,
            # (1,1,0): 0.20, (1,1,1): 0.65,

            (0,0,0): 1, (0,0,1): 1,
            (0,1,0): 1, (0,1,1): 0,
            (1,0,0): 1, (1,0,1): 0,
            (1,1,0): 0, (1,1,1): 0,

        }
        y = np.fromiter(
            (int(u[i] <= thresh[(emb[i], x[i], ses[i])]) for i in range(n)),
            dtype=int, count=n
        )

        # 4) remap everything {0,1}→{−1,+1} and to tensors
        def signed(arr): return np.where(arr==0, -1, +1)
        to_t = lambda arr: T.from_numpy(signed(arr)).float().view(n,1).to(self.device)

        return {
            'EMB': to_t(emb),
            'SES': to_t(ses),
            'X':   to_t(x),
            'Y':   to_t(y),
        }


    def generate_samples(self, n: int, do: dict = None):
        """
        Draw n i.i.d. samples under intervention `do`.  No obs-filtering.
        """
        exog = self._sample_exogenous(n)
        data = self._compute_from_exogenous(exog, do=do)

        EMB = data['EMB']
        N = EMB.shape[0]

        # 1) build a boolean mask of length N
        flip_mask = T.rand(N, device=EMB.device) < 0.15

        # 3) flip the sign on those samples
        EMB[flip_mask] = -EMB[flip_mask]

        # put it back
        data['EMB'] = EMB

        return {k: v.cpu() for k,v in data.items()}

    def calculate_query(self,
                        model: T.nn.Module = None,
                        tau: float = None,
                        m: int = 5000,
                        evaluating: bool = False,
                        log: bool = False) -> dict[str, float]:
        """
        Estimate P(Y=1 | do(...)) for a fixed set of interventions:
          • do(X=0), do(X=1), do(SES=0), do(SES=1)
        If `model` is provided, uses `model.forward(n, do, evaluating)`.
        Otherwise falls back to sampling from the SCM.
        
        Returns:
            A dict mapping the query string to its estimated probability.
        """
        m = 5000
        # 1) build constant do-tensors in {-1,+1}
        emb0  = T.full((m,1), -1, dtype=T.float32, device=self.device)
        emb1  = T.full((m,1), +1, dtype=T.float32, device=self.device)
        ses0  = T.full((m,1), -1, dtype=T.float32, device=self.device)
        ses1  = T.full((m,1), +1, dtype=T.float32, device=self.device)
        x0    = T.full((m,1), -1, dtype=T.float32, device=self.device)
        x1    = T.full((m,1), +1, dtype=T.float32, device=self.device)

        # 2) list of (name, do_dict)
        queries = [
            ("P(Y=1 | do(EMB=0, SES=0, X=0))", {"X":   x0, "EMB": emb0, "SES": ses0}),
            ("P(Y=1 | do(EMB=0, SES=0, X=1))", {"X":   x1, "EMB": emb0, "SES": ses0}),
            ("P(Y=1 | do(EMB=0, SES=1, X=0))", {"X":   x0, "EMB": emb0, "SES": ses1}),
            ("P(Y=1 | do(EMB=0, SES=1, X=1))", {"X":   x1, "EMB": emb0, "SES": ses1}),

            ("P(Y=1 | do(EMB=1, SES=0, X=0))", {"X":   x0, "EMB": emb1, "SES": ses0}),
            ("P(Y=1 | do(EMB=1, SES=0, X=1))", {"X":   x1, "EMB": emb1, "SES": ses0}),
            ("P(Y=1 | do(EMB=1, SES=1, X=0))", {"X":   x0, "EMB": emb1, "SES": ses1}),
            ("P(Y=1 | do(EMB=1, SES=1, X=1))", {"X":   x1, "EMB": emb1, "SES": ses1}),
            
            ("P(Y=1 | do(SES=0))", {"SES": ses0}),
            ("P(Y=1 | do(SES=1))", {"SES": ses1}),
        ]

        results: dict[str, float] = {}
        for name, do_dict in queries:
            # 3) get Y from model or SCM
            if model is not None:

                batch = model.forward(n=m, evaluating=False)
                x     = batch["X"].to(self.device)
                y     = batch["Y"].to(self.device)
                emb   = batch["EMB"].to(self.device)
                ses   = batch["SES"].to(self.device)

                # print(f"X: {x.mean()}") 
                # print(f"Y: {y.mean()}")
                # print(f"EMB: {emb.mean()}")
                # print(f"SES: {ses.mean()}")
                # print("=======================")   
                # assert 0

                for key in do_dict:
                    do_dict[key] = do_dict[key].to(self.device)
                batch = model.forward(n=m, do=do_dict, evaluating=evaluating)

                y     = batch["Y"].to(self.device)



            else:
                batch = self.generate_samples(m, do=do_dict)

                # # assume batch is a dict of 1-D torch tensors
                # Y   = batch['Y']   # shape (m,)
                # EMB = batch['EMB']
                # X   = batch['X']
                # SES = batch['SES']

                # # mask for EMB=1 & X=1 & SES=1
                # mask = (EMB == 1) & (X == -1) & (SES == 1)

                # # numerator: those also with Y=1
                # num = ((Y == 1) & mask).sum().float()

                # # denominator: all with EMB=0, X=0, SES=0
                # den = mask.sum().float()

                # # estimate P(Y=1 | EMB=0, X=0, SES=0)
                # estimate = num / den if den > 0 else T.tensor(float('nan'))

                # print(f"Estimated P(Y=1 | EMB=0, X=0, SES=0) = {estimate.item():.3f}")

                # assert 0

                y     = batch["Y"].to(self.device)

            # 4) convert {-1,+1}→{0,1} and average
            p = float(((y >= 0).float().mean().item()))

            # 5) optional logging
            if log:
                wandb.log({name: p})

            results[name] = p
        return [p for p in results.values()]

def estimate_joint_emb_x_ses(mdg, n_samples: int = 10000) -> dict[tuple[int,int,int], float]:
    """
    Monte–Carlo estimate of joint P(EMB, X, SES) under the observational SCM.

    Args:
        mdg:          an instance of HAM10000DataGenerator
        n_samples:    how many i.i.d. points to draw

    Returns:
        A dict mapping (emb, x, ses) each in {-1,+1} to its estimated probability.
    """
    # 1) draw data
    data = mdg.generate_samples(n_samples)
    emb = data['EMB'].view(-1).numpy()
    x   = data['X'].view(-1).numpy()
    ses = data['SES'].view(-1).numpy()

    # 2) stack into (n_samples,3) array
    arr = np.stack([emb, x, ses], axis=1)

    # 3) find unique combos and their counts
    combos, counts = np.unique(arr, axis=0, return_counts=True)

    # 4) initialize all 8 combos to zero
    joint = {combo: 0.0 for combo in itertools.product([-1, +1], repeat=3)}

    # 5) fill in empirical frequencies
    for combo, cnt in zip(combos, counts):
        joint[tuple(combo)] = cnt / n_samples

    return joint

# --- example usage ---
if __name__ == "__main__":
    from src.datagen.ham10000 import HAM10000DataGenerator

    mdg = HAM10000DataGenerator("sampling")
    joint_probs = estimate_joint_emb_x_ses(mdg, n_samples=10000)

    for (emb, x, ses), p in sorted(joint_probs.items()):
        print(f"EMB={emb}, X={x}, SES={ses}: {p:.4f}")
