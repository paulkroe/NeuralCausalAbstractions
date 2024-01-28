import numpy as np
import torch as T
from src.datagen.scm_datagen import SCMDataGenerator
from src.datagen.scm_datagen import SCMDataTypes as sdt

from src.metric.evaluation import probability_table


class BMIDataGenerator(SCMDataGenerator):
    def __init__(self, normalize=True, evaluating=False):
        super().__init__()

        self.evaluating = evaluating
        self.v_size = {
            'R': 32,
            'F': 32,
            'N1': 1,
            'N2': 1,
            'N3': 1,
            'B': 1
        }
        if normalize:
            self.v_type = {
                'R': sdt.ONE_HOT,
                'F': sdt.ONE_HOT,
                'N1': sdt.REP_BINARY_ONES,
                'N2': sdt.REP_BINARY_ONES,
                'N3': sdt.REP_BINARY_ONES,
                'B': sdt.REP_BINARY_ONES
            }
        else:
            self.v_type = {
                'R': sdt.ONE_HOT,
                'F': sdt.ONE_HOT,
                'N1': sdt.REAL,
                'N2': sdt.REAL,
                'N3': sdt.REAL,
                'B': sdt.REAL
            }

        self.v_size_high_level = {
            'F': 1,
            'N': 1,
            'B': 1
        }
        self.v_type_high_level = {
            'F': sdt.BINARY,
            'N': sdt.BINARY,
            'B': sdt.BINARY
        }

        self.normalize = normalize
        self.cg = "bmi_lowlevel"
        self.cg_high_level = "bmi_highlevel"

    def generate_samples(self, n, do_F=None):
        u_rb = np.random.binomial(1, 0.25, size=n)
        r = np.mod(np.random.randint(0, 16, size=n) + 16 * np.random.binomial(1, 0.25, size=n) + 16 * u_rb, 32)
        if do_F is None:
            f = np.mod(r + np.random.randint(-3, 4, size=n), 32)
        else:
            f = do_F

        h = np.bitwise_xor(np.floor_divide(f, 16), np.random.binomial(1, 0.1, size=n))
        f_type = np.mod(f, 3)
        u_na = np.random.dirichlet((4, 1, 1), size=n)
        u_nb = np.random.uniform(0.0, 1.0, size=n)
        n1 = u_na[np.arange(n), f_type] * 216 * (0.25 * h + 1) + 9 * u_nb
        n2 = u_na[np.arange(n), np.mod(f_type + 1, 3)] * 216 * (0.25 * h + 1) + 9 * u_nb
        n3 = u_na[np.arange(n), np.mod(f_type + 2, 3)] * 96 * (0.25 * h + 1) + 4 * u_nb

        u_b = np.random.binomial(1, 0.1, size=n)
        b = ((n1 / 9) + (n2 / 9) + (n3 / 4) + 3 * u_rb) - 5
        b = ((b - 25) * np.power(-1, u_b)) + 25

        r_one_hot = np.zeros((n, 32))
        r_one_hot[np.arange(n), r] = 1
        f_one_hot = np.zeros((n, 32))
        f_one_hot[np.arange(n), f] = 1

        if self.normalize:
            n1 = 2 * (n1 / 279.0) - 1
            n2 = 2 * (n2 / 279.0) - 1
            n3 = 2 * (n3 / 124.0) - 1
            b = (b - 25) / 6.0

        n1 = np.expand_dims(n1, axis=1)
        n2 = np.expand_dims(n2, axis=1)
        n3 = np.expand_dims(n3, axis=1)
        b = np.expand_dims(b, axis=1)

        data = {
            'R': T.tensor(r_one_hot).float(),
            'F': T.tensor(f_one_hot).float(),
            'N1': T.tensor(n1).float(),
            'N2': T.tensor(n2).float(),
            'N3': T.tensor(n3).float(),
            'B': T.tensor(b).float()
        }

        return data

    def tau(self, low_level_samples):
        high_level_samples = dict()

        if 'F' in low_level_samples:
            new_f = T.argmax(low_level_samples['F'], dim=1)
            new_f = (new_f > 15).float().detach()
            new_f = T.unsqueeze(new_f, dim=1)
            high_level_samples['F'] = new_f

        if 'N1' in low_level_samples and 'N2' in low_level_samples and 'N3' in low_level_samples:
            carbs = low_level_samples['N1']
            protein = low_level_samples['N2']
            fat = low_level_samples['N3']
            if self.normalize:
                carbs = ((carbs + 1) / 2.0) * 279
                protein = ((protein + 1) / 2.0) * 279
                fat = ((fat + 1) / 2.0) * 124
            calories = 4 * carbs + 4 * protein + 9 * fat
            new_n = (calories >= 1080).float().detach()
            high_level_samples['N'] = new_n

        if 'B' in low_level_samples:
            if self.normalize:
                new_b = (low_level_samples['B'] >= 0).float().detach()
            else:
                new_b = (low_level_samples['B'] >= 25).float().detach()
            high_level_samples['B'] = new_b

        return high_level_samples

    def calculate_query(self, model=None, tau=False, m=100000, evaluating=False, maximize=False):
        """
        Calculates the query P(B | do(F)).
        :param model: Model to calculate with query. If None, calculates ground truth from data generating model.
        :param tau: If True, calculate in high level space.
        :param evaluating: If True, calculate probability. Otherwise, calculate distance with gradients.
        :param m: Number of Monte Carlo samples.
        :param maximize: If True, return loss for maximizing, otherwise return loss for minimizing.
        """
        if model is None:
            do_F = np.ones(m, dtype=int) * 20
            samples = self.generate_samples(m, do_F)
            if tau:
                samples = self.tau(samples)
        else:
            if tau:
                do_F = T.ones(m, 1).float()
            else:
                do_F = T.zeros(1, 32).float()
                do_F[0, 20] = 1
                do_F = T.tile(do_F, (m, 1))
            samples = model(n=m, do={'F': do_F}, evaluating=evaluating)

        if evaluating:
            if tau:
                return T.sum(samples['B']) / m
            else:
                if self.normalize:
                    return T.sum((samples['B'] >= 0).float()) / m
                else:
                    return T.sum((samples['B'] >= 25).float()) / m
        else:
            if tau:
                if maximize:
                    loss = T.mean(-T.log(samples['B'] + 1e-8))
                else:
                    loss = T.mean(-T.log((1 - samples['B']) + 1e-8))
            else:
                if self.normalize:
                    loss = T.mean(samples['B'])
                else:
                    loss = T.mean(samples['B'] - 25)

                if maximize:
                    loss = -loss

            return loss


if __name__ == "__main__":
    datagen = BMIDataGenerator(normalize=True)
    data = datagen.generate_samples(5)
    print(data)
    print(datagen.tau(data))

    print(datagen.calculate_query(tau=False, evaluating=True))
    print(datagen.calculate_query(tau=True, evaluating=True))

    data = datagen.tau(datagen.generate_samples(10000))
    print(probability_table(dat=data))
