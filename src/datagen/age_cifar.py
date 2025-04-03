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

def expand_do(val, n):
    return np.ones(n, dtype=int) * val

class RandomRotateCropResize:
    def __init__(self, crop_size, output_size):
        self.crop_size = crop_size
        self.output_size = output_size

    def __call__(self, x):
        # Random rotation: choose from 0°, 90°, 180°, 270°
        angle = random.choice([0, 90, 180, 270])
        x = TF.rotate(x, angle)

        # Random crop (center crop if image is too small)
        _, h, w = x.shape
        top = random.randint(0, max(0, h - self.crop_size))
        left = random.randint(0, max(0, w - self.crop_size))
        x = TF.crop(x, top, left, self.crop_size, self.crop_size)

        # Resize back to original size
        x = TF.resize(x, (h, w))
        return x

class AgeCifarDataGenerator(SCMDataGenerator):
    def __init__(self, image_size, mode, evaluating=False):
        super().__init__(mode)
        self.evaluating = evaluating # not sure what this does
        # U_{Conf} samples an individual in the population.
        # that is: it samples an animal and then an age for that animal
        # we assume age is distributed uniformly
        self.n_classes = 6
        self.transform = RandomRotateCropResize(crop_size=128, output_size=256)  # example sizes
        # life expectancy of each animal normalized to 1
        self.life_expectancy = {
            "bird": 3,
            "cat": 9,
            "deer": 20,
            "dog": 15,
            "frog": 1,
            "horse": 25
        }      

        # Define transformations (convert to tensor and normalize)
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts to torch.Tensor
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # CIFAR-10 mean/std
        ])

        # Load CIFAR-10 dataset
        train_dataset = torchvision.datasets.CIFAR10(root="./dat/cifar", train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root="./dat/cifar", train=False, download=True, transform=transform)

        cifar10_classes = train_dataset.classes

        # Define animal classes and corresponding indices
        self.animal_classes = ["bird", "cat", "deer", "dog", "frog", "horse"]
        self.animal_indices = [cifar10_classes.index(cls) for cls in self.animal_classes]
        self.animal_class_indices = {cls: cifar10_classes.index(cls) for cls in self.animal_classes}
        self.indices_animal_classes = {v: k for k, v in self.animal_class_indices.items()}

        # Function to filter images into a dictionary where keys are animals and values are tensors of images
        def create_animal_dict(dataset):
            animal_dict = {animal_idx: [] for animal_idx in self.animal_class_indices.values()}  # Initialize empty lists for each animal
            for img, label in dataset:
                for idx in self.animal_class_indices.values():
                    if label == idx:
                        animal_dict[idx].append(img)

            # Convert lists to tensors
            for idx in self.animal_class_indices.values():
                animal_dict[idx] = T.stack(animal_dict[idx])  # Shape: [num_images, C, H, W]

            return animal_dict

        # Create dictionaries for training and test sets
        self.train_animal_dict = create_animal_dict(train_dataset)
        self.test_animal_dict = create_animal_dict(test_dataset)

        self.v_size = {
            'age': 1,
            'one_hot_animal': self.n_classes,
            'animal': 3,
            'old': 1
        }
        self.v_type = {
            'age': sdt.REAL,
            'one_hot_animal': sdt.ONE_HOT,
            'animal': sdt.IMAGE,
            'old': sdt.BINARY_ONES
        }
        self.cg = "age_cifar"

        self.group_A = [self.animal_class_indices[animal] for animal in ["cat", "dog", "horse"]]
        self.group_B = [self.animal_class_indices[animal] for animal in ["bird", "deer", "frog"]]

    def generate_samples(self, n, U={}, do={}, p_align=0.85, return_U=False):
        import numpy as np
        # sample U if not provided
        if "u_conf" in U:
            u_conf = U["u_conf"]
        else:
            animal_indices = np.random.choice(self.animal_indices, n)
            age = []
            for idx in animal_indices:
                age.append(np.random.uniform(0, self.life_expectancy[self.indices_animal_classes[idx]]))

            u_conf = [(animal_indices[i], age[i]) for i in range(n)]


        if "age" in do:
            age = do["age"]

        if "animal" in do:
            animal = do["animal"]
        else:
            animal = [u_conf[i][0] for i in range(n)]
        
        animal_idx = animal[:]
        for i in range(n):
            sample_idx = np.random.randint(0, len(self.train_animal_dict[animal_idx[i]]))
            animal[i] = self.transform(self.train_animal_dict[animal_idx[i]][sample_idx, :])

        one_hot_animal = T.zeros((n, self.n_classes))
        one_hot_animal[T.arange(n), [i - 2 for i in animal_idx]] = 1 # subtract 2 since original indices are 2-7

        old = []
        for i in range(n):
            old.append(1 if age[i] > self.life_expectancy[self.indices_animal_classes[animal_idx[i]]] / 2 else -1)       
 
        data = {
            'one_hot_animal': one_hot_animal.float(),
            'animal': T.stack(animal, dim=0),
            'age': T.Tensor(age).float().unsqueeze(1),
            'old': T.Tensor(old).float().unsqueeze(1)
        }

        if return_U:
            new_U = {
                "u_conf": u_conf,
            }
            return data, new_U
        return data

    def sample_ctf(self, q, n=64, batch=None, max_iters=1000, p_align=0.85, normalize=True):
        if batch is None:
            batch = n

        iters = 0
        n_samps = 0
        samples = dict()

        while n_samps < n:
            if iters >= max_iters:
                return float('nan')

            new_samples = self._sample_ctf(batch, q, p_align=p_align, normalize=normalize)
            if isinstance(new_samples, dict):
                if len(samples) == 0:
                    samples = new_samples
                else:
                    for var in new_samples:
                        samples[var] = T.concat((samples[var], new_samples[var]), dim=0)
                        n_samps = len(samples[var])

            iters += 1

        return {var: samples[var][:n] for var in samples}

    def show_image(self, img, label=None):
        plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
        if label is not None:
            plt.title(label)
        plt.show()

    def _sample_ctf(self, n, q, p_align=0.85, normalize=True):
        # i think this might be hard to do for continuous data
        # might want to discretize the scm
        raise NotImplementedError("Not implemented yet")

if __name__ == "__main__":
    mdg = AgeCifarDataGenerator(None, "sampling")

    n_samples = 4
    # data = mdg.generate_samples(n_samples)
    
    # for i in range(n_samples):
    #    label = f"{mdg.indices_animal_classes[(int)(data['animal_idx'][i].item())]} {data['age'][i]} {data['old'][i]}"
    #    mdg.show_image(data['animal'][i], label)
    

    data, sampled_u = mdg.generate_samples(n_samples, do={"animal": [mdg.animal_class_indices["horse"] for _ in range(n_samples)]}, return_U=True)
    animal_idx = [sampled_u["u_conf"][i][0] for i in range(n_samples)]
    
    # expect many young horses
    for i in range(n_samples):
        label = f"intervened animal {mdg.indices_animal_classes[animal_idx[i]]}, old: {data['age'][i]} {data['old'][i]}, true animal: {mdg.indices_animal_classes[animal_idx[i]]}"
        mdg.show_image(data['animal'][i], label)

    data, sampled_u = mdg.generate_samples(n_samples, do={"animal": [mdg.animal_class_indices["frog"] for _ in range(n_samples)]}, return_U=True)
    animal_idx = [sampled_u["u_conf"][i][0] for i in range(n_samples)]
    
    # expect many old frogs
    for i in range(n_samples):
        label = f"intervened animal {mdg.indices_animal_classes[animal_idx[i]]}, old: {data['age'][i]} {data['old'][i]}, true animal: {mdg.indices_animal_classes[animal_idx[i]]}"
        mdg.show_image(data['animal'][i], label)

    # test_var = "digit"
    # test_val_1_raw = 0
    # test_val_2_raw = 5
    #
    # test_val_1 = np.zeros((1, 10))
    # test_val_1[0, test_val_1_raw] = 1
    # test_val_2 = np.zeros((1, 10))
    # test_val_2[0, test_val_2_raw] = 1
    #
    # test_val_1 = T.from_numpy(test_val_1).float()
    # test_val_2 = T.from_numpy(test_val_2)
    #
    # y1 = CTFTerm({'image'}, {}, {'image': 1})
    # x1 = CTFTerm({test_var}, {}, {test_var: test_val_1})
    # x0 = CTFTerm({test_var}, {}, {test_var: test_val_2})
    # y1dox1 = CTFTerm({'image'}, {test_var: test_val_1_raw}, {'image': 1})
    #
    # py1givenx1 = CTF({y1}, {x1})
    # py1dox1 = CTF({y1dox1}, set())
    # py1dox1givenx0 = CTF({y1dox1}, {x0})
    #
    # batch1 = mdg.sample_ctf(py1givenx1, 64)
    # show_image_grid(batch1["image"])
    # batch2 = mdg.sample_ctf(py1dox1, 64)
    # show_image_grid(batch2["image"])
    # batch3 = mdg.sample_ctf(py1dox1givenx0, 64)
    # show_image_grid(batch3["image"])
