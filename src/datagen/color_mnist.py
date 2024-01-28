import numpy as np
import torch as T
from torchvision import transforms
from mnist import MNIST
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


class ColorMNISTDataGenerator(SCMDataGenerator):
    def __init__(self, image_size, mode, evaluating=False):
        super().__init__(mode)

        self.evaluating = evaluating
        self.raw_mnist_n = 0
        self.raw_mnist_images = None
        if not evaluating:
            mnist_data = MNIST('dat/mnist/')
            images, labels = mnist_data.load_training()
            self.raw_mnist_n = len(images)
            images = np.array(images).reshape((self.raw_mnist_n, 28, 28))
            labels = np.array(labels)

            self.raw_mnist_images = dict()
            for i in range(len(labels)):
                if labels[i] not in self.raw_mnist_images:
                    self.raw_mnist_images[labels[i]] = []
                self.raw_mnist_images[labels[i]].append(images[i])

        self.colors = {
            0: (1.0, 0.0, 0.0),
            1: (1.0, 0.6, 0.0),
            2: (0.8, 1.0, 0.0),
            3: (0.2, 1.0, 0.0),
            4: (0.0, 1.0, 0.4),
            5: (0.0, 1.0, 1.0),
            6: (0.0, 0.4, 1.0),
            7: (0.2, 0.0, 1.0),
            8: (0.8, 0.0, 1.0),
            9: (1.0, 0.0, 0.6)
        }

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size, antialias=True),
            transforms.ToTensor()
        ])

        self.mode = mode
        if mode == "sampling_noncausal":
            self.v_size = {
                'digit': 10,
                'image': 3
            }
            self.v_type = {
                'digit': sdt.ONE_HOT,
                'image': sdt.IMAGE
            }
            self.cg = "color_mnist_noncausal"
        else:
            self.v_size = {
                'color': 10,
                'digit': 10,
                'image': 3
            }
            self.v_type = {
                'color': sdt.ONE_HOT,
                'digit': sdt.ONE_HOT,
                'image': sdt.IMAGE
            }
            self.cg = "color_mnist"

    def colorize_image(self, image, color):
        color_value = self.colors[color]
        h, w = image.shape
        new_image = np.reshape(image, [h, w, 1])
        new_image = np.concatenate([new_image * color_value[0], new_image * color_value[1], new_image * color_value[2]],
                                   axis=2)
        return new_image

    def sample_digit(self, digit, color):
        total = len(self.raw_mnist_images[digit])
        ind = np.random.randint(0, total)
        img_choice = np.round(self.colorize_image(self.raw_mnist_images[digit][ind], color)).astype(np.uint8)

        return img_choice

    def generate_samples(self, n, U={}, do={}, p_align=0.85, return_U=False, normalize=True):
        if "u_conf" in U:
            u_conf = U["u_conf"]
        else:
            u_conf = np.random.randint(10, size=n)
        if "u_digit" in U:
            u_digit = U["u_digit"]
        else:
            u_digit = np.random.randint(10, size=n)
        if "u_color" in U:
            u_color = U["u_color"]
        else:
            u_color = np.random.randint(10, size=n)
        if "u_dig_align" in U:
            u_dig_align = U["u_dig_align"]
        else:
            u_dig_align = np.random.binomial(1, p_align, size=n)
        if "u_color_align" in U:
            u_color_align = U["u_color_align"]
        else:
            u_color_align = np.random.binomial(1, p_align, size=n)

        if "digit" in do:
            digit = do["digit"]
        else:
            digit = np.where(u_dig_align, u_conf, u_digit)
        if "color" in do:
            color = do["color"]
        else:
            color = np.where(u_color_align, u_conf, u_color)

        one_hot_digits = np.zeros((n, 10))
        one_hot_digits[np.arange(n), digit] = 1
        one_hot_colors = np.zeros((n, 10))
        one_hot_colors[np.arange(n), color] = 1

        imgs = []
        for i in range(n):
            img_sample = self.sample_digit(digit[i], color[i])
            img_sample = self.transform(img_sample).float()
            if normalize:
                img_sample = 2.0 * img_sample - 1.0
            else:
                img_sample = 255.0 * img_sample
            imgs.append(img_sample)

        if self.mode == "sampling_noncausal":
            data = {
                'digit': T.tensor(one_hot_digits).float(),
                'image': T.stack(imgs, dim=0)
            }
        else:
            data = {
                'color': T.tensor(one_hot_colors).float(),
                'digit': T.tensor(one_hot_digits).float(),
                'image': T.stack(imgs, dim=0)
            }

        if return_U:
            new_U = {
                "u_conf": u_conf,
                "u_digit": u_digit,
                "u_color": u_color,
                "u_dig_align": u_dig_align,
                "u_color_align": u_color_align
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

    def _sample_ctf(self, n, q, p_align=0.85, normalize=True):
        _, U = self.generate_samples(n, return_U=True, p_align=p_align, normalize=normalize)

        n_new = n
        for term in q.cond_term_set:
            samples = self.generate_samples(n=n_new, U=U, do={
                k: expand_do(v, n_new) for (k, v) in term.do_vals.items()
            }, return_U=False, p_align=p_align, normalize=normalize)

            cond_match = T.ones(n_new, dtype=T.bool)
            for (k, v) in term.var_vals.items():
                cond_match *= check_equal(samples[k], v)

            U = {k: v[cond_match] for (k, v) in U.items()}
            n_new = T.sum(cond_match.long()).item()

        if n_new <= 0:
                return float('nan')

        out_samples = dict()
        for term in q.term_set:
            expanded_do_terms = dict()
            for (k, v) in term.do_vals.items():
                    expanded_do_terms[k] = expand_do(v, n_new)
            q_samples = self.generate_samples(n=n_new, U=U, do=expanded_do_terms, return_U=False, p_align=p_align,
                                              normalize=normalize)
            out_samples.update(q_samples)

        return out_samples

    def show_image(self, image, label=None, dir=None):
        if label is not None:
            plt.title('Label is {label}'.format(label=label))
        image = T.movedim(image, 0, -1)
        image = (image + 1.0) / 2.0
        plt.imshow(image)

        if dir is not None:
            plt.savefig(dir)
        else:
            plt.show()
        plt.clf()

    def show_legend(self, dir=None, title="Legend"):
        photos = []
        for i in range(10):
            digit = self.sample_digit(i, i)
            digit = self.transform(digit).float()
            digit = 2.0 * digit - 1.0
            photos.append(digit)
        photos = T.stack(photos, dim=0)
        plt.figure(figsize=(10, 1))
        plt.axis("off")
        plt.title(title)
        grid = vutils.make_grid(photos, padding=2, normalize=True, nrow=10).cpu()
        plt.imshow(np.transpose(grid, (1, 2, 0)))

        if dir is not None:
            plt.savefig(dir)
        else:
            plt.show()
        plt.close()

    def show_gradient(self, dir=None):
        def color_mix(p, phase):
            if phase == 0:
                return (1.0, p, 0.0)
            elif phase == 1:
                return (1.0 - p, 1.0, 0.0)
            elif phase == 2:
                return (0.0, 1.0, p)
            elif phase == 3:
                return (0.0, 1.0 - p, 1.0)
            elif phase == 4:
                return (p, 0.0, 1.0)
            elif phase == 5:
                return (1.0, 0.0, 1.0 - p)
            else:
                return (0.0, 0.0, 0.0)

        n = 600
        fig, ax = plt.subplots(figsize=(8, 2))
        for x in range(n):
            phase = x // 100
            p = (x % 100) / 100.0
            color = color_mix(p, phase)
            ax.axvline(x, color=color, linewidth=4)
        if dir is not None:
            plt.savefig(dir)
        else:
            plt.show()
        plt.close()


if __name__ == "__main__":
    mdg = ColorMNISTDataGenerator(32, "sampling")

    # data = mdg.generate_samples(10)
    # print(data['color'])
    # print(data['digit'])
    # for i in range(len(data['image'])):
    #     mdg.show_image(data['image'][i])

    # mdg.show_legend("legend.png")
    # mdg.show_gradient("gradient.png")

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
