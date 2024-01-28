import numpy as np
import torch as T
from mnist import MNIST
import matplotlib.pyplot as plt
from .scm_datagen import SCMDataGenerator
from .scm_datagen import SCMDataTypes as sdt


class MNISTDataGenerator(SCMDataGenerator):
    def __init__(self, image_size, normalize=True):
        super().__init__(normalize=normalize)

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

        self.average_pix = np.array([np.mean(self.raw_mnist_images[dig]) / 255 for dig in range(10)])

        self.v_size = {
            'X': 10,
            'Z': 1,
            'Y': 1
        }
        self.v_type = {
            'X': sdt.ONE_HOT,
            'Z': sdt.BINARY,
            'Y': sdt.IMAGE
        }
        self.cg = "backdoor"

    def sample_digit(self, dig, thick=None):
        done = False
        img_choice = None
        while not done:
            total = len(self.raw_mnist_images[dig])
            ind = np.random.randint(0, total)
            img_choice = self.raw_mnist_images[dig][ind]

            if thick is None:
                done = True
            else:
                thickness = np.mean(img_choice) / 255
                if thick == 0:
                    thickness = 1 - thickness
                thickness = thickness ** 2
                random_num = np.random.random()
                if random_num < thickness:
                    done = True

        return img_choice

    def generate_samples(self, n):
        thickness = np.random.binomial(1, 0.5, size=n)
        p_thick = np.power(self.average_pix, 2) / np.sum(np.power(self.average_pix, 2))
        p_thin = np.power(1 - self.average_pix, 2) / np.sum(np.power(1 - self.average_pix, 2))
        digits_thick = np.random.choice(np.arange(10), n, p=p_thick)
        digits_thin = np.random.choice(np.arange(10), n, p=p_thin)
        digits = np.where(thickness, digits_thick, digits_thin)
        one_hot_digits = np.zeros((n, 10))
        one_hot_digits[np.arange(n), digits] = 1

        imgs = []
        for i in range(n):
            img_sample = self.sample_digit(digits[i], thick=thickness[i]).astype(np.float32)
            if self.normalize:
                img_sample = 2.0 * (img_sample / 255.0) - 1.0
            imgs.append(img_sample)
        imgs = np.array(imgs)
        imgs = np.expand_dims(imgs, axis=1)

        data = {
            'Z': T.tensor(np.expand_dims(thickness, axis=-1)).float(),
            'X': T.tensor(one_hot_digits).float(),
            'Y': T.tensor(imgs).float()
        }

        return data

    def show_image(self, image, label=None, dir=None):
        if label is not None:
            plt.title('Label is {label}'.format(label=label))
        plt.imshow(image, cmap='gray')

        if dir is not None:
            plt.savefig(dir)
        else:
            plt.show()
        plt.clf()


if __name__ == "__main__":
    mdg = MNISTDataGenerator()
    print(mdg.average_pix)
    data = mdg.generate_samples(10)
    print(data['Z'])
    print(data['X'])
    for i in range(len(data['Y'])):
        mdg.show_image(data['Y'][i])
