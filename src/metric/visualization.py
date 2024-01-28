import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def show_image(image, title=None, dir=None):
    if title is not None:
        plt.title(title)
    plt.imshow(image, cmap='gray')

    if dir is not None:
        plt.savefig(dir)
    else:
        plt.show()
    plt.close()


def show_image_grid(batch, dir=None, title="Images"):
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(title)
    grid = vutils.make_grid(batch[: 64], padding=2, normalize=True).cpu()
    # grid = vutils.make_grid(batch.to(device)[: 64], padding=2).cpu()
    plt.imshow(np.transpose(grid, (1, 2, 0)))

    if dir is not None:
        plt.savefig(dir)
    else:
        plt.show()
    plt.close()


def show_image_timeline(img_list, dir=None):
    """
    Shows animation of images during training.

    :param img_list: list of images of animation
    :param dir: directory and name of file, in gif format
    """
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    grid_img_list = [vutils.make_grid(x[:64], padding=2, normalize=True).cpu() for x in img_list]
    ims = [[plt.imshow(np.transpose(x, (1, 2, 0)), animated=True)] for x in grid_img_list]
    anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    if dir is not None:
        anim.save(dir, dpi=80, writer='imagemagick')
    else:
        plt.show()
    plt.close()
