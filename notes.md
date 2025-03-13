# Notes
TODO:
what are the args in the generator class doing?
don't really understand how repr works in minmax_ncm_runner

# **Wednesday, 2/26 – Meeting Notes**

# Key Takeaways from SimCLR ([Chen et al., 2020](https://arxiv.org/abs/2002.05709))

## 1. Contrastive Loss
SimCLR uses a contrastive loss function to learn representations by maximizing agreement between differently augmented views of the same image. The loss function is based on the NT-Xent (Normalized Temperature-scaled Cross Entropy Loss) and is formulated as:

$$
\ell_i = - \log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k)/\tau)}
$$

where:
- $ z_i $ and $ z_j $ are representations of the same image under different augmentations,
- $ \text{sim}(z_i, z_j) $ is the cosine similarity between the two representations,
- $ \tau $ is the temperature scaling factor.

## 2. Data Augmentation
Data augmentation is crucial for learning strong representations. The most important augmentations in SimCLR are:
- **Random cropping** (followed by resizing)
- **Color distortions** (e.g., brightness, contrast, saturation, hue)

## 3. Projection Head
The projection head maps the representation to the contrastive loss space. The best-performing architecture follows:

$$
\text{ResNet embedding} \quad + \quad \text{Linear} \quad + \quad \text{ReLU} \quad + \quad \text{Linear}
$$

This is in contrast to using only a single linear layer, i.e.:

$$
\text{ResNet embedding} \quad + \quad \text{Linear}
$$

The extra non-linearity improves the quality of learned representations.

## **1. Miscellaneous**
- Fix typos in `R101`
- Add missing `requirements.txt` file to the repository
- Address small bug fixes related to PyTorch Lightning version

---

## **2. Interesting Next Tasks**
### **a) Exploring Smaller Embeddings**
- Train on **MNIST** and experiment with reducing embedding size while maintaining performance.

### **b) Extending to FashionMNIST**
- Interesting SCMs for modeling FashionMNIST data.

### **c) Scaling Up to More Complex Tasks**
- More challenging datasets (e.g., CIFAR, ImageNet subsets)

---

## **3. Understanding Representation Learning**
- How are representations being learned in the current setup?



## Setup
`requirements.txt` is missing:
It think it should be (does not include gpu support):
```
    numpy
    torch
    torchvision
    pandas
    matplotlib
    argparse
```
pytorch lightning
