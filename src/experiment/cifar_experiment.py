# GROUND TRUTH
# Compute P(old = 1 | do(one_hot_animal = 'horse'))
# P(old = 1 | do(one_hot_animal = 'horse')) 
# = 1/6 * (7.5/20 + 2.5/15 + 12.5/25) = 0.1736

import os
import argparse
import torch as T

from src.datagen import AgeCifarDataGenerator
from src.metric.model_utility import load_model

def ground_truth_old_do_horse(n_samples = 100):
    datagen = AgeCifarDataGenerator(32, "sampling")
    samples = datagen.generate_samples(n_samples, do={"animal": [7 for _ in range(n_samples)]})
    return samples["old"].mean(dim=0).item()

# python -m src.main CIFAR sampling cifar gan --h-layers 3 --h-size 2 --scale-h-size --scale-u-size --batch-norm --gan-mode wgan --gan-arch biggan --repr auto_enc --rep-size 64 --rep-image-only --rep-h-layers 3 --rep-h-size 128 --img-size 32 --rep-no-decoder --rep-max-epochs=1 --rep-patience=1 --max-epochs=1 --patience=1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Age CIFAR Experiment Results Parser")

    parser.add_argument('--dir', help="directory of the experiment")
    parser.add_argument('--num-samples', type=int, default=10, help="number of samples to use for Monte Carlo Estimation")
    
    args = parser.parse_args()

    print(f"groud truth for P(old = 1 | do(one_hot_animal = 'horse')) = {0.1736}")
    print(f"Monte Carlo Estimation for P(old = 1 | do(one_hot_animal = 'horse')) with {args.num_samples} samples {ground_truth_old_do_horse(args.num_samples):.4f}")
    
    # load model
    exp_name = os.listdir(args.dir)[0]
    d_model = "{}/{}".format(args.dir, exp_name)
    m, _, _, _, _ = load_model(d_model, verbose=False)

    # perform forward pass
    labels = T.full((args.num_samples,), 7 - 2, dtype=T.long)
    one_hot_lables = T.nn.functional.one_hot(labels, num_classes=6)
    data = m.forward(n=args.num_samples, do={"one_hot_animal": one_hot_lables}, evaluating=True)
    estimate = data["old"].mean(dim=0).item()
    
    print(f"Estimation for P(old = 1 | do(one_hot_animal = 'horse')) with {args.num_samples} samples {estimate:.4f}")