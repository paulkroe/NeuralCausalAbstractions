import os
import argparse
import wandb
import numpy as np
from dotenv import load_dotenv

from src.pipeline import GANPipeline, GANReprPipeline
from src.scm.ncm import GAN_NCM
from src.run import NCMRunner, MinMaxNCMRunner
from src.datagen import ColorMNISTDataGenerator, BMIDataGenerator, AgeCifarDataGenerator
from src.datagen.scm_datagen import SCMDataTypes as sdt

load_dotenv()
wandb_api_key = os.getenv("WANDB_API_KEY")

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

valid_pipelines = {
    "gan": GANPipeline,
    "gan_joint": GANReprPipeline
}
valid_generators = {
    "mnist": ColorMNISTDataGenerator,
    "bmi": BMIDataGenerator,
    "cifar": AgeCifarDataGenerator
}
architectures = {
    "gan": GAN_NCM,
    "gan_joint": GAN_NCM
}

mode_choices = {"sampling", "sampling_noncausal", "img_only", "identify"}
id_generators = {"bmi"}
id_pipelines = {"gan"}
gan_choices = {"vanilla", "bgan", "wgan", "wgangp"}
gan_arch_choices = {"dcgan", "biggan"}
gan_disc_choices = {"standard", "biggan"}
type_choices = {
    "real": sdt.REAL,
    "binary": sdt.REP_BINARY,
    "ones": sdt.REP_BINARY_ONES
}


# Basic setup settings
parser = argparse.ArgumentParser(description="Basic Runner")
parser.add_argument('name', help="name of the experiment")
parser.add_argument('mode', help="type of experiment")
parser.add_argument('gen', help="data generating model")
parser.add_argument('pipeline', help="pipeline to use")

parser.add_argument('--wandb', action="store_true", help="trace run with wandb")
parser.add_argument('--wandb-project-name', type=str, default=None, help="wandb project name")
parser.add_argument('--wandb-org-name', type=str, default=None, help="wandb organization name")

# Hyper-parameters for optimization
parser.add_argument('--lr', type=float, default=1e-4, help="generator optimizer learning rate (default: 1e-4)")
parser.add_argument('--alpha', type=float, default=0.99, help="optimizer alpha (default: 0.99)")
parser.add_argument('--data-bs', type=int, default=128, help="batch size of data (default: 128)")
parser.add_argument('--ncm-bs', type=int, default=128, help="batch size of NCM samples (default: 128)")
parser.add_argument('--grad-acc', type=int, default=1, help="number of accumulated batches per backprop (default: 1)")
parser.add_argument('--max-epochs', type=int, default=1000, help="maximum number of training epochs (default: 1000)")
parser.add_argument('--patience', type=int, default=-1, help="patience for early stopping (default: -1)")

# Hyper-parameters for function NNs
parser.add_argument('--h-layers', type=int, default=2, help="number of hidden layers (default: 2)")
parser.add_argument('--h-size', type=int, default=128, help="neural network hidden layer size (default: 128)")
parser.add_argument('--scale-h-size', action="store_true", help="multiplies h-size by input dimensionality")
parser.add_argument('--feature-maps', type=int, default=64, help="CNN feature maps (default: 64)")
parser.add_argument('--u-size', type=int, default=1, help="dimensionality of U variables (default: 1)")
parser.add_argument('--scale-u-size', action="store_true", help="multiplies u-size by v dimensionality")
parser.add_argument('--neural-pu', action="store_true", help="use neural parameters in U distributions")
parser.add_argument('--batch-norm', action="store_true", help="set flag to use batch norm")

# Hyper-parameters for representational NNs
parser.add_argument('--no-repr', action="store_true", help="disable representation learning")
parser.add_argument('--rep-size', type=int, default=8, help="Size of representational embedding (default: 8)")
parser.add_argument('--rep-type', default="real", help="Data type of representational embedding (default: real)")
parser.add_argument('--rep-image-only', action="store_true",
                    help="only use representation for images")
parser.add_argument('--rep-lr', type=float, default=-1.0, help="optimizer learning rate for representation")
parser.add_argument('--rep-bs', type=int, default=-1, help="batch size of data for representation")
parser.add_argument('--rep-grad-acc', type=int, default=-1,
                    help="number of accumulated batches per backprop for representation")
parser.add_argument('--rep-h-layers', type=int, default=-1, help="number of representation hidden layers")
parser.add_argument('--rep-h-size', type=int, default=-1, help="representation hidden layer size")
parser.add_argument('--rep-feature-maps', type=int, default=-1, help="representation CNN feature maps")
parser.add_argument('--rep-class-lambda', type=float, default=0.1,
                    help="weight for classification loss in conditional encoder")
parser.add_argument('--rep-temperature', type=float, default=1.0, help="temperature for contrastive loss")
parser.add_argument('--rep-contrast-lambda', type=float, default=0.1, help="weight for contrastive loss in encoder")
parser.add_argument('--rep-max-epochs', type=int, default=None, help="maximum number of training epochs for representation")
parser.add_argument('--rep-patience', type=int, default=None, help="patience for early stopping in representation")
parser.add_argument('--rep-reconstruct', action="store_true", help="use reconstruction loss when learning representations")
parser.add_argument('--rep-unsup-contrastive', action="store_true", help="use unsupervised contrastive loss when learning representations")
parser.add_argument('--rep-sup-contrastive', action="store_true", help="use supervised contrastive loss when learning representations")
parser.add_argument('--rep-pred-parents', action="store_true", help="use loss from predicting parents when training representations")


# Hyper-parameters for GAN-NCM
parser.add_argument('--gan-mode', default="vanilla", help="GAN loss function (default: vanilla)")
parser.add_argument('--gan-arch', default="dcgan", help="NN Architecture for GANs (default: dcgan)")
parser.add_argument('--disc-type', default="standard", help="discriminator type (default: standard)")
parser.add_argument('--disc-lr', type=float, default=2e-4, help="discriminator optimizer learning rate (default: 2e-4)")
parser.add_argument('--disc-h-layers', type=int, default=2,
                    help="number of hidden layers in discriminator (default: 2)")
parser.add_argument('--disc-h-size', type=int, default=-1,
                    help="width of hidden layers in discriminator (default: computed from size of inputs)")
parser.add_argument('--d-iters', type=int, default=1,
                    help="number of discriminator iterations per generator iteration (default: 1)")
parser.add_argument('--grad-clamp', type=float, default=0.01,
                    help="value for clamping gradients in WGAN (default: 0.01)")
parser.add_argument('--gp-weight', type=float, default=10.0,
                    help="regularization constant for gradient penalty in WGAN-GP (default: 10.0)")
parser.add_argument('--gp-one-side', action="store_true",
                    help="use one-sided version of gradient penalty in WGAN-GP")

# Image settings
parser.add_argument('--img-size', type=int, default=16, help="resize images to this size (use powers of 2)")
parser.add_argument('--transform', help="transformation applied to image variables")

# ID settings
parser.add_argument('--n-reruns', '-r', type=int, default=1, help="number of reruns in id experiments")
parser.add_argument('--max-lambda', type=float, default=1e-2, help="maximum lambda value for ID loss")
parser.add_argument('--min-lambda', type=float, default=1e-4, help="minimum lambda value for ID loss")
parser.add_argument('--custom-query', action="store_true", help="use a custom query aside from image")
parser.add_argument('--use-tau', action="store_true", help="use existing tau")

# Experiment parameters
parser.add_argument('--no-normalize', action="store_true", help="turn off dataset normalizing")
parser.add_argument('--n-trials', '-t', type=int, default=1, help="number of trials")
parser.add_argument('--n-samples', '-n', type=int, default=10000, help="number of samples (default: 10000)")
parser.add_argument('--gpu', help="GPU to use")

# Developer settings
parser.add_argument('--verbose', action="store_true", help="print more information")

args = parser.parse_args()

# Basic setup
mode_choice = args.mode.lower()
pipeline_choice = args.pipeline.lower()
gen_choice = args.gen.lower()
gan_choice = args.gan_mode.lower()
gan_arch_choice = args.gan_arch.lower()
repr_type_choice = args.rep_type.lower()

if args.wandb:
    assert args.wandb_project_name is not None
    assert args.wandb_org_name is not None
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    else:
        raise ValueError("WANDB_API_KEY not found. Please check your .env file.")

assert mode_choice in mode_choices
assert pipeline_choice in valid_pipelines
assert gen_choice in valid_generators
assert gan_choice in gan_choices
assert gan_arch_choice in gan_arch_choices
assert repr_type_choice in type_choices

if args.mode == "identify":
    assert gen_choice in id_generators
    assert pipeline_choice in id_pipelines

pipeline = valid_pipelines[pipeline_choice]
dat_model = valid_generators[gen_choice]
ncm_model = architectures[pipeline_choice]

transform_name = gen_choice if args.transform is None else args.transform

gpu_used = 0 if args.gpu is None else [int(args.gpu)]

# Hyperparams to be passed to all downstream objects
hyperparams = {
    'pipeline': pipeline_choice,
    'mode': mode_choice,
    'transform': transform_name,
    'lr': args.lr,
    'alpha': args.alpha,
    'data-bs': args.data_bs,
    'ncm-bs': args.ncm_bs,
    'grad-acc': args.grad_acc,
    'max-epochs': args.max_epochs,
    'patience': args.patience if args.patience > 0 else args.max_epochs,
    'h-layers': args.h_layers,
    'h-size': args.h_size,
    'scale-h-size': args.scale_h_size,
    'feature-maps': args.feature_maps,
    'u-size': args.u_size,
    'scale-u-size': args.scale_u_size,
    'neural-pu': args.neural_pu,
    'batch-norm': args.batch_norm,
    'gan-mode': gan_choice,
    'gan-arch': gan_arch_choice,
    'disc-type': args.disc_type,
    'disc-lr': args.disc_lr,
    'disc-h-layers': args.disc_h_layers,
    'disc-h-size': args.disc_h_size,
    'd-iters': args.d_iters,
    'grad-clamp': args.grad_clamp,
    'gp-weight': args.gp_weight,
    'gp-one-side': args.gp_one_side,
    'img-size': args.img_size,
    'no-repr': args.no_repr,
    'rep-size': args.rep_size,
    'rep-type': type_choices[repr_type_choice],
    'rep-image-only': args.rep_image_only,
    'rep-lr': args.rep_lr if args.rep_lr > 0 else args.lr,
    'rep-bs': args.rep_bs if args.rep_bs > 0 else args.data_bs,
    'rep-grad-acc': args.rep_grad_acc if args.rep_grad_acc > 0 else args.grad_acc,
    'rep-h-layers': args.rep_h_layers if args.rep_h_layers > 0 else args.h_layers,
    'rep-h-size': args.rep_h_size if args.rep_h_size > 0 else args.h_size,
    'rep-feature-maps': args.rep_feature_maps if args.rep_feature_maps > 0 else args.feature_maps,
    'rep-class-lambda': args.rep_class_lambda,
    'rep-temperature': args.rep_temperature,
    'rep-contrast-lambda': args.rep_contrast_lambda,
    'rep-max-epochs': args.rep_max_epochs if args.rep_max_epochs is not None else args.max_epochs,
    'rep-patience': args.rep_patience if args.rep_patience is not None and args.rep_patience > 0 else args.rep_max_epochs,
    'rep-reconstruct': args.rep_reconstruct,
    'rep-unsup-contrastive': args.rep_unsup_contrastive,
    'rep-sup-contrastive': args.rep_sup_contrastive,
    'rep-pred-parents': args.rep_pred_parents,
    'rep-train': args.rep_reconstruct or args.rep_unsup_contrastive or args.rep_sup_contrastive or args.rep_pred_parents,
    'identify': args.mode == "identify",
    'normalize': not args.no_normalize,
    'id-reruns': args.n_reruns,
    'max-lambda': args.max_lambda,
    'min-lambda': args.min_lambda,
    'use-tau': args.use_tau,
    'img-query': not args.custom_query,
    'verbose': args.verbose,
    'wandb': args.wandb,
    'wandb-project-name': args.wandb_project_name,
    'wandb-org-name': args.wandb_org_name
}
print(hyperparams)

if pipeline_choice == "gan":
    # Adjust data batch size accordingly when training more discriminator iterations
    hyperparams['data-bs'] = hyperparams['data-bs'] * hyperparams['d-iters']

if args.n_samples == -1:
    # Run experiment on several sample sizes if not specified
    n_list = 10.0 ** np.linspace(3, 5, 5)
else:
    n_list = [args.n_samples]

# Run for each sample size in n_list
for n in n_list:
    # Avoid using more data than available if possible
    n = int(n)
    hyperparams["data-bs"] = min(args.data_bs, n)
    hyperparams["ncm-bs"] = min(args.ncm_bs, n)
    if args.rep_bs < 0:
        hyperparams["rep-bs"] = min(args.ncm_bs, n)
    else:
        hyperparams["rep-bs"] = min(args.rep_bs, n)

    # Run for n_trials amount of trials
    for i in range(args.n_trials):
        while True:
            try:
                # Create a runner for the NCM and pass all hyperparams
                if args.mode == "identify":
                    runner = MinMaxNCMRunner(pipeline, dat_model, ncm_model)
                else:
                    runner = NCMRunner(pipeline, dat_model, ncm_model)
                if not runner.run(args.name, n, i,
                                  hyperparams=hyperparams, gpu=gpu_used):
                    break
            except Exception as e:
                # Raise any errors
                print(e)
                print('[failed]', i, args.name)
                raise
