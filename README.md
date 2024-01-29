# Neural Causal Abstractions 
This repository contains the code for the paper ["Neural Causal Abstractions"](https://arxiv.org/abs/2401.02602) by Kevin Xia and Elias Bareinboim.

Please cite our work if you found this code useful.

## Setup
Run the following code to install python requirements.
```
python -m pip install -r requirements.txt
```
To run the ColoredMNIST experiments, place the MNIST data files in `dat/mnist`.

## Running the code

All experiment procedures can be run using the `main.py` file with the desired arguments entered. For any of the commands below, feel free to modify the hyperparameters from `main.py`.

### BMI Experiment

To run the identification part of the BMI experiment, navigate to the base directory of the repository and run
```
python -m src.main <NAME> identify bmi gan --u-size 16 --batch-norm --gan-mode wgan --custom-query -r 2 --gpu 0
```
where `<NAME>` is replaced with the name of the folder in which the results will be saved.

For estimation, run
```
python -m src.main <NAME> sampling bmi gan --u-size 16 --batch-norm --gan-mode wgan --custom-query --gpu 0
```

To turn off normalization of the data, use the `--no-normalize` tag, and to apply abstractions, use the `--use-tau` tag.

To compile the visualization of the ID results, name the folders of the results `naive`, `naive_nonormal` and `tau` for each setting respectively, and place all three folders into a single folder.
Then run the command
```
python -m src.experiment.experiment1_id_results <DIR>
```
where `<DIR>` refers to the outer folder with all three results. Similarly, for estimation, name the three folders identically and place them in a different directory.
Then run
```
python -m src.experiment.experiment1_est_results <DIR>
```

### ColoredMNIST Experiment

To run the GAN-NCM without representation learning, run the following command
```
python -m src.main <NAME> sampling mnist gan --h-layers 3 --h-size 2 --scale-h-size --scale-u-size --batch-norm --gan-mode wgan --gan-arch biggan --disc-type biggan --img-size 32 --gpu 0
```
The command can be run with `sampling_noncausal` in place of `sampling` to run a standard conditional causal-agnostic GAN.

To run the GAN-RNCM, run the following command
```
python -m src.main <NAME> sampling mnist gan --h-layers 3 --h-size 2 --scale-h-size --scale-u-size --batch-norm --gan-mode wgan --gan-arch biggan --repr auto_enc_conditional --rep-size 64 --rep-image-only --rep-h-layers 3 --rep-h-size 128 --img-size 32 --gpu 0
```

To compile the visualization of the results, name the folders of the results `naive`, `noncausal`, and `representational` for each setting respectively, and place all three folders into a single folder.
Then run the command
```
python -m src.experiment.experiment2_results <DIR>
```
where `<DIR>` refers to the outer folder with all three results.
