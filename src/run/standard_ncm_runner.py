import os
import glob
import shutil
import hashlib
import json

import numpy as np
import torch as T
import pytorch_lightning as pl

from src.ds.causal_graph import CausalGraph
from src.datagen import SCMDataset, get_transform
from src.datagen import SCMDataTypes as sdt
from src.scm.scm import expand_do
from src.pipeline.repr_pipeline import RepresentationalPipeline
from src.scm.repr_nn.representation_nn import RepresentationalNN
import src.metric.visualization as vis
from .base_runner import BaseRunner


class NCMRunner(BaseRunner):
    """
    Runner for general purpose NCM-training. Performs the following tasks:
    1. Creates the data generating model and generates the appropriate amount of samples.
    2. Creates a pipeline for the given NCM model.
    3. Trains an NCM on the given model.
    4. Evaluates the NCM on its performance in fitting the data.
    """
    def __init__(self, pipeline, dat_model, ncm_model):
        super().__init__(pipeline, dat_model, ncm_model)
        self.rep_pipeline = RepresentationalPipeline

    def create_trainer(self, directory, model_name, max_epochs, patience, gpu=None):
        """
        Creates a PyTorch Lightning trainer.
        """
        checkpoint = pl.callbacks.ModelCheckpoint(dirpath=f'{directory}/checkpoints/{model_name}/',
                                                  monitor="train_loss",
                                                  save_on_train_epoch_end=True)

        accelerator = "cpu"
        devices = "auto"
        if gpu is not None and T.cuda.is_available():
            accelerator = "gpu"
            devices = gpu
        
        return pl.Trainer(
            callbacks=[
                checkpoint,
                pl.callbacks.EarlyStopping(monitor='train_loss',
                                           patience=patience,
                                           min_delta=self.pipeline.min_delta,
                                           check_on_train_epoch_end=True)
            ],
            max_epochs=max_epochs,
            accumulate_grad_batches=1,
            logger=pl.loggers.TensorBoardLogger(save_dir=f'{directory}/logs/{model_name}/'),
            log_every_n_steps=1,
            devices=devices,
            accelerator=accelerator
        ), checkpoint

    def run(self, exp_name, n, trial_index, hyperparams=None, gpu=None,
            lockinfo=os.environ.get('SLURM_JOB_ID', '')):
        """
        Runs the pipeline. Returns the resulting model.

        exp_name: Name of the experiment for labeling purposes.
        n: Number of data samples.
        trial_index: If running multiple trials, this indicates the trial number.
        hyperparams: Hyperparameters passed from main.
        gpu: Which GPU to use, if available.
        lockinfo: For parallelization.
        """
        key = self.get_key(n, trial_index)
        d = 'out/%s/%s' % (exp_name, key)  # name of the output directory

        if hyperparams is None:
            hyperparams = dict()

        with self.lock(f'{d}/lock', lockinfo) as acquired_lock:
            # Attempts to grab the lock for a particular trial. Only attempts the trial if the lock is obtained.
            if not acquired_lock:
                print('[locked]', d)
                return

            try:
                # Return if best.th is generated (i.e. training is already complete)
                if os.path.isfile(f'{d}/best.th'):
                    print('[done]', d)
                    return

                # Do not replace everything if representational model exists
                if not os.path.isfile(f'{d}/best_rep.th'):
                    # Since training is not complete, delete all directory files except for the lock
                    print('[running]', d)
                    for file in glob.glob(f'{d}/*'):
                        if os.path.basename(file) != 'lock':
                            if os.path.isdir(file):
                                shutil.rmtree(file)
                            else:
                                try:
                                    os.remove(file)
                                except FileNotFoundError:
                                    pass

                # Set random seed to a hash of the parameter settings for reproducibility
                seed = int(hashlib.sha512(key.encode()).hexdigest(), 16) & 0xffffffff
                T.manual_seed(seed)
                np.random.seed(seed)
                if hyperparams["verbose"]:
                    print('Key:', key)
                    print('Seed:', seed)

                # Create data-generating model and generate data
                normalize = hyperparams["normalize"]
                use_tau = hyperparams["use-tau"]
                img_query = hyperparams["img-query"]

                if hyperparams["verbose"]:
                    print('Generating data')

                if img_query:
                    dat_m = self.dat_model(hyperparams['img-size'], mode=hyperparams['mode'])  # Data generating model
                    cg = CausalGraph.read("dat/cg/{}.cg".format(dat_m.cg))  # Causal diagram object
                    v_size = dat_m.v_size
                    v_type = dat_m.v_type
                    dat_set = SCMDataset(  # Convert data to a Torch Dataset object
                        dat_m, n, augment_transform=get_transform(hyperparams["transform"], hyperparams["img-size"]))
                else:
                    dat_m = self.dat_model(normalize=normalize)  # Data generating model
                    if use_tau:
                        cg_name = dat_m.cg_high_level
                        v_size = dat_m.v_size_high_level
                        v_type = dat_m.v_type_high_level
                    else:
                        cg_name = dat_m.cg
                        v_size = dat_m.v_size
                        v_type = dat_m.v_type

                    cg = CausalGraph.read("dat/cg/{}.cg".format(cg_name))  # Causal diagram object
                    dat_set = SCMDataset(  # Convert data to a Torch Dataset object
                        dat_m, n, augment_transform=get_transform(hyperparams["transform"], hyperparams["img-size"]))

                    if use_tau:
                        dat_set.data = dat_m.tau(dat_set.data)
                        dat_set.v_type = dat_m.v_type_high_level

                # Create Representation pipeline
                rep_model = None
                rep_v_size = {k: v for (k, v) in v_size.items()}
                rep_v_type = {k: v for (k, v) in v_type.items()}
                if hyperparams['pipeline'] == "gan_joint":
                    rep_model = RepresentationalNN(cg, v_size, v_type, hyperparams=hyperparams)

                    # Change sizes and types to match representation
                    if hyperparams['rep-image-only']:
                        for v in v_type:
                            if v_type[v] == sdt.IMAGE:
                                rep_v_size[v] = hyperparams['rep-size']
                                rep_v_type[v] = hyperparams['rep-type']
                    else:
                        rep_v_size = {v: hyperparams['rep-size'] for v in v_type}
                        rep_v_type = {v: hyperparams['rep-type'] for v in v_type}

                elif hyperparams['repr'] != "none":
                    rep_m = self.rep_pipeline(dat_set, cg, v_size, v_type, hyperparams=hyperparams)

                    # Check if representation model already exists
                    if os.path.isfile(f'{d}/best_rep.th'):
                        if hyperparams["verbose"]:
                            print("Representation model already found...")
                        rep_m.load_state_dict(T.load(f'{d}/best_rep.th'))
                    else:
                        if hyperparams["verbose"]:
                            print("Training representation system...")

                        # Initial rep visualization
                        if img_query:
                            img_sample = dat_set.get_image_batch(64)
                            regen_sample = rep_m.model(img_sample)
                            for img_var in img_sample:
                                if hyperparams["verbose"]:
                                    vis.show_image_grid(img_sample[img_var])
                                    vis.show_image_grid(regen_sample[img_var])
                                else:
                                    vis.show_image_grid(img_sample[img_var],
                                                        dir=f'{d}/before_train_data_{img_var}.png')
                                    vis.show_image_grid(regen_sample[img_var],
                                                        dir=f'{d}/before_train_repr_{img_var}.png')

                        # Train representational model
                        rep_trainer, rep_checkpoint = self.create_trainer(d, "rep_model", hyperparams['max-epochs'],
                                                                          hyperparams['patience'], gpu)
                        rep_trainer.fit(rep_m)
                        ckpt = T.load(rep_checkpoint.best_model_path)  # Find best model
                        rep_m.load_state_dict(ckpt['state_dict'])
                        T.save(rep_m.state_dict(), f'{d}/best_rep.th')  # Save best model

                        # Final rep visualization
                        if img_query:
                            img_sample = dat_set.get_image_batch(64)
                            regen_sample = rep_m.model(img_sample)
                            for img_var in img_sample:
                                if hyperparams["verbose"]:
                                    vis.show_image_grid(img_sample[img_var])
                                    vis.show_image_grid(regen_sample[img_var])
                                else:
                                    vis.show_image_grid(img_sample[img_var],
                                                        dir=f'{d}/after_train_data_{img_var}.png')
                                    vis.show_image_grid(regen_sample[img_var],
                                                        dir=f'{d}/after_train_repr_{img_var}.png')

                    # Change sizes and types to match representation
                    if hyperparams['rep-image-only']:
                        for v in v_type:
                            if v_type[v] == sdt.IMAGE:
                                rep_v_size[v] = hyperparams['rep-size']
                                rep_v_type[v] = hyperparams['rep-type']
                    else:
                        rep_v_size = {v: hyperparams['rep-size'] for v in v_type}
                        rep_v_type = {v: hyperparams['rep-type'] for v in v_type}

                    rep_model = rep_m.model

                # Create NCM pipeline
                if hyperparams['pipeline'] == "gan_joint":
                    m = self.pipeline(dat_set, cg, v_size, v_type, rep_v_size, rep_v_type, repr_model=rep_model,
                                      hyperparams=hyperparams, ncm_model=self.ncm_model)
                else:
                    m = self.pipeline(dat_set, cg, rep_v_size, rep_v_type, repr_model=rep_model,
                                      hyperparams=hyperparams, ncm_model=self.ncm_model)

                # Initial visualization
                if img_query:
                    img_sample = dat_set.get_image_batch(64)
                    img_sample_fake = m(64)
                    for img_var in img_sample:
                        if hyperparams["verbose"]:
                            vis.show_image_grid(img_sample[img_var])
                            vis.show_image_grid(img_sample_fake[img_var])
                        else:
                            vis.show_image_grid(img_sample[img_var], dir=f'{d}/before_train_real_{img_var}.png')
                            vis.show_image_grid(img_sample_fake[img_var], dir=f'{d}/before_train_fake_{img_var}.png')
                else:
                    Q_real = dat_m.calculate_query(model=None, tau=use_tau, m=100000,
                                                   evaluating=True).item()
                    Q_estimate = dat_m.calculate_query(model=m.ncm, tau=use_tau, m=100000,
                                                       evaluating=True).item()
                    print("Q real: {}".format(Q_real))
                    print("Q estimate: {}".format(Q_estimate))
                    print("Q error: {}".format(Q_estimate - Q_real))

                # Train model
                trainer, checkpoint = self.create_trainer(d, "ncm", hyperparams['max-epochs'], hyperparams['patience'],
                                                          gpu)
                trainer.fit(m)  # Fit the pipeline on the data
                #ckpt = T.load(checkpoint.best_model_path)  # Find best model
                #m.load_state_dict(ckpt['state_dict'])  # Save best model

                # Save results
                with open(f'{d}/hyperparams.json', 'w') as file:
                    new_hp = {k: str(v) for (k, v) in hyperparams.items()}
                    json.dump(new_hp, file)
                T.save(m.state_dict(), f'{d}/best.th')

                if img_query:
                    # Final visualization
                    img_lists = m.img_lists
                    for v in img_lists:
                        if v_type[v] == sdt.IMAGE:
                            if hyperparams["verbose"]:
                                print("Image count: {}".format(len(img_lists[v])))
                                vis.show_image_timeline(img_lists[v])
                            else:
                                vis.show_image_timeline(img_lists[v], dir=f'{d}/training_fake_{img_var}.gif')

                    img_sample = dat_set.get_image_batch(64)
                    img_sample_fake = m(64)
                    for img_var in img_sample:
                        if hyperparams["verbose"]:
                            vis.show_image_grid(img_sample[img_var])
                            vis.show_image_grid(img_sample_fake[img_var])
                        else:
                            vis.show_image_grid(img_sample[img_var], dir=f'{d}/after_train_real_{img_var}.png')
                            vis.show_image_grid(img_sample_fake[img_var], dir=f'{d}/after_train_fake_{img_var}.png')
                else:
                    Q_real = dat_m.calculate_query(model=None, tau=use_tau, m=100000,
                                                   evaluating=True).item()
                    Q_estimate = dat_m.calculate_query(model=m.ncm, tau=use_tau, m=100000,
                                                       evaluating=True).item()
                    results = {
                        "Q_real": Q_real,
                        "Q_estimate": Q_estimate,
                        "Q_err": Q_estimate - Q_real
                    }
                    with open(f'{d}/results.json', 'w') as file:
                        json.dump(results, file)

                return m
            except Exception:
                # Move out/*/* to err/*/*/#
                e = d.replace("out/", "err/").rsplit('-', 1)[0]
                e_index = len(glob.glob(e + '/*'))
                e += '/%s' % e_index
                os.makedirs(e.rsplit('/', 1)[0], exist_ok=True)
                shutil.move(d, e)
                print(f'moved {d} to {e}')
                raise
