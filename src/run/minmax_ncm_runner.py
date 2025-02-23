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
import src.metric.visualization as vis
from .base_runner import BaseRunner


class MinMaxNCMRunner(BaseRunner):
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
            logger=pl.loggers.TensorBoardLogger(f'{directory}/logs/{model_name}/'),
            log_every_n_steps=1,
            devices=devices,  
            accelerator=accelerator,
        ), checkpoint

    def run(self, exp_name, n, trial_index, hyperparams=None, gpu=None,
            lockinfo=os.environ.get('SLURM_JOB_ID', ''), verbose=False):
        """
        Runs the pipeline. Returns the resulting model.

        exp_name: Name of the experiment for labeling purposes.
        n: Number of data samples.
        trial_index: If running multiple trials, this indicates the trial number.
        hyperparams: Hyperparameters passed from main.
        gpu: Which GPU to use, if available.
        lockinfo: For parallelization.
        verbose: Prints more information if True.
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

                # Set random seed to a hash of the parameter settings for reproducibility
                seed = int(hashlib.sha512(key.encode()).hexdigest(), 16) & 0xffffffff
                T.manual_seed(seed)
                np.random.seed(seed)
                if verbose:
                    print('Key:', key)
                    print('Seed:', seed)

                if gpu is None:
                    gpu = int(T.cuda.is_available())

                for r in range(hyperparams.get("id-reruns", 1)):
                    # Create data-generating model and generate data
                    normalize = hyperparams["normalize"]
                    use_tau = hyperparams["use-tau"]

                    if verbose:
                        print('Generating data')
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

                    # Create NCM pipeline
                    m_max = self.pipeline(dat_set, cg, v_size, v_type, hyperparams=hyperparams,
                                          ncm_model=self.ncm_model, maximize=True)
                    m_min = self.pipeline(dat_set, cg, v_size, v_type, hyperparams=hyperparams,
                                          ncm_model=self.ncm_model, maximize=False)
                    if verbose:
                        for v in m_max.ncm.v_size:
                            print("FUNCTION {}".format(v))
                            print(m_max.ncm.f[v])
                        for v in m_min.ncm.v_size:
                            print("FUNCTION {}".format(v))
                            print(m_min.ncm.f[v])

                    # Train model
                    trainer_max, checkpoint_max = self.create_trainer("{}/{}".format(d, r), "ncm_max",
                                                              hyperparams['max-epochs'], hyperparams['patience'], gpu)
                    trainer_max.fit(m_max)
                    trainer_min, checkpoint_min = self.create_trainer("{}/{}".format(d, r), "ncm_min",
                                                              hyperparams['max-epochs'], hyperparams['patience'], gpu)
                    trainer_min.fit(m_min)
                    #ckpt = T.load(checkpoint.best_model_path)  # Find best model
                    #m.load_state_dict(ckpt['state_dict'])  # Save best model

                    # Calculate metrics
                    Q_real = dat_m.calculate_query(model=None, tau=hyperparams["use-tau"], m=100000,
                                                   evaluating=True).item()
                    Q_max = dat_m.calculate_query(model=m_max.ncm, tau=hyperparams["use-tau"], m=100000,
                                                   evaluating=True).item()
                    Q_min = dat_m.calculate_query(model=m_min.ncm, tau=hyperparams["use-tau"], m=100000,
                                                   evaluating=True).item()
                    results = {
                        "Q_real": Q_real,
                        "Q_max": Q_max,
                        "Q_min": Q_min,
                        "max_err": Q_max - Q_real,
                        "min_err": Q_min - Q_real,
                        "max_min_gap": Q_max - Q_min
                    }

                    # Save results
                    with open(f'{d}/{r}/results.json', 'w') as file:
                        json.dump(results, file)
                    T.save(m_max.state_dict(), f'{d}/{r}/best_max.th')
                    T.save(m_min.state_dict(), f'{d}/{r}/best_min.th')

                with open(f'{d}/hyperparams.json', 'w') as file:
                    new_hp = {k: str(v) for (k, v) in hyperparams.items()}
                    json.dump(new_hp, file)
                T.save(dict(), f'{d}/best.th')  # breadcrumb file
                return True

            except Exception:
                # Move out/*/* to err/*/*/#
                e = d.replace("out/", "err/").rsplit('-', 1)[0]
                e_index = len(glob.glob(e + '/*'))
                e += '/%s' % e_index
                os.makedirs(e.rsplit('/', 1)[0], exist_ok=True)
                shutil.move(d, e)
                print(f'moved {d} to {e}')
                raise
