import os
from tempfile import NamedTemporaryFile
from contextlib import contextmanager


class BaseRunner:
    """
    Basic runner class for running PyTorch pipelines. Any runners should extend this class.
    """
    def __init__(self, pipeline, dat_model, ncm_model):
        """
        pipeline: Pipeline object
        dat_model: Data generating object
        ncm_model: NCM object
        """
        self.pipeline = pipeline
        self.pipeline_name = pipeline.__name__
        self.dat_model = dat_model
        self.dat_model_name = dat_model.__name__
        self.ncm_model = ncm_model
        self.ncm_model_name = ncm_model.__name__

    @contextmanager
    def lock(self, file, lockinfo):
        """
        Locking mechanism for the purposes of running parallel experiments.
        Attempts to acquire a file lock; yield whether or not lock was acquired.
        """
        os.makedirs(os.path.dirname(file), exist_ok=True)
        os.makedirs('tmp/', exist_ok=True)
        with NamedTemporaryFile(dir='tmp/') as tmp:
            try:
                os.link(tmp.name, file)
                acquired_lock = True
            except FileExistsError:
                acquired_lock = os.stat(tmp.name).st_nlink == 2
        if acquired_lock:
            with open(file, 'w') as fp:
                fp.write(lockinfo)
            try:
                yield True
            finally:
                try:
                    os.remove(file)
                except FileNotFoundError:
                    pass
        else:
            yield False

    def get_key(self, n, trial_index):
        """
        Creates an identifier for the specific trial.
        """
        return ('gen=%s-pipeline=%s-model=%s-n_samples=%s-trial_index=%s'
                % (self.dat_model_name, self.pipeline_name, self.ncm_model_name, n, trial_index))

    def run(self, exp_name, n, trial_index, hyperparams=None, gpu=None,
            lockinfo=os.environ.get('SLURM_JOB_ID', '')):
        """
        Runs the implemented pipeline.
        """
        raise NotImplementedError()
