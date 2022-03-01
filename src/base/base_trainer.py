
from tqdm import tqdm
import datetime
import os
from torch.utils.tensorboard  import SummaryWriter
from utils import utils


class base_trainer():
    def __init__(self,
                    epochs,
                    model,
                    val_epoch  = 1,
                    run_name   = 'default',
                    log_dir    = 'save',
                    checkpoint_dir = 'checkpoints'
                    ):

        self.run_name    = run_name
        self.log_root    = log_dir
        self.checkpoint_root = checkpoint_dir
        self.epochs      = epochs
        self.val_epoch   = val_epoch

        # CHECKPOINTS 
        date_time = datetime.datetime.now().strftime('%m-%d_%H-%M')
        self.checkpoint_dir = os.path.join(self.checkpoint_root, run_name)
        utils.dir_exists(self.checkpoint_dir)
        # TENSOBOARD
        self.writer_dir = os.path.join(self.log_root,run_name)
        utils.dir_exists(self.writer_dir)
        self.writer = SummaryWriter(self.writer_dir)

    def train(self):
        for epoch in range(self.epochs):
            results = self.train_epoch(epoch)

            if epoch == self.val_epoch:
                val_results = self.valid_epoch(epoch)


       




