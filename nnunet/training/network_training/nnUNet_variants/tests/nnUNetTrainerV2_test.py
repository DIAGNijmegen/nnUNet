from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2


class nnUNetTrainerV2_test(nnUNetTrainerV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_num_epochs = 2
        self.num_batches_per_epoch = 2
        self.num_val_batches_per_epoch = 2
