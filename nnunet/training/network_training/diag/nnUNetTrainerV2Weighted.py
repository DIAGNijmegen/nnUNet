import os
import warnings

from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

from nnunet.network_architecture.neural_network import SegmentationNetwork
from typing import Optional, List

import torch
from torch.cuda.amp import autocast

from nnunet.training.dataloading.dataset_loading import unpack_dataset

from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss
from nnunet.training.loss_functions.dice_loss import SoftDiceLoss, SoftDiceLossSquared

from nnunet.training.dataloading.diag.dataset_loading_weightmaps import DataLoader3DWeighted
from nnunet.training.data_augmentation.diag.data_augmentation_moreDA_weightmaps import get_moreDA_augmentation_weightmaps
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda

import numpy as np


class DC_and_CE_loss_weighted(torch.nn.Module):
    def __init__(
        self,
        soft_dice_kwargs,
        ce_kwargs,
        aggregate="sum",
        square_dice=False,
        weight_ce=1.0,
        weight_dice=0.1,
        log_dice=False,
        ignore_label=None,
        eps: float = 1e-8,
    ):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss_weighted, self).__init__()
        ce_kwargs["reduction"] = "none"
        if ignore_label is not None:
            assert not square_dice, "not implemented"
            ce_kwargs["reduction"] = "none"
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.eps = eps

        self.ignore_label = ignore_label

        if not square_dice:
            self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared(
                apply_nonlin=softmax_helper, **soft_dice_kwargs
            )

    def forward(self, net_output, target, weights):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, "not implemented for one hot encoding"
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        if self.weight_dice != 0:
            warnings.warn(
                f"Applying dice with factor {self.weight_dice}, but it doesn't use the weightmap"
            )
            # TODO make dice loss weighted..., does this make sense???
            dc_loss = self.dc(net_output, target, loss_mask=mask)
        else:
            dc_loss = 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        if self.weight_ce == 0:
            ce_loss = 0
        else:
            ce_loss = self.ce(net_output, target[:, 0].long())
            if self.ignore_label is not None:
                ce_loss *= mask[:, 0] * weights[:, 0]
                ce_loss = ce_loss.sum() / (
                    (mask[:, 0] * weights[:, 0]).sum() + self.eps
                )
            else:
                ce_loss = (ce_loss * weights[:, 0]).sum() / (
                    weights[:, 0].sum() + self.eps
                )

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son")  # reserved for other stuff (later)
        return result


class MultipleOutputLoss2WithWeightMaps(torch.nn.Module):
    def __init__(
        self, loss: torch.nn.Module, weight_factors: Optional[List[float]] = None
    ):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2WithWeightMaps, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(
        self, x: List[torch.Tensor], y: List[torch.Tensor], w: List[torch.Tensor]
    ) -> torch.Tensor:
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        assert isinstance(w, (tuple, list)), "w must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(x[0], y[0], w[0])
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[i], w[i])
        return l


class nnUNetTrainerV2Weighted(nnUNetTrainerV2):
    def run_iteration(
        self, data_generator, do_backprop=True, run_online_evaluation=False
    ):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict["data"]
        target = data_dict["target"]
        weightmaps = data_dict["weightmap"]

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)
        weightmaps = maybe_to_torch(weightmaps)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
            weightmaps = to_cuda(weightmaps)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.loss(output, target, weightmaps)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target, weightmaps)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            print("Online evaluation doesn't take weightmaps into account...")
            self.run_online_evaluation(
                output, target
            )  # TODO online eval with weight maps???

        del target

        return l.detach().cpu().numpy()

    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        if self.threeD:
            dl_tr = DataLoader3DWeighted(
                self.dataset_tr,
                self.basic_generator_patch_size,
                self.patch_size,
                self.batch_size,
                False,
                oversample_foreground_percent=self.oversample_foreground_percent,
                pad_mode="constant",
                pad_sides=self.pad_all_sides,
                memmap_mode="r",
            )
            dl_val = DataLoader3DWeighted(
                self.dataset_val,
                self.patch_size,
                self.patch_size,
                self.batch_size,
                False,
                oversample_foreground_percent=self.oversample_foreground_percent,
                pad_mode="constant",
                pad_sides=self.pad_all_sides,
                memmap_mode="r",
            )
        else:
            raise NotImplementedError(
                "DataLoader2DWeighted hasn't been implemented yet..."
            )
        return dl_tr, dl_val

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation (with support for weightmaps)
        - enforce to only run this code once
        - loss function wrapper for deep supervision (with support for weightmaps)

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array(
                [True]
                + [
                    True if i < net_numpool - 1 else False
                    for i in range(1, net_numpool)
                ]
            )
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            print(
                "Changed loss to MultipleOutputLoss2WithWeightMaps (DC_and_CE_loss_weighted) "
                "dice weight is set to 0 (not weighted with the weightmap)"
            )
            dc_ce_loss = DC_and_CE_loss_weighted(
                {"batch_dice": self.batch_dice, "smooth": 1e-5, "do_bg": False},
                {},
                weight_dice=0,
            )
            self.loss = MultipleOutputLoss2WithWeightMaps(
                loss=dc_ce_loss, weight_factors=self.ds_loss_weights
            )
            ################# END ###################

            self.folder_with_preprocessed_data = os.path.join(
                self.dataset_directory,
                self.plans["data_identifier"] + "_stage%d" % self.stage,
            )
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!"
                    )
                print("Add data augmentation with weightmap support...")
                (
                    self.tr_gen,
                    self.val_gen,
                ) = get_moreDA_augmentation_weightmaps(
                    self.dl_tr,
                    self.dl_val,
                    self.data_aug_params["patch_size_for_spatialtransform"],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False,
                )
                self.print_to_log_file(
                    "TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                    also_print_to_console=False,
                )
                self.print_to_log_file(
                    "VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                    also_print_to_console=False,
                )
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(
                self.network, (SegmentationNetwork, torch.nn.DataParallel)
            )
        else:
            self.print_to_log_file(
                "self.was_initialized is True, not running self.initialize again"
            )
        self.was_initialized = True
