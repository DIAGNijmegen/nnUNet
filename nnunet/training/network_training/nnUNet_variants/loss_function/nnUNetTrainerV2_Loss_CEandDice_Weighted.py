#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from nnunet.training.loss_functions.dice_loss import DCandCEWeightedLoss
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2


class nnUNetTrainer_V2_Loss_CEandDice_Weighted(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, **kwargs):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.loss = DCandCEWeightedLoss(
            class_weights=kwargs['class_weights'],
            weight_dc=kwargs.get('weight_dc', 1),
            weight_ce=kwargs.get('weight_ce', 1),
            soft_dice_kwargs=kwargs.get("soft_dice_kwargs", {'batch_dice': self.batch_dice, 'smooth': 1e-5,
                                                             'do_bg': False}),
            ce_kwargs=kwargs.get("ce_kwargs", {})
        )
