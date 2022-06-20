# DIAG customizations

We use a customized fork of the original nnU-net code which can be found at [DIAGNijmegen/nnUNet](https://github.com/DIAGNijmegen/nnUNet)

* Replaces `shutil.copy` and `shutil.copytree` with custom implementations that work on Chansey and Blissey.
* Adds optional features that can be turned on by setting environment variables (see [switches.py](https://github.com/DIAGNijmegen/nnUNet/blob/master/nnunet/utilities/switches.py)):
  * An alternative resampling strategy that is more memory efficient (`DIAG_NNUNET_ALT_RESAMPLING`)
* Adds custom trainers:
  * nnUNetTrainerV2Sparse
  * nnUNetTrainerV2SparseNormalSampling
  * nnUNetTrainer_V2_Loss_CEandDice_Weighted


## nnUNetTrainerV2Sparse and nnUNetTrainerV2SparseNormalSampling

Both classes implement a similar simple training scheme that works with partially annotated segmentation data. 
Unlabeled segmentation data should be marked with a value of -1, inside the labels/ ground truth segmentation maps. These are just the default ground truth nnUNet data format files.

The difference between the classes is that the nnUNetTrainerV2Sparse trainer only will try to sample around annotated data, by randomly selecting an annotated voxel that would result in a valid patch (handles distance to border, minimal patch size, etc...), and that the nnUNetTrainerV2SparseNormalSampling trainer will use the default sampling without regard for where the annotated voxels are situated.

Some caveats with the sampling around annotated data with the current implementation, which are good to know before usage:

* Label sampling may result in label patches without the actual annotated labels due to augmentations (elastic deform, ) issues. This mostly occurs when only having a few non-clustered annotated voxels. So annotating whole slices or larger connected portions of the data is recommended.
* Each label segmentation map should always have at least some annotations, otherwise, the training process will crash with a RuntimeError.
* Currently, label sampling will always center on a randomly selected annotated voxel/pixel, this will limit augmentations to within the annotated label mask only (i.e. there is no random offset atm).
* The random_crop argument doesn't do anything for the sampling, since it is assumed to be random within the annotated data anyway.
