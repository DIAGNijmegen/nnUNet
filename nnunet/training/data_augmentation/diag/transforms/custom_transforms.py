import numpy as np

from batchgenerators.transforms.abstract_transforms import AbstractTransform


def convert_3d_to_2d_generator(data_dict):
    shp = data_dict['data'].shape
    data_dict['data'] = data_dict['data'].reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
    data_dict['orig_shape_data'] = shp
    shp = data_dict['seg'].shape
    data_dict['seg'] = data_dict['seg'].reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
    data_dict['orig_shape_seg'] = shp
    shp = data_dict['weightmap'].shape
    data_dict['weightmap'] = data_dict['weightmap'].reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
    data_dict['orig_shape_weightmap'] = shp
    return data_dict


def convert_2d_to_3d_generator(data_dict):
    shp = data_dict['orig_shape_data']
    current_shape = data_dict['data'].shape
    data_dict['data'] = data_dict['data'].reshape((shp[0], shp[1], shp[2], current_shape[-2], current_shape[-1]))
    shp = data_dict['orig_shape_seg']
    current_shape_seg = data_dict['seg'].shape
    data_dict['seg'] = data_dict['seg'].reshape((shp[0], shp[1], shp[2], current_shape_seg[-2], current_shape_seg[-1]))
    shp = data_dict['orig_shape_weightmap']
    current_shape_weightmap = data_dict['weightmap'].shape
    data_dict['weightmap'] = data_dict['weightmap'].reshape(
        (shp[0], shp[1], shp[2], current_shape_weightmap[-2], current_shape_weightmap[-1]))
    return data_dict


class Convert3DTo2DTransformWithWeights(AbstractTransform):
    def __init__(self):
        pass

    def __call__(self, **data_dict):
        return convert_3d_to_2d_generator(data_dict)


class Convert2DTo3DTransformWithWeights(AbstractTransform):
    def __init__(self):
        pass

    def __call__(self, **data_dict):
        return convert_2d_to_3d_generator(data_dict)


class MirrorTransform2D(AbstractTransform):
    """ Randomly mirrors data along specified axes. Mirroring is evenly distributed. Probability of mirroring along
    each axis is 0.5

    Args:
        axes (tuple of int): axes along which to mirror

    """

    def __init__(self, axes=(0, 1), data_key="data", label_key="seg", p_per_sample=1):
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.label_key = label_key
        self.axes = axes
        if max(axes) > 2:
            raise ValueError("MirrorTransform now takes the axes as the spatial dimensions. What previously was "
                             "axes=(2, 3, 4) to mirror along all spatial dimensions of a 5d tensor (b, c, x, y, z) "
                             "is now axes=(0, 1, 2). Please adapt your scripts accordingly.")

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        for b in range(len(data)):
            if np.random.uniform() < self.p_per_sample:
                sample_seg = None
                if seg is not None:
                    sample_seg = seg[b]
                ret_val = self.augment_mirroring(data[b], sample_seg, axes=self.axes)
                data[b] = ret_val[0]
                if seg is not None:
                    seg[b] = ret_val[1]

        data_dict[self.data_key] = data
        if seg is not None:
            data_dict[self.label_key] = seg

        return data_dict

    @staticmethod
    def augment_mirroring(sample_data, sample_seg=None, axes=(0, 1, 2)):
        if (len(sample_data.shape) != 3) and (len(sample_data.shape) != 4):
            raise Exception(
                "Invalid dimension for sample_data and sample_seg. sample_data and sample_seg should be either "
                "[channels, x, y] or [channels, x, y, z]")
        if 0 in axes and np.random.uniform() < 0.5:
            sample_data[:, :, :] = sample_data[:, :, ::-1]
            if sample_seg is not None:
                sample_seg[:, :] = sample_seg[:, ::-1]
        if 1 in axes and np.random.uniform() < 0.5:
            sample_data[:, :, :, :] = sample_data[:, :, :, ::-1]
            if sample_seg is not None:
                sample_seg[:, :, :] = sample_seg[:, :, ::-1]
        return sample_data, sample_seg