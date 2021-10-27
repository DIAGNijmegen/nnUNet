"""
Alternative implementations of some memory-inefficient functions
"""

import numpy as np

from skimage.transform import resize
from scipy.ndimage import zoom


def resample_data_or_seg(data, new_shape, is_seg, axis=None, order=3, do_separate_z=False, cval=0, order_z=0):
    """
    separate_z=True will resample with order 0 along z
    :param data:
    :param new_shape:
    :param is_seg:
    :param axis:
    :param order:
    :param do_separate_z:
    :param cval:
    :param order_z: only applies if do_separate_z is True
    :return:
    """
    assert len(data.shape) == 4, "data must be (c, x, y, z)"
    assert not is_seg, "do not use this patch for resampling segmentations"

    print("running patched resample_data_or_seg function")

    dtype_data = data.dtype
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if np.all(shape == new_shape):
        print("no resampling necessary")
        return data

    data = data.astype(float)
    resize_fn = resize
    kwargs = {'mode': 'edge', 'anti_aliasing': False}

    if do_separate_z:
        print("separate z, order in z is", order_z, "order inplane is", order)
        assert len(axis) == 1, "only one anisotropic axis supported"
        axis = axis[0]
        if axis == 0:
            new_shape_2d = new_shape[1:]
        elif axis == 1:
            new_shape_2d = new_shape[[0, 2]]
        else:
            new_shape_2d = new_shape[:-1]

        reshaped_final_data = np.empty(shape=(data.shape[0], new_shape[0], new_shape[1], new_shape[2]), dtype=dtype_data)
        do_z = shape[axis] != new_shape[axis]
        if do_z:
            if axis == 0:
                buffer = np.empty(shape=(shape[axis], new_shape_2d[0], new_shape_2d[1]), dtype=float)
            elif axis == 1:
                buffer = np.empty(shape=(new_shape_2d[0], shape[axis], new_shape_2d[1]), dtype=float)
            else:
                buffer = np.empty(shape=(new_shape_2d[0], new_shape_2d[1], shape[axis]), dtype=float)
        else:
            buffer = None

        for c in range(data.shape[0]):
            if do_z:
                reshaped_data = buffer
            else:
                reshaped_data = reshaped_final_data[c]

            for slice_id in range(shape[axis]):
                if axis == 0:
                    reshaped_data[slice_id, :, :] = resize_fn(data[c, slice_id], new_shape_2d, order, cval=cval, **kwargs)
                elif axis == 1:
                    reshaped_data[:, slice_id, :] = resize_fn(data[c, :, slice_id], new_shape_2d, order, cval=cval, **kwargs)
                else:
                    reshaped_data[:, :, slice_id] = resize_fn(data[c, :, :, slice_id], new_shape_2d, order, cval=cval, **kwargs)

            if do_z:
                # The following few lines are blatantly copied and modified from sklearn's resize()
                rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                orig_rows, orig_cols, orig_dim = reshaped_data.shape

                row_scale = float(orig_rows) / rows
                col_scale = float(orig_cols) / cols
                dim_scale = float(orig_dim) / dim

                reshaped_final_data[c] = zoom(reshaped_data, (1 / row_scale, 1 / col_scale, 1 / dim_scale), order=order_z, cval=cval, mode='nearest')
    else:
        print("no separate z, order", order)
        reshaped_final_data = np.empty(shape=(data.shape[0], new_shape[0], new_shape[1], new_shape[2]), dtype=dtype_data)
        for c in range(data.shape[0]):
            reshaped_final_data[c] = resize_fn(data[c], new_shape, order, cval=cval, **kwargs)
    return reshaped_final_data
