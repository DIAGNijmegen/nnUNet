import os
from glob import glob
from os.path import isfile

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_pickle
from wholeslidedata import WholeSlideAnnotation, WholeSlideImage
from wholeslidedata.accessories.asap.parser import AsapAnnotationParser

from nnunet.training.dataloading.dataset_loading import DataLoader2D
import re

from nnunet.utilities.to_torch import maybe_to_torch


class DataLoader2DROIsMultiScale(DataLoader2D):
    def __init__(self, data_origin, spacing, crop_to_patch_size=True, training=True,
                 regex_pattern=r"(?P<file>\d+_\d+_\d+)_x_(?P<x>\d+)_y_(?P<y>\d+)", context_label_problem=None,
                 context_file_extension="svs",
                 *args, **kwargs):
        """
        This class is an extension of the original DataLoader2D that contained a lot of 3D specific code, that is now
        cleaned up. Additionally, this class is able to sample context around the sampled patch. Depending on the
        format of the data, you should either use DataLoader2DROIsMultiScaleFileName or
        DataLoader2DROIsMultiScaleCoordinatesFilename (see corresponding class for file name pattern).
        """
        self.context_file_extension = context_file_extension
        self.context_label_problem = context_label_problem
        self.training = training
        self.crop_to_patch_size = crop_to_patch_size
        self.data_origin = data_origin
        self.spacing = spacing
        self.regex_pattern = re.compile(regex_pattern)
        super(DataLoader2DROIsMultiScale, self).__init__(*args, **kwargs)
        self.max_num_class = int(max([values['properties']['classes'][-1] for key, values in self._data.items()]))

    def __next__(self):
        """
        Training batch are cropped patches from the ROI and single_roi is the entire ROI.
        """
        return self.generate_train_batch() if self.training else self.generate_single_roi()

    def generate_train_batch(self):
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)

        b, c, w, h = self.data_shape
        data = np.zeros((b, 2, c, w, h), dtype=np.float32)
        seg = np.zeros(self.seg_shape, dtype=np.float32)

        case_properties = []
        for j, i in enumerate(selected_keys):
            if 'properties' in self._data[i].keys():
                properties = self._data[i]['properties']
            else:
                properties = load_pickle(self._data[i]['properties_file'])
            case_properties.append(properties)

            case_all_data = self.load_data(selected_keys[j])

            # case all data should now be (c, x, y)
            assert len(case_all_data.shape) == 3

            shape = case_all_data.shape[1:]
            lb_x = 0  # Don't sample below this or you will get black borders
            # (future: sample ROIs larger, since CP has a lot of WSI)
            ub_x = shape[0] - self.patch_size[0]
            lb_y = 0
            ub_y = shape[1] - self.patch_size[1]

            # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
            # at least one of the foreground classes in the patch
            bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
            bbox_y_lb = np.random.randint(lb_y, ub_y + 1)

            bbox_x_ub = bbox_x_lb + self.patch_size[0]
            bbox_y_ub = bbox_y_lb + self.patch_size[1]

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_x_lb = max(0, bbox_x_lb)
            valid_bbox_x_ub = min(shape[0], bbox_x_ub)
            valid_bbox_y_lb = max(0, bbox_y_lb)
            valid_bbox_y_ub = min(shape[1], bbox_y_ub)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)

            width, height = case_all_data.shape[-2:]

            offset_x = (valid_bbox_x_lb + (valid_bbox_x_ub - valid_bbox_x_lb) // 2) - width // 2
            offset_y = (valid_bbox_y_lb + (valid_bbox_y_ub - valid_bbox_y_lb) // 2) - height // 2

            case_properties[j]['offset_x'] = int(offset_y)
            case_properties[j]['offset_y'] = int(offset_x)

            case_all_data = case_all_data[:, valid_bbox_x_lb:valid_bbox_x_ub, valid_bbox_y_lb:valid_bbox_y_ub]

            case_all_data_donly = np.pad(case_all_data[:-1], ((0, 0),
                                                              (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                              (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                         self.pad_mode, **self.pad_kwargs_data)

            case_all_data_segonly = np.pad(case_all_data[-1:], ((0, 0),
                                                                (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                                (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                           'constant', **{'constant_values': -1})
            case_properties[j]['context_label'] = self.get_label_for_loss(case_all_data_segonly, i)
            data[j] = np.stack([case_all_data_donly,
                                self.sample_context(case_properties[j], selected_keys[j],
                                                    case_all_data_donly.shape[-2:])])
            seg[j] = case_all_data_segonly

        keys = selected_keys
        return {'data': data, 'seg': seg, 'properties': case_properties, "keys": keys}

    def generate_single_roi(self):
        if not len(self.list_of_keys):
            raise StopIteration
        selected_key = self.list_of_keys.pop()

        if 'properties' in self._data[selected_key].keys():
            properties = self._data[selected_key]['properties']
        else:
            properties = load_pickle(self._data[selected_key]['properties_file'])

        case_all_data = self.load_data(selected_key)

        c, w, h = case_all_data.shape

        data = np.stack([case_all_data[:-1], self.sample_context(properties, selected_key, (w, h))])

        return {'data': data[np.newaxis],
                'seg': np.array(case_all_data[-1:])[np.newaxis],
                'properties': properties,
                "keys": [selected_key]}

    def load_data(self, selected_key):
        if not isfile(self._data[selected_key]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[selected_key]['data_file'][:-4] + ".npz")['data'][:, None][:, 0]
        else:
            case_all_data = np.load(self._data[selected_key]['data_file'][:-4] + ".npy", self.memmap_mode)[:, 0]
        return case_all_data

    def sample_context(self, props, key, shape=None):
        pass

    def get_label_for_loss(self, case_all_data_segonly, i):
        if self.context_label_problem == 'multi_label':
            labels = np.zeros(self.max_num_class + 1)
            labels[self._data[i]['properties']['classes'].astype(int)] = 1
            return maybe_to_torch(labels)
        elif self.context_label_problem == 'regression':
            labels = np.zeros(self.max_num_class + 1)
            lbls, counts = np.unique(case_all_data_segonly, return_counts=True)
            labels[lbls.astype(int)] = counts / counts.sum()
            return maybe_to_torch(labels)


class DataLoader2DROIsMultiScaleFileName(DataLoader2DROIsMultiScale):
    def __init__(self, *args, **kwargs):
        """
        This samples context using a specific formatting of the filename: KEY_ROI{NUMBER}
        It only works whenever the WSI is segmented using ROIs. These ROIs (annotations), should not be a part of
        a group (in ASAP for example). Only then will they be named 'none' and will the sampling work (see code below)
        """
        super(DataLoader2DROIsMultiScaleFileName, self).__init__(*args, **kwargs)

    def sample_context(self, props, key, shape=None):
        parser = AsapAnnotationParser(labels={'none': 0}, sample_label_names=['none'])
        anno_number = int(key.split("_")[-1].strip("ROI"))
        file_identifier = os.path.join(self.data_origin, '_'.join(key.split('_')[:-1]))
        wsa = WholeSlideAnnotation(glob(f"{file_identifier}.xml")[0], parser=parser)
        wsi = WholeSlideImage(glob(f"{file_identifier}.{self.context_file_extension}")[0], backend='asap')
        anno = wsa.sampling_annotations[anno_number]
        x, y = anno.center
        x += props.get("offset_x", 0)
        y += props.get("offset_y", 0)
        return wsi.get_patch(x, y, *shape[::-1], spacing=self.spacing).transpose(2, 0, 1) / 255.


class DataLoader2DROIsMultiScaleCoordinatesFilename(DataLoader2DROIsMultiScale):
    def __init__(self, *args, **kwargs):
        """
        Samples context based on coordinates in the filename: KEY_X_{x_coordinate}_Y_{y_coordinate}
        """
        super(DataLoader2DROIsMultiScaleCoordinatesFilename, self).__init__(*args, **kwargs)

    def sample_context(self, props, key, shape=None):
        matches = re.search(self.regex_pattern, key)
        file_name = matches.group("file")
        x = int(matches.group("x"))
        y = int(matches.group("y"))
        file_identifier = os.path.join(self.data_origin, file_name)
        wsi = WholeSlideImage(glob(f"{file_identifier}.{self.context_file_extension}")[0], backend='asap')
        x += props.get("offset_x", 0)
        y += props.get("offset_y", 0)
        return wsi.get_patch(x, y, *shape[::-1], spacing=self.spacing).transpose(2, 0, 1) / 255.
