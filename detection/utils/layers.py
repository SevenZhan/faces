import numpy as np

import torch
import torch.nn as nn



class RegressionTransform(nn.Module):
    def __init__(self, mean=None, std_box=None, std_ldm=None):
        super().__init__()

        if mean is None:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))
        else:
            self.mean = mean
        if std_box is None:
            self.std_box = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
        else:
            self.std_box = std_box
        if std_ldm is None:
            self.std_ldm = (torch.ones(1, 10) * 0.1)

    def forward(self, anchors, bbox_deltas, ldm_deltas, img):

        widths = anchors[:, :, 2] - anchors[:, :, 0]
        heights = anchors[:, :, 3] - anchors[:, :, 1]
        ctr_x = anchors[:, :, 0] + 0.5 * widths
        ctr_y = anchors[:, :, 1] + 0.5 * heights
        # Rescale
        ldm_deltas = ldm_deltas * self.std_ldm.cuda()
        bbox_deltas = bbox_deltas * self.std_box.cuda()

        bbox_dx = bbox_deltas[:, :, 0]
        bbox_dy = bbox_deltas[:, :, 1]
        bbox_dw = bbox_deltas[:, :, 2]
        bbox_dh = bbox_deltas[:, :, 3]
        # get predicted boxes
        pred_ctr_x = ctr_x + bbox_dx * widths
        pred_ctr_y = ctr_y + bbox_dy * heights
        pred_w = torch.exp(bbox_dw) * widths
        pred_h = torch.exp(bbox_dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        # get predicted landmarks
        pt0_x = ctr_x + ldm_deltas[:,:,0] * widths
        pt0_y = ctr_y + ldm_deltas[:,:,1] * heights
        pt1_x = ctr_x + ldm_deltas[:,:,2] * widths
        pt1_y = ctr_y + ldm_deltas[:,:,3] * heights
        pt2_x = ctr_x + ldm_deltas[:,:,4] * widths
        pt2_y = ctr_y + ldm_deltas[:,:,5] * heights
        pt3_x = ctr_x + ldm_deltas[:,:,6] * widths
        pt3_y = ctr_y + ldm_deltas[:,:,7] * heights
        pt4_x = ctr_x + ldm_deltas[:,:,8] * widths
        pt4_y = ctr_y + ldm_deltas[:,:,9] * heights

        pred_landmarks = torch.stack([pt0_x, pt0_y, pt1_x, pt1_y, pt2_x, pt2_y, pt3_x, pt3_y, pt4_x, pt4_y], dim=2)

        # clip bboxes and landmarks
        B, C, H, W = img.shape

        pred_boxes[:, :, ::2] = torch.clamp(pred_boxes[:, :, ::2], min=0, max=W)
        pred_boxes[:, :, 1::2] = torch.clamp(pred_boxes[:, :, 1::2], min=0, max=H)
        pred_landmarks[:, :, ::2] = torch.clamp(pred_landmarks[:, :, ::2], min=0, max=W)
        pred_landmarks[:, :, 1::2] = torch.clamp(pred_landmarks[:, :, 1::2], min=0, max=H)

        return pred_boxes, pred_landmarks



class Anchors(object):
    def __init__(self, pyramid_levels=None, strides=None, base_sizes=None, ratios=None, scales=None):

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5]
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        if base_sizes is None:
            self.base_sizes = [2 ** 4.0, 2 ** 6.0, 2 ** 8.0]
        if ratios is None:
            self.ratios = np.array([1, 1, 1])
        if scales is None:
            self.scales = np.array([2 ** 0, 2 ** (1 / 2.0), 2 ** 1.0])

    def __call__(self, input_size=[640, 640]):

        image_shape = np.array(input_size)
        image_shapes = [(image_shape + stride - 1) // stride for stride in self.strides]
        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 4)).astype(np.float32)

        for idx, p in enumerate(self.pyramid_levels):
            anchors = generate_anchors(base_size=self.base_sizes[idx], ratios=self.ratios, scales=self.scales)
            shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

        all_anchors = np.expand_dims(all_anchors, axis=0)

        return torch.from_numpy(all_anchors.astype(np.float32))



def generate_anchors(base_size=16, ratios=None, scales=None):

    if ratios is None:
        ratios = np.array([1, 1, 1])
    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(scales)
    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))
    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, 1)).T
    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def shift(shape, stride, anchors):

    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to cell K shifts (K, 1, 4) to get shift anchors (K, A, 4)
    # reshape to (K * A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors