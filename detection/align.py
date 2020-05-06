# import cv2
# import time
# import argparse
# import numpy as np
#
# import torch
# from torchvision.ops import nms
# import torch.backends.cudnn as cudnn
#
# from models import Retina
# from utils import norm_crop, Anchors
#
#
#
#
# torch.set_grad_enabled(False)
# checkpoint = ''
# model = Retina()
# model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
# model.eval()
#
# for i in range(1):
#     image_path = 'samples/test.jpg'
#     img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
#
#     img = np.float32(img_raw)
#
#     im_height, im_width, _ = img.shape
#     scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
#
#     img = img.transpose(2, 0, 1)
#     img = torch.from_numpy(img).unsqueeze(0)
#     img = img
#     scale = scale
#
#     loc, conf, landms = model(img)
#
#     Anchor = Anchors()()
#     priors = Anchor.forward()
#     priors = priors.to(device)
#     prior_data = priors.data
#     boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
#     boxes = boxes * scale / resize
#     scores = conf.data.squeeze(0)[:, 1]
#     landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
#     scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
#                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
#                            img.shape[3], img.shape[2]])
#     scale1 = scale1.to(device)
#     landms = landms * scale1 / resize
#
#     # ignore low scores
#     inds = torch.where(scores > args.confidence_threshold)[0]
#     boxes = boxes[inds]
#     landms = landms[inds]
#     scores = scores[inds]
#
#     # do NMS
#     keep = nms(boxes, scores, 0.4)
#     dets = torch.cat((boxes, scores[:, np.newaxis]), dim=-1)
#     dets = dets[keep, :]
#     landms = landms[keep]
#
#     dets = torch.cat((dets, landms), axis=-1)
#
# # save image
# for i, b in enumerate(dets):
#     box = b[:4].numpy()
#     score = b[4].numpy()
#     lmrk = b[5:].reshape(-1, 2).numpy()
#     if score >= 0.6:
#         aligned_face = norm_crop(img_raw, lmrk, mode='face')
#         name = 'results/face_{}.jpg'.format(i)
#         cv2.imwrite(name, aligned_face)







import cv2
import numpy as np

import torch

from models import Retina
from utils import norm_crop
from valid import get_detections



checkpoint = ''
model = Retina()
model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
model.eval()



# Read image
img = cv2.imread('')
img = torch.from_numpy(img)
img = img.permute(2, 0, 1)
input_img = img.unsqueeze(0).float()
picked_boxes, picked_landmarks, picked_scores = get_detections(input_img, model, score_threshold=0.5, iou_threshold=0.3)

np_img = img.cpu().permute(1, 2, 0).numpy()
np_img.astype(int)
img = cv2.cvtColor(np_img.astype(np.uint8), cv2.COLOR_BGR2RGB)

for j, boxes in enumerate(picked_boxes):
    if boxes is not None:
        for box, landmark, score in zip(boxes, picked_landmarks[j], picked_scores[j]):
            if score >= 0.6:
                landmark = landmark.view(-1, 2).numpy()
                aligned_face = norm_crop(img, landmark, mode='face')
                name = 'results/face_{}.jpg'.format(1)
                cv2.imwrite(name, aligned_face)