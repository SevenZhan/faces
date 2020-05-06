import cv2
import numpy as np

import torch

from models import Retina
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
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), thickness=2)
            cv2.circle(img, (landmark[0], landmark[1]), radius=1, color=(0, 0, 255), thickness=2)
            cv2.circle(img, (landmark[2], landmark[3]), radius=1, color=(0, 255, 0), thickness=2)
            cv2.circle(img, (landmark[4], landmark[5]), radius=1, color=(255, 0, 0), thickness=2)
            cv2.circle(img, (landmark[6], landmark[7]), radius=1, color=(0, 255, 255), thickness=2)
            cv2.circle(img, (landmark[8], landmark[9]), radius=1, color=(255, 255, 0), thickness=2)
            cv2.putText(img, text=str(score.item())[:5], org=(box[0], box[1]), fontFace=cv2.FONT_HERSHEY_DUPLEX, thickness=1, color=(255, 255, 255))

cv2.imwrite('results/test.jpg', img)