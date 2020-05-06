import torch
import torchvision.ops as ops



def get_detections(img_batch, model, score_threshold=0.5, iou_threshold=0.5):

    model.eval()
    with torch.no_grad():
        classifications, bboxes, landmarks = model(img_batch)
        batch_size = classifications.shape[0]
        picked_boxes = []
        picked_landmarks = []
        picked_scores = []

        for i in range(batch_size):
            classification = torch.exp(classifications[i, :, :])
            bbox = bboxes[i, :, :]
            landmark = landmarks[i, :, :]

            # choose positive and scores > score_threshold
            scores, argmax = torch.max(classification, dim=1)
            argmax_indice = argmax == 0
            scores_indice = scores > score_threshold
            positive_indices = argmax_indice & scores_indice

            scores = scores[positive_indices]

            if scores.shape[0] == 0:
                picked_boxes.append(None)
                picked_landmarks.append(None)
                picked_scores.append(None)
                continue

            bbox = bbox[positive_indices]
            landmark = landmark[positive_indices]

            keep = ops.boxes.nms(bbox, scores, iou_threshold)
            keep_boxes = bbox[keep]
            keep_landmarks = landmark[keep]
            keep_scores = scores[keep]
            keep_scores.unsqueeze_(1)
            picked_boxes.append(keep_boxes)
            picked_landmarks.append(keep_landmarks)
            picked_scores.append(keep_scores)

        return picked_boxes, picked_landmarks, picked_scores



def evaluate(val_data, retinaFace, threshold=0.5):
    recall = 0.
    precision = 0.

    for data in val_data:
        img_batch = data['img'].cuda()
        annots = data['annot'].cuda()

        picked_boxes, _, _ = get_detections(img_batch, retinaFace)
        recall_iter = 0.
        precision_iter = 0.

        for j, boxes in enumerate(picked_boxes):
            annot_boxes = annots[j]
            annot_boxes = annot_boxes[annot_boxes[:, 0] != -1]

            if boxes is None and annot_boxes.shape[0] == 0:
                continue
            elif boxes is None and annot_boxes.shape[0] != 0:
                recall_iter += 0.
                precision_iter += 1.
                continue
            elif boxes is not None and annot_boxes.shape[0] == 0:
                recall_iter += 1.
                precision_iter += 0.
                continue

            overlap = ops.boxes.box_iou(annot_boxes, boxes)

            # compute recall
            max_overlap, _ = torch.max(overlap, dim=1)
            mask = max_overlap > threshold
            detected_num = mask.sum().item()
            recall_iter += detected_num / annot_boxes.shape[0]

            # compute precision
            max_overlap, _ = torch.max(overlap, dim=0)
            mask = max_overlap > threshold
            true_positives = mask.sum().item()
            precision_iter += true_positives / boxes.shape[0]

        recall += recall_iter / len(picked_boxes)
        precision += precision_iter / len(picked_boxes)

    return recall / len(val_data), precision / len(val_data)