
import torch
import numpy as np
import cv2
import time

from IPython import embed


def get_center(x):
    return (x - 1.) / 2.


def xyxy2cxcywh(bbox):
    return get_center(bbox[0] + bbox[2]), \
        get_center(bbox[1] + bbox[3]), \
        (bbox[2] - bbox[0]), \
        (bbox[3] - bbox[1])


def cxcywh2xyxy(bboxes):
    if len(np.array(bboxes).shape) == 1:
        bboxes = np.array(bboxes)[None, :]
    else:
        bboxes = np.array(bboxes)
    x1 = bboxes[:, 0:1] + 1 / 2 - bboxes[:, 2:3] / 2.
    x2 = x1 + bboxes[:, 2:3] - 1
    y1 = bboxes[:, 1:2] + 1 / 2 - bboxes[:, 3:4] / 2.
    y2 = y1 + bboxes[:, 3:4] - 1
    return np.concatenate([x1, y1, x2, y2], 1)


def nms(bboxes, scores, num, threshold=0.7):
    sort_index = np.argsort(scores)[::-1]
    sort_boxes = bboxes[sort_index]
    selected_bbox = [sort_boxes[0]]
    selected_index = [sort_index[0]]
    for i, bbox in enumerate(sort_boxes):
        iou = compute_iou(selected_bbox, bbox)
        # print(iou, bbox, selected_bbox)
        if np.max(iou) < threshold:
            selected_bbox.append(bbox)
            selected_index.append(sort_index[i])
            if len(selected_bbox) >= num:
                break
    return selected_index


def nms_worker(x, threshold=0.7):
    bboxes, scores, num = x
    if len(bboxes) == 0:
        selected_index = [0]
    else:
        sort_index = np.argsort(scores)[::-1]
        sort_boxes = bboxes[sort_index]
        selected_bbox = [sort_boxes[0]]
        selected_index = [sort_index[0]]
        for i, bbox in enumerate(sort_boxes):
            iou = compute_iou(selected_bbox, bbox)
            # print(iou, bbox, selected_bbox)
            if np.max(iou) < threshold:
                selected_bbox.append(bbox)
                selected_index.append(sort_index[i])
                if len(selected_bbox) >= num:
                    break
    return selected_index


def round_up(value):
    return round(value + 1e-6 + 1000) - 1000


def crop_and_pad(img, cx, cy, model_sz, original_sz, img_mean=None):
    img_mean = [int(img[:, :, 0].mean()), int(
        img[:, :, 1].mean()), int(img[:, :, 2].mean())]
    im_h, im_w, _ = img.shape

    xmin = cx - (original_sz - 1) / 2.
    xmax = xmin + original_sz - 1
    ymin = cy - (original_sz - 1) / 2.
    ymax = ymin + original_sz - 1

    left = int(round_up(max(0., -xmin)))
    top = int(round_up(max(0., -ymin)))
    right = int(round_up(max(0., xmax - im_w + 1)))
    bottom = int(round_up(max(0., ymax - im_h + 1)))

    xmin = int(round_up(xmin + left))
    xmax = int(round_up(xmax + left))
    ymin = int(round_up(ymin + top))
    ymax = int(round_up(ymax + top))
    r, c, k = img.shape
    if any([top, bottom, left, right]):
        # 0 is better than 1 initialization
        te_im = np.zeros((r + top + bottom, c + left + right, k), np.uint8)
        te_im[top:top + r, left:left + c, :] = img
        if top:
            te_im[0:top, left:left + c, :] = img_mean
        if bottom:
            te_im[r + top:, left:left + c, :] = img_mean
        if left:
            te_im[:, 0:left, :] = img_mean
        if right:
            te_im[:, c + left:, :] = img_mean
        im_patch_original = te_im[int(ymin):int(
            ymax + 1), int(xmin):int(xmax + 1), :]
    else:
        im_patch_original = img[int(ymin):int(
            ymax + 1), int(xmin):int(xmax + 1), :]
    if not np.array_equal(model_sz, original_sz):
        # zzp: use cv to get a better speed
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    else:
        im_patch = im_patch_original
    scale = float(model_sz) / im_patch_original.shape[0]
    return im_patch, scale


def get_exemplar_image(img, bbox, size_z, context_amount, img_mean=None):
    cx, cy, w, h = bbox
    wc_z = w + context_amount * (w + h)
    hc_z = h + context_amount * (w + h)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = float(size_z) / s_z
    exemplar_img, _ = crop_and_pad(img, cx, cy, size_z, s_z, img_mean)
    return exemplar_img, scale_z, s_z


def get_instance_image(img, bbox, size_z, size_x, context_amount, img_mean=None):
    cx, cy, w, h = bbox  # float type
    wc_z = w + context_amount * (w + h)
    hc_z = h + context_amount * (w + h)
    s_z = np.sqrt(wc_z * hc_z)  # the width of the crop box
    scale_z = float(size_z) / s_z

    s_x = s_z * size_x / size_z
    instance_img, scale_x = crop_and_pad(img, cx, cy, size_x, s_x, img_mean)
    w_x = w * scale_x
    h_x = h * scale_x
    # point_1 = (size_x + 1) / 2 - w_x / 2, (size_x + 1) / 2 - h_x / 2
    # point_2 = (size_x + 1) / 2 + w_x / 2, (size_x + 1) / 2 + h_x / 2
    # frame = cv2.rectangle(instance_img, (int(point_1[0]),int(point_1[1])), (int(point_2[0]),int(point_2[1])), (0, 255, 0), 2)
    # cv2.imwrite('1.jpg', frame)
    return instance_img, w_x, h_x, scale_x


def box_transform(anchors, gt_box):
    eps = 1e-5
    anchor_xctr = anchors[:, :1]
    anchor_yctr = anchors[:, 1:2]
    anchor_w = anchors[:, 2:3]
    anchor_h = anchors[:, 3:]
    gt_cx, gt_cy, gt_w, gt_h = gt_box

    target_x = (gt_cx - anchor_xctr) / (anchor_w + eps)
    target_y = (gt_cy - anchor_yctr) / (anchor_h + eps)
    target_w = np.log(gt_w / (anchor_w + eps))
    target_h = np.log(gt_h / (anchor_h + eps))
    regression_target = np.hstack((target_x, target_y, target_w, target_h))
    return regression_target


def box_transform_inv(anchors, offset):
    anchor_xctr = anchors[:, :1]
    anchor_yctr = anchors[:, 1:2]
    anchor_w = anchors[:, 2:3]
    anchor_h = anchors[:, 3:]
    offset_x, offset_y, offset_w, offset_h = offset[:,
                                                    :1], offset[:, 1:2], offset[:, 2:3], offset[:, 3:],

    box_cx = anchor_w * offset_x + anchor_xctr
    box_cy = anchor_h * offset_y + anchor_yctr
    box_w = anchor_w * np.exp(offset_w)
    box_h = anchor_h * np.exp(offset_h)
    box = np.hstack([box_cx, box_cy, box_w, box_h])
    return box


def get_topk_box(cls_score, pred_regression, anchors, topk=10):
    # anchors xc,yc,w,h
    regress_offset = pred_regression.cpu().detach().numpy()

    scores, index = torch.topk(cls_score, topk, )
    index = index.view(-1).cpu().detach().numpy()

    topk_offset = regress_offset[index, :]
    anchors = anchors[index, :]
    pred_box = box_transform_inv(anchors, topk_offset)
    return pred_box


def compute_iou(anchors, box):
    if np.array(anchors).ndim == 1:
        anchors = np.array(anchors)[None, :]
    else:
        anchors = np.array(anchors)
    if np.array(box).ndim == 1:
        box = np.array(box)[None, :]
    else:
        box = np.array(box)
    gt_box = np.tile(box.reshape(1, -1), (anchors.shape[0], 1))

    anchor_x1 = anchors[:, :1] - anchors[:, 2:3] / 2. + 0.5
    anchor_x2 = anchors[:, :1] + anchors[:, 2:3] / 2. - 0.5
    anchor_y1 = anchors[:, 1:2] - anchors[:, 3:] / 2. + 0.5
    anchor_y2 = anchors[:, 1:2] + anchors[:, 3:] / 2. - 0.5

    gt_x1 = gt_box[:, :1] - gt_box[:, 2:3] / 2. + 0.5
    gt_x2 = gt_box[:, :1] + gt_box[:, 2:3] / 2. - 0.5
    gt_y1 = gt_box[:, 1:2] - gt_box[:, 3:] / 2. + 0.5
    gt_y2 = gt_box[:, 1:2] + gt_box[:, 3:] / 2. - 0.5

    xx1 = np.max([anchor_x1, gt_x1], axis=0)
    xx2 = np.min([anchor_x2, gt_x2], axis=0)
    yy1 = np.max([anchor_y1, gt_y1], axis=0)
    yy2 = np.min([anchor_y2, gt_y2], axis=0)

    inter_area = np.max([xx2 - xx1, np.zeros(xx1.shape)], axis=0) * np.max([yy2 - yy1, np.zeros(xx1.shape)],
                                                                           axis=0)
    area_anchor = (anchor_x2 - anchor_x1) * (anchor_y2 - anchor_y1)
    area_gt = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    iou = inter_area / (area_anchor + area_gt - inter_area + 1e-6)
    return iou


def get_pyramid_instance_image(img, center, size_x, size_x_scales, img_mean=None):
    if img_mean is None:
        img_mean = tuple(map(int, img.mean(axis=(0, 1))))
    pyramid = [crop_and_pad(img, center[0], center[1], size_x, size_x_scale, img_mean)
               for size_x_scale in size_x_scales]
    return pyramid


def add_box_img(img, boxes, color=(0, 255, 0)):
    # boxes (x,y,w,h)
    if boxes.ndim == 1:
        boxes = boxes[None, :]
    img = img.copy()
    img_ctx = (img.shape[1] - 1) / 2.
    img_cty = (img.shape[0] - 1) / 2.
    for box in boxes:
        point_1 = [img_ctx - box[2] / 2. + box[0] +
                   0.5, img_cty - box[3] / 2. + box[1] + 0.5]
        point_2 = [img_ctx + box[2] / 2. + box[0] -
                   0.5, img_cty + box[3] / 2. + box[1] - 0.5]
        point_1[0] = np.clip(point_1[0], 0, img.shape[1])
        point_2[0] = np.clip(point_2[0], 0, img.shape[1])
        point_1[1] = np.clip(point_1[1], 0, img.shape[0])
        point_2[1] = np.clip(point_2[1], 0, img.shape[0])
        img = cv2.rectangle(img, (int(point_1[0]), int(point_1[1])), (int(point_2[0]), int(point_2[1])),
                            color, 2)
    return img


def adjust_learning_rate(optimizer, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']
