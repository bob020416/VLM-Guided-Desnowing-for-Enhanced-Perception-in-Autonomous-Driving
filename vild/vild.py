import random
import cv2
import numpy as np
import torch
import clip
import numpy as np
import tensorflow.compat.v1 as tf
import os

from tqdm import tqdm
from easydict import EasyDict
from PIL import Image
from scipy.special import softmax

model = None
session = None
ckpt_path = '/home/hcis-s17/multimodal_manipulation/patrick/CV/VLM-Guided-Desnowing-for-Enhanced-Perception-in-Autonomous-Driving/vild/ckpt' 

def get_vild_bbox(image_path, object_list, params):
    global model, session
    FLAGS = {
        'prompt_engineering': True,
        'this_is': True,

        'temperature': 100.0,
        'use_softmax': False,
    }
    FLAGS = EasyDict(FLAGS)


    def article(name):
        return 'an' if name[0] in 'aeiou' else 'a'

    def processed_name(name, rm_dot=False):
        # _ for lvis
        # / for obj365
        res = name.replace('_', ' ').replace('/', ' or ').lower()
        if rm_dot:
            res = res.rstrip('.')
        return res
    single_template = [
        'a photo of {article} {}.'
    ]

    multiple_templates = [
        'There is {article} {} in the scene.',
        'There is the {} in the scene.',
        'a photo of {article} {} in the scene.',
        'a photo of the {} in the scene.',
        'a photo of one {} in the scene.',


        'itap of {article} {}.',
        'itap of my {}.',  # itap: I took a picture of
        'itap of the {}.',
        'a photo of {article} {}.',
        'a photo of my {}.',
        'a photo of the {}.',
        'a photo of one {}.',
        'a photo of many {}.',

        'a good photo of {article} {}.',
        'a good photo of the {}.',
        'a bad photo of {article} {}.',
        'a bad photo of the {}.',
        'a photo of a nice {}.',
        'a photo of the nice {}.',
        'a photo of a cool {}.',
        'a photo of the cool {}.',
        'a photo of a weird {}.',
        'a photo of the weird {}.',

        'a photo of a small {}.',
        'a photo of the small {}.',
        'a photo of a large {}.',
        'a photo of the large {}.',

        'a photo of a clean {}.',
        'a photo of the clean {}.',
        'a photo of a dirty {}.',
        'a photo of the dirty {}.',

        'a bright photo of {article} {}.',
        'a bright photo of the {}.',
        'a dark photo of {article} {}.',
        'a dark photo of the {}.',

        'a photo of a hard to see {}.',
        'a photo of the hard to see {}.',
        'a low resolution photo of {article} {}.',
        'a low resolution photo of the {}.',
        'a cropped photo of {article} {}.',
        'a cropped photo of the {}.',
        'a close-up photo of {article} {}.',
        'a close-up photo of the {}.',
        'a jpeg corrupted photo of {article} {}.',
        'a jpeg corrupted photo of the {}.',
        'a blurry photo of {article} {}.',
        'a blurry photo of the {}.',
        'a pixelated photo of {article} {}.',
        'a pixelated photo of the {}.',

        'a black and white photo of the {}.',
        'a black and white photo of {article} {}.',

        'a plastic {}.',
        'the plastic {}.',

        'a toy {}.',
        'the toy {}.',
        'a plushie {}.',
        'the plushie {}.',
        'a cartoon {}.',
        'the cartoon {}.',

        'an embroidered {}.',
        'the embroidered {}.',

        'a painting of the {}.',
        'a painting of a {}.',
    ]


    def build_text_embedding(categories):
        if FLAGS.prompt_engineering:
            templates = multiple_templates
        else:
            templates = single_template

        run_on_gpu = torch.cuda.is_available()

        with torch.no_grad():
            all_text_embeddings = []
            for category in categories:
                texts = [
                    template.format(processed_name(category['name'], rm_dot=True),
                                    article=article(category['name']))for template in templates
                    ]
                if FLAGS.this_is:
                    texts = [
                            'This is ' + text if text.startswith('a') or text.startswith('the') else text
                            for text in texts
                    ]
                texts = clip.tokenize(texts) #tokenize
                if run_on_gpu:
                    texts = texts.cuda()
                text_embeddings = model.encode_text(texts) #embed with text encoder
                text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
                text_embedding = text_embeddings.mean(dim=0)
                text_embedding /= text_embedding.norm()
                all_text_embeddings.append(text_embedding)
            all_text_embeddings = torch.stack(all_text_embeddings, dim=1)
            if run_on_gpu:
                all_text_embeddings = all_text_embeddings.cuda()
        return all_text_embeddings.cpu().numpy().T

    def nms(dets, scores, thresh, max_dets=1000):
        """Non-maximum suppression.
        Args:
        dets: [N, 4]
        scores: [N,]
        thresh: iou threshold. Float
        max_dets: int.
        """
        y1 = dets[:, 0]
        x1 = dets[:, 1]
        y2 = dets[:, 2]
        x2 = dets[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0 and len(keep) < max_dets:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            overlap = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-12)

            inds = np.where(overlap <= thresh)[0]
            order = order[inds + 1]
        return keep

    #################################################################
    # Preprocessing categories and get params
    object_list = ['background'] + object_list
    categories = [{'name': item, 'id': idx+1,} for idx, item in enumerate(object_list)]
    prob_records = [0.] * len(object_list)

    max_boxes_to_draw, nms_threshold, min_rpn_score_thresh, min_box_area = params


    #################################################################
    # Obtain results and read image
    roi_boxes, roi_scores, detection_boxes, scores_unused, box_outputs, detection_masks, visual_features, image_info = session.run(
        ['RoiBoxes:0', 'RoiScores:0', '2ndStageBoxes:0', '2ndStageScoresUnused:0', 'BoxOutputs:0', 'MaskOutputs:0', 'VisualFeatOutputs:0', 'ImageInfo:0'],
        feed_dict={'Placeholder:0': [image_path,]}
    )
    roi_boxes = np.squeeze(roi_boxes, axis=0)  # squeeze
    # no need to clip the boxes, already done
    roi_scores = np.squeeze(roi_scores, axis=0)

    detection_boxes = np.squeeze(detection_boxes, axis=(0, 2))
    scores_unused = np.squeeze(scores_unused, axis=0)
    box_outputs = np.squeeze(box_outputs, axis=0)
    detection_masks = np.squeeze(detection_masks, axis=0)
    visual_features = np.squeeze(visual_features, axis=0)

    image_info = np.squeeze(image_info, axis=0)  # obtain image info
    image_scale = np.tile(image_info[2:3, :], (1, 2))
    image_height = int(image_info[0, 0])
    image_width = int(image_info[0, 1])

    rescaled_detection_boxes = detection_boxes / image_scale # rescale

    # Read image
    image = np.asarray(Image.open(open(image_path, 'rb')).convert("RGB"))
    assert image_height == image.shape[0]
    assert image_width == image.shape[1]


    #################################################################
    # Filter boxes

    # Apply non-maximum suppression to detected boxes with nms threshold.
    nmsed_indices = nms(
        detection_boxes,
        roi_scores,
        thresh=nms_threshold
    )

    # Compute RPN box size.
    box_sizes = (rescaled_detection_boxes[:, 2] - rescaled_detection_boxes[:, 0]) * (rescaled_detection_boxes[:, 3] - rescaled_detection_boxes[:, 1])

    # Filter out invalid rois (nmsed rois)
    valid_indices = np.where(
        np.logical_and(
            np.isin(np.arange(len(roi_scores), dtype=int), nmsed_indices),
            np.logical_and(
                np.logical_not(np.all(roi_boxes == 0., axis=-1)),
                np.logical_and(roi_scores >= min_rpn_score_thresh, box_sizes > min_box_area)
            )
        )
    )[0]
    detection_boxes = detection_boxes[valid_indices][max_boxes_to_draw:, ...]
    detection_masks = detection_masks[valid_indices][max_boxes_to_draw:, ...]
    detection_visual_feat = visual_features[valid_indices][max_boxes_to_draw:, ...]
    rescaled_detection_boxes = rescaled_detection_boxes[valid_indices][max_boxes_to_draw:, ...] 


    #################################################################
    # Compute text embeddings and detection scores, and rank results
    text_features = build_text_embedding(categories)

    raw_scores = detection_visual_feat.dot(text_features.T)
    if FLAGS.use_softmax:
        scores_all = softmax(FLAGS.temperature * raw_scores, axis=-1)
    else:
        scores_all = raw_scores

    indices = np.argsort(-np.max(scores_all, axis=1))  # Results are ranked by scores
    indices_fg = np.array([i for i in indices if np.argmax(scores_all[i]) != 0])
    max_bbox = [float('inf'), float('inf'), float('-inf'), float('-inf')]
    bbox_list = []

    for anno_idx in indices_fg[:max_boxes_to_draw]:
        bbox = rescaled_detection_boxes[anno_idx]
        bbox_list.append(bbox)
        max_bbox = [min(max_bbox[i], bbox[i]) if i <= 1 else max(max_bbox[i], bbox[i]) for i in range(4)]
    return max_bbox, bbox_list
        
def main():
    global model, session, ckpt_path
    clip.available_models()
    model, preprocess = clip.load("ViT-B/32")

    session = tf.Session(graph=tf.Graph())

    saved_model_dir = ckpt_path

    _ = tf.saved_model.load(session, ['serve'], saved_model_dir)

        
    img_root = '/home/hcis-s17/multimodal_manipulation/patrick/CV/final/drive_dataset/snow_resized'
    object_list = ['car', 'pedestrian', 'vehicle', 'person', 'bicycle', 'truck', 'bus', 'motorcycle', 'traffic light', 'stop sign']
    clear_img_postfix = 'leftImg8bit.png'
    
    max_boxes_to_draw = 20
    nms_threshold = 0.6
    min_rpn_score_thresh = 0.9
    min_box_area = 220
    params = max_boxes_to_draw, nms_threshold, min_rpn_score_thresh, min_box_area
    for img in tqdm(os.listdir(img_root), ncols=100, leave=False):
        image_path = os.path.join(img_root, img)
        if not image_path.endswith('.png') and not image_path.endswith('.jpg'):
            continue
        max_bbox, _ = get_vild_bbox(image_path, object_list, params)
        with open(image_path.replace(clear_img_postfix, 'bbox.txt'), 'w') as f:
            f.write('\t'.join(map(str, max_bbox)))

def show_example(image_path):
    
    global model, session, ckpt_path
    clip.available_models()
    model, preprocess = clip.load("ViT-B/32")

    session = tf.Session(graph=tf.Graph())

    saved_model_dir = ckpt_path

    _ = tf.saved_model.load(session, ['serve'], saved_model_dir)
    object_list = ['car', 'pedestrian', 'vehicle', 'person', 'bicycle', 'truck', 'bus', 'motorcycle', 'traffic light', 'stop sign']
    
    max_boxes_to_draw = 50
    nms_threshold = 0.6
    min_rpn_score_thresh = 0.9
    min_box_area = 220
    params = max_boxes_to_draw, nms_threshold, min_rpn_score_thresh, min_box_area
    if not image_path.endswith('.png') and not image_path.endswith('.jpg'):
        raise Exception('Invalid image path')
    max_bbox, bbox_list = get_vild_bbox(image_path, object_list, params)
    image = cv2.imread(image_path)
    cv2.imwrite('raw.png', image)
    for bbox in bbox_list:
        y1, x1, y2, x2 = bbox
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.rectangle(image, (int(max_bbox[1]), int(max_bbox[0])), (int(max_bbox[3]), int(max_bbox[2])), (0, 0, 0), 5)
    cv2.imwrite('result.png', image)
    
if __name__ == '__main__':
    image_path = '/home/hcis-s17/multimodal_manipulation/patrick/CV/VLM-Guided-Desnowing-for-Enhanced-Perception-in-Autonomous-Driving/drive_dataset/snow/zurich_000074_000019_leftImg8bit.png'
    # image_path = '/home/hcis-s17/multimodal_manipulation/patrick/CV/VLM-Guided-Desnowing-for-Enhanced-Perception-in-Autonomous-Driving/drive_dataset/snow/zurich_000036_000019_leftImg8bit.png'
    show_example(image_path)
    