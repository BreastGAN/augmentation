# -*- coding: utf-8 -*-
# File: eval.py

import tqdm
import os
from collections import namedtuple, defaultdict
from contextlib import ExitStack
import numpy as np
import cv2
import json

from tensorpack.utils.utils import get_tqdm_kwargs

from models.rcnn.common import CustomResize, clip_boxes
from models.rcnn.config import config as cfg

DetectionResult = namedtuple('DetectionResult', ['box', 'score', 'class_id', 'mask'])
"""
box: 4 float
score: float
class_id: int, 1~NUM_CLASS
mask: None, or a binary image of the original image shape
"""


def fill_full_mask(box, mask, shape):
    """
    Args:
        box: 4 float
        mask: MxM floats
        shape: h,w
    """
    # int() is floor
    # box fpcoor=0.0 -> intcoor=0.0
    x0, y0 = list(map(int, box[:2] + 0.5))
    # box fpcoor=h -> intcoor=h-1, inclusive
    x1, y1 = list(map(int, box[2:] - 0.5))  # inclusive
    x1 = max(x0, x1)  # require at least 1x1
    y1 = max(y0, y1)

    w = x1 + 1 - x0
    h = y1 + 1 - y0

    # rounding errors could happen here, because masks were not originally computed for this shape.
    # but it's hard to do better, because the network does not know the "original" scale
    mask = (cv2.resize(mask, (w, h)) > 0.5).astype('uint8')
    ret = np.zeros(shape, dtype='uint8')
    ret[y0:y1 + 1, x0:x1 + 1] = mask
    return ret


def detect_one_image(img, model_func):
    """
    Run detection on one image, using the TF callable.
    This function should handle the preprocessing internally.

    Args:
        img: an image
        model_func: a callable from TF model,
            takes image and returns (boxes, probs, labels, [masks])

    Returns:
        [DetectionResult]
    """

    orig_shape = img.shape[:2]
    resizer = CustomResize(cfg.PREPROC.TEST_SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE)
    resized_img = resizer.augment(img)
    scale = np.sqrt(resized_img.shape[0] * 1.0 / img.shape[0] * resized_img.shape[1] / img.shape[1])
    boxes, probs, labels, *masks = model_func(resized_img)
    boxes = boxes / scale
    # boxes are already clipped inside the graph, but after the floating point scaling, this may not be true any more.
    boxes = clip_boxes(boxes, orig_shape)

    if masks:
        # has mask
        full_masks = [fill_full_mask(box, mask, orig_shape) for box, mask in zip(boxes, masks[0])]
        masks = full_masks
    else:
        # fill with none
        masks = [None] * len(boxes)

    results = [DetectionResult(*args) for args in zip(boxes, probs, labels, masks)]
    return results


def eval_coco_old(df, detect_func, tqdm_bar=None):
    """
    Args:
        df: a DataFlow which produces (image, image_id)
        detect_func: a callable, takes [image] and returns [DetectionResult]
        tqdm_bar: a tqdm object to be shared among multiple evaluation instances. If None,
            will create a new one.

    Returns:
        list of dict, to be dumped to COCO json format
    """
    # lazy import
    import pycocotools.mask as cocomask
    from coco import COCOMeta

    df.reset_state()
    all_results = []
    # tqdm is not quite thread-safe: https://github.com/tqdm/tqdm/issues/323
    with ExitStack() as stack:
        if tqdm_bar is None:
            tqdm_bar = stack.enter_context(tqdm.tqdm(total=df.size(), **get_tqdm_kwargs()))
        for img, img_id in df:
            results = detect_func(img)
            for r in results:
                box = r.box
                cat_id = COCOMeta.class_id_to_category_id[r.class_id]
                box[2] -= box[0]
                box[3] -= box[1]

                res = {
                        'image_id': img_id,
                        'category_id': cat_id,
                        'bbox': list(map(lambda x: round(float(x), 3), box)),
                        'score': round(float(r.score), 4),
                }

                # also append segmentation to results
                if r.mask is not None:
                    rle = cocomask.encode(np.array(r.mask[:, :, None], order='F'))[0]
                    rle['counts'] = rle['counts'].decode('ascii')
                    res['segmentation'] = rle
                all_results.append(res)
            tqdm_bar.update(1)
    return all_results


def eval_coco(df, detect_func, tqdm_bar=None):
    """
    Args:
        df: a DataFlow which produces (image, image_id)
        detect_func: a callable, takes [image] and returns [DetectionResult]
        tqdm_bar: a tqdm object to be shared among multiple evaluation instances. If None,
            will create a new one.

    Returns:
        list of dict, to be dumped to COCO json format
    """
    from models.rcnn.breasts import CLASS_NAMES
    df.reset_state()
    all_results = []
    # tqdm is not quite thread-safe: https://github.com/tqdm/tqdm/issues/323
    with ExitStack() as stack:
        if tqdm_bar is None:
            tqdm_bar = stack.enter_context(tqdm.tqdm(total=df.size(), **get_tqdm_kwargs()))
        for img, img_id in df:
            results = detect_func(img)
            for r in results:
                box = r.box
                cat_id = CLASS_NAMES[r.class_id]
                box[2] -= box[0]
                box[3] -= box[1]

                res = {
                        'image_id': img_id,
                        'category_id': cat_id,
                        'bbox': list(map(lambda x: round(float(x), 3), box)),
                        'score': round(float(r.score), 4),
                }

                # also append segmentation to results
                assert r.mask is None
                # if r.mask is not None:
                #     rle = cocomask.encode(
                #         np.array(r.mask[:, :, None], order='F'))[0]
                #     rle['counts'] = rle['counts'].decode('ascii')
                #     res['segmentation'] = rle
                all_results.append(res)
            tqdm_bar.update(1)
    return all_results


# https://github.com/pdollar/coco/blob/master/PythonAPI/pycocoEvalDemo.ipynb
def print_evaluation_scores_old(json_file):
    import pycocotools.mask as COCOeval
    from coco import COCO

    ret = {}
    assert cfg.DATA.BASEDIR and os.path.isdir(cfg.DATA.BASEDIR)
    annofile = os.path.join(cfg.DATA.BASEDIR, 'annotations', 'instances_{}.json'.format(cfg.DATA.VAL))
    coco = COCO(annofile)
    cocoDt = coco.loadRes(json_file)
    cocoEval = COCOeval(coco, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    fields = ['IoU=0.5:0.95', 'IoU=0.5', 'IoU=0.75', 'small', 'medium', 'large']
    for k in range(6):
        ret['mAP(bbox)/' + fields[k]] = cocoEval.stats[k]

    if cfg.MODE_MASK:
        cocoEval = COCOeval(coco, cocoDt, 'segm')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        for k in range(6):
            ret['mAP(segm)/' + fields[k]] = cocoEval.stats[k]
    return ret


# https://github.com/riblidezso/frcnn_cad/blob/master/demo.ipynb
def print_evaluation_scores(json_file,
                            include_fooling_stats=cfg.BREASTS.CALC_FOOLING_STATS,
                            confidence_score=cfg.BREASTS.MALIGNANT_CONFIDENCE):
    from models.rcnn.breasts import BreastDetection
    from sklearn import metrics
    with open(json_file, 'r') as f:
        results = json.load(f)
    breast_metadata = BreastDetection.load_many(cfg.DATA.VAL_PATTERN)
    breast_metadata = {m['id']: m for m in breast_metadata}

    def get_predictions(results, annotations, include_gen):
        preds = defaultdict(set)
        for result in results:
            preds[result['image_id']].add((result['category_id'], result['score']))
        output = {}
        scores = {}
        for id, findings in preds.items():
            if (not include_gen) and id.endswith("_gen"):
                continue
            malignant_scores = [score for klass, score in findings if klass == cfg.DATA.CLASS_NAMES[2]]
            if not malignant_scores:
                scores[id] = 0.0
            else:
                scores[id] = max(malignant_scores)

            if malignant_scores and max(malignant_scores) >= confidence_score:
                output[id] = 1
            else:
                output[id] = 0

        # Handle cases when no bbox is found.
        for key, value in annotations.items():
            if (not include_gen) and key.endswith("_gen"):
                continue
            if key not in output:
                output[key] = 0
                scores[key] = 0.0
        return output, scores

    def to_numpy(preds, annotations, dtype=np.int32):
        pred = []
        truth = []
        for id, lbl in preds.items():
            assert id in annotations
            pred.append(lbl)
            truth.append(annotations[id]['label'])
        return np.asarray(pred, dtype=dtype), np.asarray(truth, dtype=np.int32)

    preds, scores = get_predictions(results, breast_metadata, False)
    pred, truth = to_numpy(preds, breast_metadata)
    scores, truth_scores = to_numpy(scores, breast_metadata, dtype=np.float32)

    def get_fooling_stats(preds, annotations):
        total = wrong_clf_H = wrong_clf_C = fooled_H2C = fooled_C2H = 0
        inference_not_found = 0
        for id, pred_lbl in preds.items():
            assert id in annotations
            if id.endswith("_gen"):
                continue
            total += 1
            lbl = annotations[id]['label']
            if pred_lbl == lbl:
                # Correctly classified.
                # t = list([k for k in preds.keys() if k.startswith(id)])
                # print(t)
                # print(t[0])
                # print(t[1])
                if (id + "_gen") not in preds:
                    inference_not_found += 1
                    continue
                gen_pred_lbl = preds[id + "_gen"]
                if lbl == 1:
                    if gen_pred_lbl == 0:
                        fooled_C2H += 1
                else:
                    assert lbl == 0
                    if gen_pred_lbl == 1:
                        fooled_H2C += 1
            else:
                if lbl == 1:
                    wrong_clf_C += 1
                else:
                    wrong_clf_H += 1
        return {
                'fooling/total_num': total,
                'fooling/inference_not_found': inference_not_found,
                'fooling/wrong_clf_H': wrong_clf_H,
                'fooling/wrong_clf_C': wrong_clf_C,
                'fooling/correct_clf': total - wrong_clf_H - wrong_clf_C,
                'fooling/fooled': fooled_H2C + fooled_C2H,
                'fooling/fooled_H2C': fooled_H2C,
                'fooling/fooled_C2H': fooled_C2H,
        }

    ret = {
            'acc': metrics.accuracy_score(truth, pred),
            'roc_auc': metrics.roc_auc_score(truth_scores, scores),
            'f1': metrics.f1_score(truth, pred),
            'recall': metrics.recall_score(truth, pred),
            'precision': metrics.precision_score(truth, pred),
            # 'roc': metrics.roc_curve(truth_scores, scores),
    }

    if include_fooling_stats:
        preds, scores = get_predictions(results, breast_metadata, True)
        pred, truth = to_numpy(preds, breast_metadata)
        scores, truth_scores = to_numpy(scores, breast_metadata, dtype=np.float32)
        ret2 = {
                'all/acc': metrics.accuracy_score(truth, pred),
                'all/roc_auc': metrics.roc_auc_score(truth_scores, scores),
                'all/f1': metrics.f1_score(truth, pred),
                'all/recall': metrics.recall_score(truth, pred),
                'all/precision': metrics.precision_score(truth, pred),
                # 'all/roc': metrics.roc_curve(truth_scores, scores),
        }
        ret.update(get_fooling_stats(preds, breast_metadata))
        ret.update(ret2)

    return ret
