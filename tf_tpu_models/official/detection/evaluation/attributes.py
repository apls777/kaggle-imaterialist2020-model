from collections import defaultdict
import numpy as np
from pycocotools import mask as mask_utils
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
import copy


def evaluate_attributes(annotations_gt: list, annotations_pr: list):
    # get a number of attributes
    num_attributes = len(annotations_pr[0]['attribute_probabilities'])

    # extract category IDs
    category_ids = sorted(set(int(ann['category_id']) for ann in annotations_gt))

    # create multi-hot vectors
    annotations_gt = copy.deepcopy(annotations_gt)
    for ann in annotations_gt:
        attributes_multi_hot = np.zeros(num_attributes, dtype=np.float32)
        attributes_multi_hot[ann['attribute_ids']] = 1
        ann['attributes_multi_hot'] = attributes_multi_hot

    # group annotations and predictions by image ID and category ID
    annotations_gt = _group_annotations(annotations_gt)
    annotations_pr = _group_annotations(annotations_pr)

    # get image IDs
    image_ids = list(annotations_gt.keys())

    # collect all attribute predictions and labels
    attributes_pr = []
    attributes_gt = []
    cat_attributes_pred = defaultdict(list)
    cat_attributes_gt = defaultdict(list)
    for category_id in category_ids:
        for image_id in image_ids:
            boxes_pr = annotations_pr[image_id].get(category_id)
            boxes_gt = annotations_gt[image_id].get(category_id)
            if not boxes_pr or not boxes_gt:
                continue

            pred_bboxes = [box['bbox'] for box in boxes_pr]
            gt_bboxes = [box['bbox'] for box in boxes_gt]
            is_crowd = [0] * len(gt_bboxes)
            ious = mask_utils.iou(pred_bboxes, gt_bboxes, is_crowd)

            for i, box_pr in enumerate(boxes_pr):
                for j, box_gt in enumerate(boxes_gt):
                    if ious[i, j] > 0.5:
                        attributes_pr.append(box_pr['attribute_probabilities'])
                        attributes_gt.append(box_gt['attributes_multi_hot'])
                        cat_attributes_pred[category_id].append(box_pr['attribute_probabilities'])
                        cat_attributes_gt[category_id].append(box_gt['attributes_multi_hot'])
                        break

    attributes_pr = np.array(attributes_pr)
    attributes_gt = np.array(attributes_gt)
    cat_attributes_pred = {category_id: np.array(arr) for category_id, arr in cat_attributes_pred.items()}
    cat_attributes_gt = {category_id: np.array(arr) for category_id, arr in cat_attributes_gt.items()}

    # compute attribute thresholds and metrics
    attr_precisions = np.zeros(num_attributes, dtype=np.float32)
    attr_recalls = np.zeros(num_attributes, dtype=np.float32)
    attr_f1_scores = np.zeros(num_attributes, dtype=np.float32)
    attr_thresholds = np.zeros(num_attributes, dtype=np.float32)
    attr_num_predictions = np.zeros(num_attributes, dtype=np.int32)
    attr_num_examples = np.zeros(num_attributes, dtype=np.int32)

    for attr_id in range(num_attributes):
        best_precision, best_recall, best_f1_score, best_threshold, best_num_predictions, num_examples = \
            _compute_best_f1_score(attributes_gt[:, attr_id], attributes_pr[:, attr_id])

        attr_precisions[attr_id] = best_precision
        attr_recalls[attr_id] = best_recall
        attr_f1_scores[attr_id] = best_f1_score
        attr_thresholds[attr_id] = best_threshold
        attr_num_predictions[attr_id] = best_num_predictions
        attr_num_examples[attr_id] = num_examples

    # compute attribute thresholds and metrics per category
    attr_precisions_per_cat = defaultdict(lambda: np.zeros(num_attributes, dtype=np.float32))
    attr_recalls_per_cat = defaultdict(lambda: np.zeros(num_attributes, dtype=np.float32))
    attr_f1_scores_per_cat = defaultdict(lambda: np.zeros(num_attributes, dtype=np.float32))
    attr_thresholds_per_cat = defaultdict(lambda: np.zeros(num_attributes, dtype=np.float32))
    attr_num_predictions_per_cat = defaultdict(lambda: np.zeros(num_attributes, dtype=np.int32))
    attr_num_examples_per_cat = defaultdict(lambda: np.zeros(num_attributes, dtype=np.int32))

    for category_id in category_ids:
        for attr_id in range(num_attributes):
            if category_id in cat_attributes_pred:
                best_precision, best_recall, best_f1_score, best_threshold, best_num_predictions, num_examples = \
                    _compute_best_f1_score(cat_attributes_gt[category_id][:, attr_id],
                                           cat_attributes_pred[category_id][:, attr_id])

                attr_precisions_per_cat[category_id][attr_id] = best_precision
                attr_recalls_per_cat[category_id][attr_id] = best_recall
                attr_f1_scores_per_cat[category_id][attr_id] = best_f1_score
                attr_thresholds_per_cat[category_id][attr_id] = best_threshold
                attr_num_predictions_per_cat[category_id][attr_id] = best_num_predictions
                attr_num_examples_per_cat[category_id][attr_id] = num_examples

    # evaluate attribute predictions for boxes
    all_cat_precisions = {}
    all_cat_recalls = {}
    all_cat_f1_scores = {}

    all_cat_precisions_cat = {}
    all_cat_recalls_cat = {}
    all_cat_f1_scores_cat = {}

    for category_id in category_ids:
        cat_precisions = []
        cat_recalls = []
        cat_f1_scores = []

        cat_precisions_cat = []
        cat_recalls_cat = []
        cat_f1_scores_cat = []

        for image_id in image_ids:
            boxes_pr = annotations_pr[image_id][category_id]
            boxes_gt = annotations_gt[image_id][category_id]

            pred_bboxes = [box['bbox'] for box in boxes_pr]
            gt_bboxes = [box['bbox'] for box in boxes_gt]
            is_crowd = [0] * len(gt_bboxes)
            ious = mask_utils.iou(pred_bboxes, gt_bboxes, is_crowd)

            img_precisions = []
            img_recalls = []
            img_f1_scores = []

            img_precisions_cat = []
            img_recalls_cat = []
            img_f1_scores_cat = []

            for i, box_pr in enumerate(boxes_pr):
                for j, box_gt in enumerate(boxes_gt):
                    if ious[i, j] > 0.5:
                        precision, recall, f1_score = \
                            _compute_best_attribute_metric(box_gt['attributes_multi_hot'],
                                                           box_pr['attribute_probabilities'],
                                                           attr_thresholds)

                        precision_cat, recall_cat, f1_score_cat = \
                            _compute_best_attribute_metric(box_gt['attributes_multi_hot'],
                                                           box_pr['attribute_probabilities'],
                                                           attr_thresholds_per_cat[category_id])

                        img_precisions.append(precision)
                        img_recalls.append(recall)
                        img_f1_scores.append(f1_score)

                        img_precisions_cat.append(precision_cat)
                        img_recalls_cat.append(recall_cat)
                        img_f1_scores_cat.append(f1_score_cat)

                        break

            if img_precisions:
                cat_precisions.append(sum(img_precisions) / len(img_precisions))
                cat_recalls.append(sum(img_recalls) / len(img_recalls))
                cat_f1_scores.append(sum(img_f1_scores) / len(img_f1_scores))

                cat_precisions_cat.append(sum(img_precisions_cat) / len(img_precisions_cat))
                cat_recalls_cat.append(sum(img_recalls_cat) / len(img_recalls_cat))
                cat_f1_scores_cat.append(sum(img_f1_scores_cat) / len(img_f1_scores_cat))

        # average attribute precision, recall and F1-score across all images
        all_cat_precisions[category_id] = (sum(cat_precisions) / len(cat_precisions)) if cat_precisions else 0.
        all_cat_recalls[category_id] = (sum(cat_recalls) / len(cat_recalls)) if cat_recalls else 0.
        all_cat_f1_scores[category_id] = (sum(cat_f1_scores) / len(cat_f1_scores)) if cat_f1_scores else 0.

        all_cat_precisions_cat[category_id] = (sum(cat_precisions_cat) / len(cat_precisions_cat)) \
            if cat_precisions_cat else 0.
        all_cat_recalls_cat[category_id] = (sum(cat_recalls_cat) / len(cat_recalls_cat)) \
            if cat_recalls_cat else 0.
        all_cat_f1_scores_cat[category_id] = (sum(cat_f1_scores_cat) / len(cat_f1_scores_cat)) \
            if cat_f1_scores_cat else 0.

    eval_results = {}

    for attr_id in range(num_attributes):
        if attr_num_examples[attr_id]:
            eval_results['attribute_precision/attr_%03d' % attr_id] = attr_precisions[attr_id]
            eval_results['attribute_recall/attr_%03d' % attr_id] = attr_recalls[attr_id]
            eval_results['attribute_f1_score/attr_%03d' % attr_id] = attr_f1_scores[attr_id]
            eval_results['attribute_threshold/attr_%03d' % attr_id] = attr_thresholds[attr_id]
            eval_results['attribute_num_predictions/attr_%03d' % attr_id] = attr_num_predictions[attr_id]
            eval_results['attribute_num_examples/attr_%03d' % attr_id] = attr_num_examples[attr_id]

    for category_id in category_ids:
        for attr_id in range(num_attributes):
            if attr_num_examples_per_cat[category_id][attr_id]:
                eval_results['attribute_precision_per_category/cat_%02d/attr_%03d' % (category_id, attr_id)] = \
                    attr_precisions_per_cat[category_id][attr_id]
                eval_results['attribute_recall_per_category/cat_%02d/attr_%03d' % (category_id, attr_id)] = \
                    attr_recalls_per_cat[category_id][attr_id]
                eval_results['attribute_f1_score_per_category/cat_%02d/attr_%03d' % (category_id, attr_id)] = \
                    attr_f1_scores_per_cat[category_id][attr_id]
                eval_results['attribute_threshold_per_category/cat_%02d/attr_%03d' % (category_id, attr_id)] = \
                    attr_thresholds_per_cat[category_id][attr_id]
                eval_results['attribute_num_predictions_per_category/cat_%02d/attr_%03d' % (category_id, attr_id)] = \
                    attr_num_predictions_per_cat[category_id][attr_id]
                eval_results['attribute_num_examples_per_category/cat_%02d/attr_%03d' % (category_id, attr_id)] = \
                    attr_num_examples_per_cat[category_id][attr_id]

    for category_id in category_ids:
        eval_results['segment_attributes_precision/cat_%02d' % category_id] = all_cat_precisions[category_id]
        eval_results['segment_attributes_recall/cat_%02d' % category_id] = all_cat_recalls[category_id]
        eval_results['segment_attributes_f1_score/cat_%02d' % category_id] = all_cat_f1_scores[category_id]

        eval_results['segment_attributes_precision_per_category/cat_%02d' % category_id] = all_cat_precisions_cat[category_id]
        eval_results['segment_attributes_recall_per_category/cat_%02d' % category_id] = all_cat_recalls_cat[category_id]
        eval_results['segment_attributes_f1_score_per_category/cat_%02d' % category_id] = all_cat_f1_scores_cat[category_id]

    attr_non_zero_indices = np.argwhere(attr_f1_scores > 0).flatten()

    cat_mean_precisions = []
    cat_mean_recalls = []
    cat_mean_f1_scores = []
    for category_id in category_ids:
        cat_attr_non_zero_indices = np.argwhere(attr_f1_scores_per_cat[category_id] > 0).flatten()
        if len(cat_attr_non_zero_indices):
            cat_mean_precision = attr_precisions_per_cat[category_id][cat_attr_non_zero_indices].mean()
            cat_mean_recall = attr_recalls_per_cat[category_id][cat_attr_non_zero_indices].mean()
            cat_mean_f1_score = attr_f1_scores_per_cat[category_id][cat_attr_non_zero_indices].mean()

            cat_mean_precisions.append(cat_mean_precision)
            cat_mean_recalls.append(cat_mean_recall)
            cat_mean_f1_scores.append(cat_mean_f1_score)

    eval_results.update({
        'performance/segment_attributes_precision': np.array(list(all_cat_precisions.values()), dtype=np.float32).mean(),
        'performance/segment_attributes_recall': np.array(list(all_cat_recalls.values()), dtype=np.float32).mean(),
        'performance/segment_attributes_f1_score': np.array(list(all_cat_f1_scores.values()), dtype=np.float32).mean(),
        'performance/segment_attributes_precision_per_category': np.array(list(all_cat_precisions_cat.values()), dtype=np.float32).mean(),
        'performance/segment_attributes_recall_per_category': np.array(list(all_cat_recalls_cat.values()), dtype=np.float32).mean(),
        'performance/segment_attributes_f1_score_per_category': np.array(list(all_cat_f1_scores_cat.values()), dtype=np.float32).mean(),
        'performance/attribute_precision': attr_precisions[attr_non_zero_indices].mean(),
        'performance/attribute_recall': attr_recalls[attr_non_zero_indices].mean(),
        'performance/attribute_f1_score': attr_f1_scores[attr_non_zero_indices].mean(),
        'performance/attribute_precision_per_category': np.array(cat_mean_precisions).mean(),
        'performance/attribute_recall_per_category': np.array(cat_mean_recalls).mean(),
        'performance/attribute_f1_score_per_category': np.array(cat_mean_f1_scores).mean(),
    })

    return eval_results


def _group_annotations(annotations: list):
    grouped_annotations = defaultdict(lambda: defaultdict(list))
    for ann in annotations:
        grouped_annotations[int(ann['image_id'])][int(ann['category_id'])].append(ann)

    return grouped_annotations


def _compute_f1_score(precision: float, recall: float):
    if precision + recall == 0.:
        return 0.

    return (2 * precision * recall) / (precision + recall)


def _compute_best_f1_score(y_true, probas_pred):
    best_f1_score = 0
    best_precision = 0
    best_recall = 0
    best_threshold = 1.1
    best_num_predictions = 0
    num_examples = 0

    if len(y_true):
        num_examples = int(y_true.sum())
        precisions, recalls, thresholds = precision_recall_curve(y_true, probas_pred)

        for precision, recall, threshold in zip(precisions[:-1], recalls[:-1], thresholds):
            f1_score_val = _compute_f1_score(precision, recall)
            if f1_score_val > best_f1_score:
                best_precision = precision
                best_recall = recall
                best_f1_score = f1_score_val
                best_threshold = threshold
                best_num_predictions = probas_pred[probas_pred >= threshold].sum()

    return best_precision, best_recall, best_f1_score, best_threshold, best_num_predictions, num_examples


def _compute_best_attribute_metric(attr_gt, attr_pred, attr_thresholds):
    attr_pred = (attr_pred >= attr_thresholds).astype(np.int32)

    has_pred = (attr_pred > 0).any()
    has_gt = (attr_gt > 0).any()

    if (has_pred and not has_gt) or (not has_pred and has_gt):
        precision, recall, f1_score = 0., 0., 0.
    elif not has_gt and not has_pred:
        precision, recall, f1_score = 1., 1., 1.
    else:
        precision, recall, f1_score, _ = precision_recall_fscore_support(attr_gt, attr_pred, average='binary')

    return precision, recall, f1_score
