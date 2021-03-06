import numpy as np
import matplotlib.pyplot as plt
import json
import copy
from task2_tools import read_predicted_boxes, read_ground_truth_boxes

def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE

    union = 0
    overlap = 0

    #If boxes don't touch we return 0
    if prediction_box[0] >= gt_box[2] or prediction_box[2] <= gt_box[0]: return 0
    if prediction_box[1] >= gt_box[3] or prediction_box[3] <= gt_box[1]: return 0

    #Calculating intersection
    dx = min(prediction_box[2], gt_box[2]) - max(prediction_box[0], gt_box[0])
    dy = min(prediction_box[3], gt_box[3]) - max(prediction_box[1], gt_box[1])
    intersection = dx*dy

    #Calculating union
    union = (prediction_box[2] - prediction_box[0])*(prediction_box[3] - prediction_box[1])\
            + (gt_box[2] - gt_box[0])*(gt_box[3] - gt_box[1]) - intersection

    return intersection/union

def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """

    if num_tp + num_fp == 0: return 1
    return num_tp/(num_tp+num_fp)


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """

    if num_tp + num_fn == 0: return 0
    return num_tp/(num_tp+num_fn)


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # Find all possible matches with a IoU >= iou threshold

    # Sort all matches on IoU in descending order

    # Find all matches with the highest IoU threshold

    n_prediction_boxes = (np.shape(prediction_boxes))[0]
    n_gt_boxes = (np.shape(gt_boxes))[0]

    matches = []

    for i in range(n_gt_boxes):

        temp_matches = []

        for q in range(n_prediction_boxes):

            iou = calculate_iou(prediction_boxes[q], gt_boxes[i])

            if iou >= iou_threshold:
                temp_matches.append((iou, gt_boxes[i], prediction_boxes[q]))

        temp_matches.sort(key=lambda temp_matches: temp_matches[0])

        if len(temp_matches) > 0:
            matches.append(temp_matches[0])

    matches.sort(key=lambda matches: matches[0])
    gt_boxes = np.zeros((len(matches), 4))
    prediction_boxes = np.zeros((len(matches), 4))

    for i in range(len(matches)):
        gt_boxes[i,:] = (matches[i])[1]
        prediction_boxes[i, :] = (matches[i])[2]

    return prediction_boxes, gt_boxes


def calculate_individual_image_result(
        prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, "false_neg": int}
    """

    # Find the bounding box matches with the highes IoU threshold
    
    # Compute true positives, false positives, false negatives

    n_prediction_boxes = np.shape(prediction_boxes)[0]
    n_gt_boxes = np.shape(gt_boxes)[0]

    best_prediction_boxes, best_gt_boxes = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)

    tp = np.shape(best_prediction_boxes)[0]
    fp = n_prediction_boxes - tp
    fn = n_gt_boxes - tp

    return {"true_pos": tp, "false_pos": fp, "false_neg": fn}

def calculate_precision_recall_all_images(
        all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images.
       
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    # Find total true positives, false positives and false negatives
    # over all images

    # Compute precision, recall

    n_images = len(all_prediction_boxes)
    assert n_images == len(all_gt_boxes)

    tp_sum = 0
    fp_sum = 0
    fn_sum = 0

    for i in range(n_images):
        res = calculate_individual_image_result(all_prediction_boxes[i],all_gt_boxes[i], iou_threshold)

        tp_sum += res["true_pos"]
        fp_sum += res["false_pos"]
        fn_sum += res["false_neg"]

    return (calculate_precision(tp_sum,fp_sum,fn_sum), calculate_recall(tp_sum,fp_sum,fn_sum))


def get_precision_recall_curve(all_prediction_boxes, all_gt_boxes,
                               confidence_scores, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the precision-recall curve over all images. Use the given
       confidence thresholds to find the precision-recall curve.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        tuple: (precision, recall). Both np.array of floats.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    # DO NOT CHANGE. If you change this, the tests will not pass when we run the final
    # evaluation
    confidence_thresholds = np.linspace(0, 1, 500)
    # YOUR CODE HERE

    n_images = len(all_prediction_boxes)
    assert n_images == len(all_gt_boxes)

    result_precision = np.zeros(np.shape(confidence_thresholds))
    result_recall = np.zeros(np.shape(confidence_thresholds))

    for q in range(np.shape(confidence_thresholds)[0]):

        threshold = confidence_thresholds[q]
        work_copy_all_prediction_boxes = copy.deepcopy(all_prediction_boxes)

        for i in range(n_images):
            filter_mask = np.greater(confidence_scores[i], threshold)
            work_copy_all_prediction_boxes[i] = (work_copy_all_prediction_boxes[i])[filter_mask]


        precision, recall = calculate_precision_recall_all_images(work_copy_all_prediction_boxes, all_gt_boxes, iou_threshold)

        result_precision[q] = precision
        result_recall[q] = recall

    return result_precision, result_recall




def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    # No need to edit this code.
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    # DO NOT CHANGE. If you change this, the tests will not pass when we run the final
    # evaluation
    recall_levels = np.linspace(0, 1.0, 11)
    # YOUR CODE HERE

    print(recalls)

    p = recalls.argsort()
    recalls = recalls[p]
    precisions = precisions[p]
    precisions_interpolated = np.interp(recall_levels, recalls, precisions)

    print(recalls)

    for i in range(np.shape(precisions_interpolated)[0]):
        precisions_interpolated[i] = np.max(precisions_interpolated[i:])

    return np.mean(precisions_interpolated)



def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)
    iou_threshold = 0.5
    precisions, recalls = get_precision_recall_curve(all_prediction_boxes,
                                                     all_gt_boxes,
                                                     confidence_scores,
                                                     iou_threshold)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions,
                                                              recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
