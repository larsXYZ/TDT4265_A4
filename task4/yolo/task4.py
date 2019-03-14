import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from drawing_utils import read_classes, draw_boxes, scale_boxes

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    """ Filters YOLO boxes by thresholding on object and class confidence.

    Arguments:
        box_confidence -- np.array of shape (19, 19, 5, 1)
        boxes -- np.array of shape (19, 19, 5, 4)
        box_class_probs -- np.array of shape (19, 19, 5, 80)
        threshold -- real value, if [ highest class probability score < threshold],
            then get rid of the corresponding box

    Returns:
        scores -- np.array of shape (None,), containing the class probability score for selected boxes
        boxes -- np.array of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
        classes -- np.array of shape (None,), containing the index of the class detected by the selected boxes

    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold.
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """
    #Dimension of arrays (D1,D2,D3,) i.e 19x19x5
    D1 = len(boxes)
    D2 = len(boxes[0])
    D3 = len(boxes[0][0])

    # Step 1: Compute box scores
    box_scores = box_confidence*box_class_probs

    # Step 2: Find the box_classes thanks to the max box_scores, keep track of the corresponding score
    max_box_score = np.zeros((D1,D2,D3))
    max_box_score_index = np.zeros((D1,D2,D3))
    for i in range(D1):
        for j in range(D2):
            for k in range(D3):
                max_score = box_scores[i][j][k].max()
                index = np.argmax(box_scores[i][j][k])
                max_box_score[i][j][k] = max_score
                max_box_score_index[i][j][k] = index

    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
    mask = max_box_score >= threshold

    # Step 4: Apply the mask to scores, boxes and classes
    scores = max_box_score[mask]
    boxes = boxes[mask]
    classes = max_box_score_index[mask]
    return scores, boxes, classes

def iou(box1, box2):

    """Implement the intersection over union (IoU) between box1 and box2

    Arguments:
        box1 -- first box, list object with coordinates (x1, y1, x2, y2)
        box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """
    # YOUR CODE HERE

    union = 0
    overlap = 0

    #If boxes don't touch we return 0
    if box1[0] >= box2[2] or box1[2] <= box2[0]: return 0
    if box1[1] >= box2[3] or box1[3] <= box2[1]: return 0

    #Calculating intersection
    dx = min(box1[2], box2[2]) - max(box1[0], box2[0])
    dy = min(box1[3], box2[3]) - max(box1[1], box2[1])
    intersection = dx*dy

    #Calculating union
    union = (box1[2] - box1[0])*(box1[3] - box1[1])\
            + (box2[2] - box2[0])*(box2[3] - box2[1]) - intersection

    return intersection/union

def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):

    """
    Applies Non-max suppression (NMS) to set of boxes

    Arguments:
        scores -- np.array of shape (None,), output of yolo_filter_boxes()
        boxes -- np.array of shape (None, 4), output of yolo_filter_boxes()
            that have been scaled to the image size (see later)
        classes -- np.array of shape (None,), output of yolo_filter_boxes()
        max_boxes -- integer, maximum number of predicted boxes you'd like
        iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

    Returns:
        scores -- tensor of shape (, None), predicted score for each box
        boxes -- tensor of shape (4, None), predicted box coordinates
        classes -- tensor of shape (, None), predicted class for each box

    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes.
    Note also that this function will transpose the shapes of scores, boxes, classes.
    This is made for convenience.
    """

    nms_indices = []

    # Use iou() to get the list of indices corresponding to boxes you keep
    temp_scores = np.copy(scores)
    highest_score_index = np.argmax(temp_scores)
    while temp_scores[highest_score_index] != -1e9:
        for i in range(len(temp_scores)):
            if highest_score_index == i:
                continue
            if iou(boxes[highest_score_index],boxes[i]) > iou_threshold:
                temp_scores[i] = -1e9
        temp_scores[highest_score_index] = -1e9
        nms_indices.append(highest_score_index)
        highest_score_index = np.argmax(temp_scores)

    # Use index arrays to select only nms_indices from scores, boxes and classes
    scores = scores[np.array(nms_indices)]
    boxes = boxes[np.array(nms_indices)]
    classes = classes[np.array(nms_indices)]

    return scores, boxes, classes

def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.

    Arguments:
        yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 np.array:
                        box_confidence: tensor of shape (None, 19, 19, 5, 1)
                        boxes: tensor of shape (None, 19, 19, 5, 4)
                        box_class_probs: tensor of shape (None, 19, 19, 5, 80)
        image_shape -- np.array of shape (2,) containing the input shape, in this notebook we use
            (608., 608.) (has to be float32 dtype)
        max_boxes -- integer, maximum number of predicted boxes you'd like
        score_threshold -- real value, if [ highest class probability score < threshold],
            then get rid of the corresponding box
        iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

    Returns:
        scores -- np.array of shape (None, ), predicted score for each box
        boxes -- np.array of shape (None, 4), predicted box coordinates
        classes -- np.array of shape (None,), predicted class for each box
    """

    ### START CODE HERE ###

    # Retrieve outputs of the YOLO model (≈1 line)
    box_confidence = np.copy(yolo_outputs[0])
    box_confidence = np.squeeze(box_confidence, axis=0)

    boxes = np.copy(yolo_outputs[1])
    boxes = np.squeeze(boxes, axis=0)

    box_class_probs = np.copy(yolo_outputs[2])
    box_class_probs = np.squeeze(box_class_probs, axis=0)

    # Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold (≈1 line)
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)
    # Scale boxes back to original image shape.
    boxes = scale_boxes(boxes, image_shape)
    # Use one of the functions you've implemented to perform Non-max suppression with a threshold of iou_threshold (≈1 line)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)
    ### END CODE HERE ###

    return scores, boxes, classes


# DO NOT CHANGE
image = Image.open("test.jpg")
box_confidence = np.load("box_confidence.npy")
boxes = np.load("boxes.npy")
box_class_probs = np.load("box_class_probs.npy")
yolo_outputs = (box_confidence, boxes, box_class_probs)

# DO NOT CHANGE
image_shape = (720., 1280.)

#DO NOT EDIT THIS CODE
out_scores, out_boxes, out_classes = yolo_eval(yolo_outputs, image_shape)
#Assures that out_classes is of type int64 (required for draw_boxes())
out_classes=out_classes.astype(np.int64)

#DO NOT EDIT THIS CODE
# Print predictions info
print('Found {} boxes'.format(len(out_boxes)))
# Draw bounding boxes on the image
draw_boxes(image, out_scores, out_boxes, out_classes)
# Display the results in the notebook
plt.imshow(image)
plt.show()
