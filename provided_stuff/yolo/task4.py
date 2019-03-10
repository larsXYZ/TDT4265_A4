import numpy as np
#from PIL import Image
#from matplotlib.pyplot import imshow
#from drawing_utils import read_classes, draw_boxes, scale_boxes
import operator



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

#DO NOT EDIT THIS CODE
np.random.seed(0)
box_confidence = np.random.normal(size=(19, 19, 5, 1), loc=1, scale=4)
boxes = np.random.normal(size=(19, 19, 5, 4), loc=1, scale=4)
box_class_probs = np.random.normal(size=(19, 19, 5, 80), loc=1, scale=4)
scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = 0.5)
print("scores[2] = " + str(scores[2]))
print("boxes[2] = " + str(boxes[2]))
print("classes[2] = " + str(classes[2]))
print("scores.shape = " + str(scores.shape))
print("boxes.shape = " + str(boxes.shape))
print("classes.shape = " + str(classes.shape))
