import tensorflow as tf
import numpy as np

def centerBoxes(boxes):
    ''' boxes [xmin, ymin, xmax, ymax]'''
    ''' output [xc yc w h]'''

    w = boxes[:,2] - boxes[:,0]
    h = boxes[:,3] - boxes[:,1]
    xc = boxes[:,0] + w / 2.
    yc = boxes[:,1] + h / 2.

    return np.stack((xc, yc, w, h), axis=-1)

def selectPrior(boxes, priors):
    '''
    Finds the bounding box prior that best fits boxes
    Args:
        boxes (np.ndarray): (Nx4) bouding boxes
        priors (np.ndarray): (Px2) bounding box priors

    Returns: (N,) indices into priors for each bounding box
    '''
    bw, bh = boxes[:,2], boxes[:,3] # (N,)
    pw, ph = priors[:,0], priors[:,1] #(P,)

    i_w = np.minimum(bw[:,np.newaxis], pw) # (N, P)
    i_h = np.minimum(bh[:,np.newaxis], ph)
    intersection = i_w * i_h

    s_bb = bw * bh
    s_pr = pw * ph

    union = s_bb[:,np.newaxis] + s_pr - intersection

    iou = intersection / union
    print(iou)

    return np.argmax(iou, axis=-1)


def processBoundingBoxes(boxes, priors, output_shape):
    #bounding boxes w.r.t grid cells
    cb = centerBoxes(boxes) / 32
    target_priors = selectPrior(cb, priors)

    y_true = np.zeros_like(output_shape) # s x s x b x 5

    cell_x = np.floor(cb[:,0]).astype(np.uint32)
    cell_y = np.floor(cb[:,1]).astype(np.uint32)

    values = np.concatenate((cb, np.ones((len(cb),1))), axis=1)

    y_true[cell_y, cell_x, target_priors] = values

    return y_true


def yolo_loss(y_true, y_pred, priors):



    """

    Args:
        y_true: m x s x s x b x 5
        y_pred: m x s x s x b x 5
        priors: 5 x 2

    Returns: scalar loss for the batch

    """
    pass
