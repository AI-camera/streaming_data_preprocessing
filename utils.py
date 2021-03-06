import numpy as np
import os
import cv2
import math


# Maximum number of boxes. Only the top scoring ones will be considered.
MAX_BOXES = 30
def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.shape[0:2][::-1]
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    image = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_CUBIC)
    new_image = np.zeros((size[1], size[0], 3), np.uint8)
    new_image.fill(128)
    dx = (w-nw)//2
    dy = (h-nh)//2
    new_image[dy:dy+nh, dx:dx+nw,:] = image
    return new_image

def featuresToBoxes(outputs, anchors, n_classes, net_input_shape, 
        img_orig_shape, threshold):
    grid_shape = outputs.shape[1:3]
    n_anchors = len(anchors)

    # Numpy screwaround to get the boxes in reasonable amount of time
    grid_y = np.tile(np.arange(grid_shape[0]).reshape(-1, 1), grid_shape[0]).reshape(1, grid_shape[0], grid_shape[0], 1).astype(np.float32)
    grid_x = grid_y.copy().T.reshape(1, grid_shape[0], grid_shape[1], 1).astype(np.float32)
    outputs = outputs.reshape(1, grid_shape[0], grid_shape[1], n_anchors, -1)
    _anchors = anchors.reshape(1, 1, 3, 2).astype(np.float32)

    # Get box parameters from network output and apply transformations
    bx = (sigmoid(outputs[..., 0]) + grid_x) / grid_shape[0] 
    by = (sigmoid(outputs[..., 1]) + grid_y) / grid_shape[1]
    # Should these be inverted?
    bw = np.multiply(_anchors[..., 0] / net_input_shape[1], np.exp(outputs[..., 2]))
    bh = np.multiply(_anchors[..., 1] / net_input_shape[2], np.exp(outputs[..., 3]))
    
    # Get the scores 
    scores = sigmoid(np.expand_dims(outputs[..., 4], -1)) * \
             sigmoid(outputs[..., 5:])
    scores = scores.reshape(-1, n_classes)

    # Reshape boxes and scale back to original image size
    ratio = net_input_shape[2] / img_orig_shape[1]
    letterboxed_height = ratio * img_orig_shape[0] 
    scale = net_input_shape[1] / letterboxed_height
    offset = (net_input_shape[1] - letterboxed_height) / 2 / net_input_shape[1]
    bx = bx.flatten()
    by = (by.flatten() - offset) * scale
    bw = bw.flatten()
    bh = bh.flatten() * scale
    half_bw = bw / 2.
    half_bh = bh / 2.

    tl_x = np.multiply(bx - half_bw, img_orig_shape[1])
    tl_y = np.multiply(by - half_bh, img_orig_shape[0]) 
    br_x = np.multiply(bx + half_bw, img_orig_shape[1])
    br_y = np.multiply(by + half_bh, img_orig_shape[0])

    # Get indices of boxes with score higher than threshold
    indices = np.argwhere(scores >= threshold)
    selected_boxes = []
    selected_scores = []
    for i in indices:
        i = tuple(i)
        selected_boxes.append( ((tl_x[i[0]], tl_y[i[0]]), (br_x[i[0]], br_y[i[0]])) )
        selected_scores.append(scores[i])

    selected_boxes = np.array(selected_boxes)
    selected_scores = np.array(selected_scores)
    selected_classes = indices[:, 1]

    return selected_boxes, selected_scores, selected_classes
    
def get_anchors(path):
    anchors_path = os.path.expanduser(path)
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def get_classes(path):
    classes_path = os.path.expanduser(path)
    with open(classes_path) as f:
        classes = [line.strip('\n') for line in f.readlines()]
    return classes

def nms_boxes(boxes, scores, classes):
    # present_classes = np.unique(classes)
    if type(boxes) is list:
        boxes = np.asarray(boxes)
        scores = np.asarray(scores)
        classes = np.asarray(classes)
    assert(boxes.shape[0] == scores.shape[0])
    assert(boxes.shape[0] == classes.shape[0])

    # Sort based on score
    indices = np.arange(len(scores))
    scores, sorted_is = (list(l) for l in zip(*sorted(zip(scores, indices), reverse=True)))
    boxes = list(boxes[sorted_is])
    classes = list(classes[sorted_is])

    # Run nms for each class
    i = 0
    while True:
        if len(boxes) == 1 or i >= len(boxes) or i >= MAX_BOXES:
            break

        # Get box with highest score
        best_box = boxes[i]
        best_cl = classes[i]

        # Iterate over other boxes
        to_remove = []
        for j in range(i+1, len(boxes)):
            other_cl = classes[j]
            #if other_cl != best_cl:
            #    continue

            other_box = boxes[j]
            box_iou = iou(best_box, other_box)
            if box_iou > 0.25:
                to_remove.append(j)

        i += 1
        if len(to_remove) == 0:
            continue
        else:
            # Remove boxes
            for r in to_remove[::-1]:
                del boxes[r]
                del scores[r]
                del classes[r]

    return boxes[:MAX_BOXES], scores[:MAX_BOXES], classes[:MAX_BOXES]

def iou(box1, box2):
    '''
    Find iou of 2 boxes. 
    Box format must be ((x1,y1),(x2,y2)) or (x1,y1,x2,y2)
    '''
    if(len(box1) == 4):
        x1_1,y1_1, x2_1, y2_1 = box1
    elif (len(box1) == 2):
        (x1_1,y1_1), (x2_1, y2_1) = box1
    else:
        print("Box1 format must be ((x1,y1),(x2,y2)) or (x1,y1,x2,y2) ")
    
    if(len(box2) == 4):
        x1_2,y1_2, x2_2, y2_2 = box2
    elif (len(box2) == 2):
        (x1_2,y1_2), (x2_2, y2_2) = box2
    else:
        print("Box2 format must be ((x1,y1),(x2,y2)) or (x1,y1,x2,y2) ")
        
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_area = (xi2 - xi1)*(yi2 - yi1)
    # Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (y2_1 - y1_1)*(x2_1- x1_1)
    box2_area = (y2_2 - y1_2)*(x2_2- x1_2)
    union_area = (box1_area + box2_area) - inter_area
    # compute the IoU
    IoU = inter_area / union_area

    return IoU

def imcrop(img,x1,y1,x2,y2):
    '''
        Crop the image using bounding box
        x1: x top-left of bounding box
        y1: y top-left of bounding box
        x2: x bot-right of bounding box
        y2: y bot-right of bounding box
    '''
    return img[y1:y2,x1:x2]

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def intersect(A,B,C,D):
    '''Return true if line segments AB and CD intersect'''
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def denormalize_coordinate(frame,coordinate):
    '''
    * Denormalize a coordinate by multiply it with the frame's shape
    * Returns integer
    '''
    coordinate_x, coordinate_y = coordinate
    coordinate_x = int(coordinate_x*len(frame[0]))
    coordinate_y = int(coordinate_y*len(frame))
    return coordinate_x, coordinate_y

def denormalize_coordinates(frame,coordinates):
    '''
    * Denormalize multple coordinates by multiply them with the frame's shape
    * Returns integer
    '''
    result = []
    for coordinate in coordinates:
        result.append(denormalize_coordinate(frame, coordinate))
    return result

def block_distance(point1, point2):
    x1,y1 = point1
    x2,y2 = point2
    result = x1-y1+x2-y2
    if result <0:
        result = -result
    return result

def point_line_distance(point,linePoint1, linePoint2):
    '''Return distance between line and point'''
    p1 = np.array(linePoint1)
    p2 = np.array(linePoint2)
    p3 = np.array(point)
    print(np.abs(np.cross(p2-p1, p3-p1))/ np.linalg.norm(p2-p1))
    return np.abs(np.cross(p2-p1, p3-p1))/ np.linalg.norm(p2-p1)

def entropy(frame):
    marg = np.histogramdd(np.ravel(frame), bins = 256)[0]/frame.size
    marg = list(filter(lambda p: p > 0, np.ravel(marg)))
    entropy = -np.sum(np.multiply(marg, np.log2(marg)))
    return entropy

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    # img1 = img1.astype(np.float64)
    # img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    # img1 = img1.astype(np.float64)
    # img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def brightness(frame):
    frame = frame.astype('uint8')
    l_channel = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[...,0]
    brightness_sum = 0
    num_pixel = 0
    for row in l_channel:
        for pixel in row:
            brightness_sum += pixel
            num_pixel += 1
    
    return brightness_sum/num_pixel

def is_in_box(position,box):
    (x,y) = position
    (x1,y1,x2,y2) = box
    if (x>x1) and (y>y1) and (x<x2) and (y<y2):
        return True
    else:
        return False