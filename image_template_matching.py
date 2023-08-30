import cv2
import numpy as np
from tqdm import tqdm

def non_max_suppression(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    
    pick = []
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    scores = boxes[:,4]
    
    idxs = np.argsort(scores)
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        overlap = (w * h) / ((x2[i] - x1[i]) * (y2[i] - y1[i]))
        
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    
    return boxes[pick].astype("int")

# Read the images
template = cv2.imread('./data/arrow_square.jpg', 0)
target = cv2.imread('./data/large2.jpg', 0)


def draw_matched_area(template, threshold=0.65):

    result = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)

    loc = np.where(result >= threshold)

    boxes = []

    for pt in zip(*loc[::-1]):
        boxes.append([pt[0], pt[1], pt[0] + template.shape[1], pt[1] + template.shape[0], result[pt[1], pt[0]]])

    boxes = np.array(boxes)
    pick = non_max_suppression(boxes, 0.4)

    for (startX, startY, endX, endY, _) in pick:
        cv2.rectangle(target, (startX, startY), (endX, endY), (127, 0, 0), 2)

    cv2.resizeWindow("temp", 1000, 1000)
    cv2.imshow('temp', target)
    # cv2.waitKey(0)

cv2.namedWindow("temp", cv2.WINDOW_NORMAL)
cv2.resizeWindow("temp", 1000, 1000)

for angle in tqdm(np.arange(0, 360, 1)):
    # Rotate the template
    M = cv2.getRotationMatrix2D((template.shape[1] / 2, template.shape[0] / 2), angle, 1)
    rotated_template = cv2.warpAffine(template, M, (template.shape[1], template.shape[0]), borderValue=(255,255,255))

    # cv2.imshow("temp", rotated_template)
    # cv2.waitKey(0)
    draw_matched_area(rotated_template, 0.83)

cv2.imshow('temp', target)
cv2.waitKey(0)
cv2.destroyAllWindows()

