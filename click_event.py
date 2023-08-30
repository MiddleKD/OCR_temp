import cv2
import numpy as np
import random

# Global variables to store the two points
points = []
img_copy = None
top_left_pt, bottom_right_pt = (-1, -1), (-1, -1)

def mouse_callback(event, x, y, flags, param):
    global points, img_copy, top_left_pt, bottom_right_pt

    # Check for left mouse button click
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

        # If two points are selected, draw the rectangle
        if len(points) == 2:
            cv2.imshow('Image', img_copy)
            mask = create_mask(img_copy, points)
            
            image_mask_combined = img * mask
            img_white = np.ones(img.shape) * 255
            masked_img = np.where(mask==1, image_mask_combined, img_white).astype(np.uint8)

            contours, _ = cv2.findContours(cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            largest_area = 0
            largest_rect = None
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                if area > largest_area:
                    largest_area = area
                    largest_rect = (x, y, x+w, y+h)

            top_left_pt = largest_rect[:2]
            bottom_right_pt = largest_rect[2:]
            
            component, largest_label = get_component_selected_area(masked_img[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]])
            process(component, largest_label)

            points = []  # Reset the points

def create_mask(img, points, thickness=8):
    # Create a mask of zeros with the same shape as the input image
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    
    # Draw a line on the mask using the points, with the specified thickness
    cv2.line(mask, points[0], points[1], 1, thickness)
    
    return cv2.merge([mask,mask,mask])

def load_image(image_path):
    img = cv2.imread(image_path)
    return img

def get_component_selected_area(target_roi):
    roi_gray = cv2.cvtColor(target_roi, cv2.COLOR_BGR2GRAY)
    a, binary_roi = cv2.threshold(roi_gray, 140, 255, cv2.THRESH_BINARY_INV)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # dilated_image = cv2.dilate(binary_image, kernel, iterations=3)

    num_labels, image_components, stats, _ = cv2.connectedComponentsWithStats(binary_roi)
    
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    return image_components, largest_label


def process(components_with_label, target_label = None):
    if target_label != None:
        img[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]]\
            [components_with_label == target_label] = [255,0,0]
        cv2.imshow("Image", img)
        return

    unique_labels = np.unique(components_with_label)
    
    cmap = np.array([[random.randint(0, 255) for _ in range(3)] for _ in unique_labels])
    for idx, label in enumerate(unique_labels):
        if label != 0:
            img[components_with_label == label] = cmap[idx]
            cv2.imshow("Image", img)
            cv2.waitKey(0)

# Read the image
img = load_image('./data/0.jpg')
img_copy = img.copy()

# Set the mouse callback
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_callback)

while True:
    cv2.imshow('Image', img)

    key = cv2.waitKey(1)

    if key == 27:  # ESC key
        img = img_copy.copy()
        cv2.imshow('Image', img)

    elif key == 100:  # D key
        break

cv2.destroyAllWindows()

