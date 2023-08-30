# drawing = False
# top_left_pt, bottom_right_pt = (-1, -1), (-1, -1)

# def apply_canny(img, top_left, bottom_right):
#     roi = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
#     gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray_roi, 100, 200)
    
#     # Highlight edges in red
#     roi[edges > 0] = [0, 0, 255]

# def draw_rectangle(event, x, y, flags, param):
#     global drawing, top_left_pt, bottom_right_pt

#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         top_left_pt = (x, y)

#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False
#         bottom_right_pt = (x, y)
#         apply_canny(img, top_left_pt, bottom_right_pt)

# # Read the image
# img = cv2.imread('./data/large2.jpg')
# original_img = img.copy()

# cv2.namedWindow('Edge Highlight')
# cv2.setMouseCallback('Edge Highlight', draw_rectangle)

# while True:
#     cv2.imshow('Edge Highlight', img)

#     if cv2.waitKey(1) & 0xFF == 27:  # ESC key
#         img = original_img.copy()

#     elif cv2.waitKey(1) & 0xFF == 100:
#         break


# cv2.destroyAllWindows()


import cv2
import numpy as np
import random

drawing = False
top_left_pt, bottom_right_pt = (-1, -1), (-1, -1)

def draw_rectangle(event, x, y, flags, param):
    global drawing, top_left_pt, bottom_right_pt

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        top_left_pt = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bottom_right_pt = (x, y)

        component, largest_label = get_component_selected_area(img[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]])
        process(component, largest_label)

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
        cv2.imshow("temp", img)
        return

    unique_labels = np.unique(components_with_label)
    
    cmap = np.array([[random.randint(0, 255) for _ in range(3)] for _ in unique_labels])
    for idx, label in enumerate(unique_labels):
        if label != 0:
            img[components_with_label == label] = cmap[idx]
            cv2.imshow("temp", img)
            cv2.waitKey(0)

img = load_image('./data/0.jpg')
orignial_img = img.copy()
cv2.namedWindow('temp')
cv2.setMouseCallback('temp', draw_rectangle)

while True:
    cv2.imshow('temp', img)

    key = cv2.waitKey(1)

    if key == 27:  # ESC key
        img = orignial_img.copy()
        cv2.imshow('temp', img)

    elif key == 100:  # D key
        break

cv2.destroyAllWindows()

