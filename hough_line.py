import cv2
import numpy as np

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

        get_component_selected_area(img[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]])


def get_component_selected_area(target_roi):

    global img
    gray = cv2.cvtColor(target_roi, cv2.COLOR_BGR2GRAY)

    # Detect edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    # Draw lines
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        
        overlay = img.copy()
        cv2.line(overlay[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]], (x1, y1), (x2, y2), (0, 0, 255), 2)
        img = apply_transparency(overlay, alpha=0.3)

def apply_transparency(overlay, alpha=0.5):
    return (img * (1 - alpha) + overlay * alpha).astype(np.uint8)


# Read image
img = cv2.imread('./data/1.jpg')
orignial_img = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect edges
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Detect lines
# lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

# # Draw lines
# for rho, theta in lines[:, 0]:
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a * rho
#     y0 = b * rho
#     x1 = int(x0 + 1000 * (-b))
#     y1 = int(y0 + 1000 * (a))
#     x2 = int(x0 - 1000 * (-b))
#     y2 = int(y0 - 1000 * (a))
#     cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

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