import cv2
import numpy as np

# Create a black image with a white rectangle and circle
img = np.zeros((500, 500), dtype=np.uint8)
cv2.rectangle(img, (100, 100), (400, 400), 255, -1)
cv2.circle(img, (250, 250), 50, 0, -1)

# Display the original image
cv2.imshow('Original Image', img)

# Define the ROI coordinates
x, y, w, h = 150, 150, 200, 200

# Crop the ROI from the original image
roi = img[y:y+h, x:x+w]

# Display the ROI
cv2.imshow('ROI', roi)
cv2.waitKey(0)
# Perform circle detection only on the ROI
# gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
gray = roi
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=50, param1=100, param2=30, minRadius=0, maxRadius=100)

# If some circles are detected, draw them
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (c_x, c_y, r) in circles:
        cv2.circle(roi, (c_x, c_y), r, (0, 255, 0), 3)

# Display the ROI with detected circles
cv2.imshow('Detected Circles in ROI', roi)

# Wait for a key press and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()