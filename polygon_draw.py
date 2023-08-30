import cv2
import numpy as np

drawing = False
point = (-1, -1)

def draw_circle(event, x, y, flags, param):
    global drawing, point

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(img, point, (x, y), (255, 0, 0), 2)
            point = (x, y)

# Create a black image
img = np.zeros((512, 512, 3), np.uint8)

cv2.namedWindow('Freehand Drawing')
cv2.setMouseCallback('Freehand Drawing', draw_circle)

while True:
    cv2.imshow('Freehand Drawing', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()