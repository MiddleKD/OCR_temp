import cv2

def visualize(img_array):
    height, width = img_array.shape[:2]
    ratio = width/height

    maxheight = 800
    height = maxheight 

    cv2.namedWindow("temp", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("temp", int(height*ratio), height)
    cv2.imshow("temp", img_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def apply_post_process(img_array):
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    visualize(img_gray)

if __name__ == "__main__":
    img_array = cv2.imread("./data/large2.jpg")

    apply_post_process(img_array)
    # visualize(img_array)
