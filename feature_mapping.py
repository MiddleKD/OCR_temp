import cv2

# Read the images
template = cv2.imread('./data/arrow_small.jpg', 0)
target = cv2.imread('./data/arrow_rain.jpg', 0)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Find keypoints and descriptors
keypoints1, descriptors1 = sift.detectAndCompute(template, None)
keypoints2, descriptors2 = sift.detectAndCompute(target, None)

# Use FLANN based matcher
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test
good_matches = []
for m, n in matches:
    if m.distance < 5 * n.distance:
        good_matches.append(m)

# Draw matches
# Draw matches with custom colors and thickness
img_matches = cv2.drawMatches(template, keypoints1, target, keypoints2, good_matches, None, 
                              matchColor=(127, 0, 0), # Green for matches
                              singlePointColor=(127, 0, 0), # Red for keypoints
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow('Feature-Based Matching', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()