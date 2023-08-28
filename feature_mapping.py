import cv2
import numpy as np
# 원본 이미지와 템플릿 이미지 불러오기
large_image = cv2.imread('./data/arrow_rain.jpg', 0)
small_image = cv2.imread('./data/arrow.jpg', 0)


# ORB 특성 검출기 생성
orb = cv2.ORB_create()

# 특성점과 특성 기술자 추출
kp1, des1 = orb.detectAndCompute(large_image, None)
kp2, des2 = orb.detectAndCompute(small_image, None)


# BFMatcher 객체 생성
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 특성점 매칭
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# 매칭된 특성점들을 이용해 어떻게든 'small_image'의 위치를 찾을 수 있습니다.
# 여기에서는 간단하게 처음 매칭된 특성점만 사용합니다.

if matches:
    img_idx = matches[0].queryIdx
    templ_idx = matches[0].trainIdx

    # 매칭된 특성점의 위치
    x, y = kp1[img_idx].pt

    cv2.circle(large_image, (int(x), int(y)), 10, (0, 0, 0), 2)

# 결과 출력
cv2.imshow('Matched Features', large_image)
cv2.waitKey(0)
cv2.destroyAllWindows()