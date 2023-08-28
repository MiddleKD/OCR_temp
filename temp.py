import matplotlib.pyplot as plt
import keras_ocr

detector = keras_ocr.detection.Detector()
recognizer = keras_ocr.recognition.Recognizer()

image = keras_ocr.tools.read('large2.jpg')
boxes = detector.detect(images=[image])[0]

# canvas = keras_ocr.tools.drawBoxes(image, boxes)

recognized_text = recognizer.recognize_from_boxes(images=[image], box_groups=[boxes])[0]

# 인식된 텍스트와 해당 영역을 출력합니다.
for idx, text in enumerate(recognized_text):
    print(f"Text: {text}, Box_idx: {idx}")

# 이미지에 박스와 텍스트를 그립니다.
fig, ax = plt.subplots(1)
ax.imshow(image)

for text, box in zip(recognized_text, boxes):
    # 박스 좌표
    start_point = box[0]  # 시작점 (왼쪽 상단)
    end_point = box[2]  # 종료점 (오른쪽 하단)
    
    # 박스 그리기
    rect = plt.Rectangle(start_point, end_point[0]-start_point[0], end_point[1]-start_point[1], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    # 텍스트 추가 (보통은 시작점에 추가합니다)
    plt.annotate(text, start_point-10, color='blue', fontsize=7)

plt.savefig("result_img.jpg")
plt.show()

# plt.imshow(canvas)
# plt.show()

