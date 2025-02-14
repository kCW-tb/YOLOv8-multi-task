#label이 yolo format으로 되어있을 경우 txt파일을 읽고 원본 이미지에 구현하는 코드

import cv2
import numpy as np
import os
from glob import glob

#path
image_folder = 'C:/datasets/bdd100k/images/train2017'
label_folder = 'C:/datasets/bdd100k/seg-drivable-10/labels/train2017'
output_folder = './segmented_output'

os.makedirs(output_folder, exist_ok=True)

image_files = glob(os.path.join(image_folder, '*.jpg'))  # JPG 이미지 대상

for image_path in image_files:
    filename = os.path.basename(image_path).replace('.jpg', '')
    txt_file_path = os.path.join(label_folder, filename + '.txt')

    # 원본 이미지 로드
    original_image = cv2.imread(image_path)
    img_height, img_width = original_image.shape[:2]

    mask = np.zeros(original_image.shape[:2], dtype=np.uint8)

    with open(txt_file_path, 'r') as file:
        lines = file.readlines()

    # 각 줄을 읽으며 라벨링 마스크 생성
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])  # 클래스와 ID 따로 저장 -> ID가 처음에 위치할 때.
        
        # 0~1로 정규화된 좌표에 대해 -> 이미지 사이즈에 맞게 픽셀 좌표 변환
        mask_points = np.array([
            [float(parts[i]) * img_width, float(parts[i+1]) * img_height] 
            for i in range(1, len(parts), 2)
        ], dtype=np.int32)

        cv2.fillPoly(mask, [mask_points], 255)

    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    masked_image = cv2.addWeighted(original_image, 1.0, mask_3ch, 0.5, 0)
    cv2.imshow("masked_image", masked_image)
    
    # 결과 저장
    output_path = os.path.join(output_folder, filename + '_masked.jpg')
    cv2.imwrite(output_path, masked_image)

    cv2.waitKey()

    print(f"저장: {output_path}")
