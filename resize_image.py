import cv2
import os

def resize_and_save(image_path):
    img = cv2.imread(image_path)
  
    if img is None:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return

    resized_img = cv2.resize(img, (1280, 720))

    base, ext = os.path.splitext(image_path)
    
    new_filename = f"{base}_size{ext}"

    cv2.imwrite(new_filename, resized_img)
    print(f"이미지 저장 완료: {new_filename}")


for i in range(20):
    filename = "C:/YOLOv8-multi-task/drive/"
    filename += (str(i) + ".jpg")
    resize_and_save(filename)  
