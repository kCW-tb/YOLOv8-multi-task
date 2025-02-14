import cv2
import time
import sys

sys.path.insert(0, "C:/YOLOv8-multi-task/ultralytics")

from ultralytics import YOLO
import ultralytics.yolo.engine.predictor_multi as pt

number = 3 #input how many tasks in your work
model = YOLO('C:/YOLOv8-multi-task/runs/multi/yolopm14/weights/best.pt')  # Validate the model
 
# Open the video file
video_path = "C:/YOLOv8-multi-task/autodrive.mp4"
cap = cv2.VideoCapture(video_path)
 
# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)  # Frame rate
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height
 
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
output_path = "autodrive_predict.mp4"
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
 
black= (0, 0, 0) 
font =  cv2.FONT_HERSHEY_PLAIN
 
frame_number = 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        cv2.imwrite("./runs/save_frame/frame_" + str(frame_number) + ".jpg", frame)     

        model.predict(source="./runs/save_frame/frame_" + str(frame_number) + ".jpg", 
            imgsz=(384,672), device=0, name='autodrive', save=True, conf=0.25, iou=0.45, show_labels=False, speed=True)

        print("C:/YOLOv8-multi-task/runs/multi/autodrive/frame_" + str(frame_number) + ".jpg")
        annotated_frame = cv2.imread("C:/YOLOv8-multi-task/runs/multi/autodrive/frame_" + str(frame_number) + ".jpg")

        cv2.imshow("YOLO Inference", annotated_frame)

        frame_number+=1

        predict_time= pt.only_inference_time

        print("video.py : ",pt.only_inference_time)

        annotated_frame = cv2.putText(annotated_frame, "predict_time : "
                                       + predict_time, (20, 40), font, 2, black, 1, cv2.LINE_AA)

        # Write the annotated frame to the output video
        out.write(annotated_frame)
        # Display the annotated frame
        cv2.imshow("YOLO Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break
 
cap.release()
out.release()
cv2.destroyAllWindows()
