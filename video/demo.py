#1, 分解视频
import cv2

def video_to_frames(video_path, frames_dir):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(f"{frames_dir}/frame{count}.jpg", image)  # save frame as JPEG file
        success, image = vidcap.read()
        print(f'Reading new frame: {success}')  # Log the status
        count += 1

#2, 通过yolo5进行目标检测
import torch
from pathlib import Path
from glob import glob

def detect_frames(frames_dir):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # Load the YOLOv5 model

    frames = glob(f'{frames_dir}/*.jpg')
    for frame_path in frames:
        results = model(frame_path)
        results.save(Path(frame_path).parent)  # Save the result in the frame's directory
        results_info = results.pandas().xyxy[0]
        print(f'Detected and saved: {frame_path}')

# 完成目标检测后，你可能希望将带有检测结果的帧合并回视频
def frames_to_video(frames_dir, output_video_path, fps=25):
    img_array = []
    for filename in sorted(glob(f'{frames_dir}/*.jpg'), key=lambda x: int(x.split('frame')[1].split('.jpg')[0])):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

video_path = 'path/to/your/video.mp4'
frames_dir = 'path/to/save/frames'
output_video_path = 'path/to/save/output_video.mp4'

video_to_frames(video_path, frames_dir)
detect_frames(frames_dir)
frames_to_video(frames_dir, output_video_path)

