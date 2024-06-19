import cv2
import os

# This is to extract frames from the video for creating the dataset.
def extract_frames(video_path, output_dir, interval=1):
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_extraction_interval = int(fps * interval)
    

    frame_count = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if frame_count % frame_extraction_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
        
        frame_count += 1
        
    cap.release()
    print(f"Frames extracted successfully to {output_dir}")

# Path to the input video file
video_path = "./Raw_Data/Dataset_Video/Valorant_Trim_Data.mp4"

#Path to the output directory
output_dir = "./Raw_Data/Dataset_Frames"

interval = 0.2  # Extract every 5th frame
extract_frames(video_path, output_dir, interval)
