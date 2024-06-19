import os
import cv2

# This to resize the frames extraceted from the video before feeding to the model for training.
def resize_images(input_folder, output_folder, target_size=(640, 640)):
   
    os.makedirs(output_folder, exist_ok=True)
    
  
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
       
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            
          
            resized_image = cv2.resize(image, target_size)
            
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, resized_image)
            print(f"Resized and saved {filename} to {output_folder}")


input_folder = "./Raw_Data/Dataset_Frames"
output_folder = "./Raw_Data/Dataset_Frames_rsz"
resize_images(input_folder, output_folder)
