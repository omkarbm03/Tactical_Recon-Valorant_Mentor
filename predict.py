import cv2
from ultralytics import YOLO

# Path to the video file
video_path = './Raw_Data/Dataset_Video/test.mp4'

# Path to trained weights 
weights_path = './Trained_Weights/best.pt'

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Blue, Green, Red

# Initialize the YOLO model with trained weights
model = YOLO(weights_path)

# Load the video capture object
cap = cv2.VideoCapture(video_path)

# class-specific messages
class_messages = {

    "B_Main_Wallbang": "You can spam through this wall using LMGs",
    "Throwables": "This can be used to throw utilities like nades",
    "B_Main_Corner": "Check this corner (Potential camping spot !)",
    "A_Door": "This door is breakable",
    "A_Heaven_Box": "Potential hidding spot",
    "A_Heaven_Wall": "You can spam through this wall using LMGs",
    "A_Hell": "Clear left corner (Potential camping spot !)",
    "A_Site_Box": "You can spam through this box using LMGs",
    "B_Boathouse": "Clear before planting/defusing (Potential camping spot !)",
    "B_Cubby": "Check cubby (Potential camping spot !)",
    "B_Garage_Left": "You can spam through this wall using LMGs",
    "B_Garage_Right": "You can spam through this wall using LMGs",
    "B_Site_Box": "You can spam through this box using LMGs",
    "Lion_Corner": "Check this corner (Potential camping spot !)",
    "Mid_Wall": "You can spam through this wall using LMGs",
    "Wine": "Potential hidding spot on the right"

}

# Initialize set to track processed classes
processed_classes = set()

# Loop through each frame in the video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the frame
    results = model(frame)

    class_texts = [] 

    # Process and display the results
    for result in results: 
        # Extract information from each frame
        for i in range(len(result.boxes.xyxy)):
            x_min, y_min, x_max, y_max = result.boxes.xyxy[i]
            conf = result.boxes.conf[i]
            class_id = result.boxes.cls[i]
            class_name = model.names[int(class_id)]

            # Draw bounding box 
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), colors[int(class_id) % len(colors)], 2)

            # Display class label and confidence score 
            text = f"{class_name}: {conf:.2f}"  
            cv2.putText(frame, text, (int(x_min), int(y_min) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[int(class_id) % len(colors)], 2)

            # Display class-specific message
            if class_name in class_messages and class_name not in processed_classes:
                message = class_messages[class_name]
                data = (class_id, message)
                class_texts.append(data)
                processed_classes.add(class_name)

    text_y = frame.shape[0] - 30  # Initial y-coordinate for the bottom-right corner
    for id, text in class_texts:
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 2)[0]
        text_y -= text_size[1] + 5  # Move up to avoid overlap

        cv2.rectangle(frame, (frame.shape[1]-text_size[0]-20, text_y - int(text_size[1]*1.2)), (frame.shape[1]-10, text_y), (255, 255, 255), -1)
        
        cv2.putText(frame, text, (frame.shape[1]-text_size[0]-10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, colors[int(id) % len(colors)], 2)

    processed_classes.clear()

    # Display the frame with detections
    cv2.namedWindow("Valorant Trainer", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Valorant Trainer", frame)
    cv2.resizeWindow("Valorant Trainer", 1280, 720)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()