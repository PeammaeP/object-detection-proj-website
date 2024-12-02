# Using CV2
import cv2
from ultralytics import YOLO
import torch

# Save configuration
save = False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLO("../MU_Model_V4/MU_50epochOBB11s_V1.2.pt")
model = model.to(device)

import cv2
from ultralytics import YOLO

# Define path to video file
# source = "C:/Users/supak/Downloads/parking_CCTV_mock_car_move_1080.mp4"
source = "source-parking.mp4"

# Optional: Define output video parameters (adjust as desired)
output_filename = "test.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
fps = 60
video_writer = ""

# Run inference on the video stream
for results in model(source, stream=True):  # generator of Results objects

    # Extract frame from results
    frame = results.plot()  # Assuming single image output
    frame_width, frame_height = frame.shape[1], frame.shape[0]

    # Create video writer object (if not already created)
    if not video_writer and save:
        video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    # Process results generator
    print("Bounding Boxes:")
    print(results.boxes)

    print("Segmentation Masks:")
    print(results.masks)

    print("Keypoints:")
    print(results.keypoints)

    print("Classification Probabilities:")
    print(results.probs)

    print("Oriented Bounding Boxes:")
    print(results.obb)

    print("Oriented Bounding Boxes Conf:")
    obb_conf = results.obb.conf  # Assuming results.obb is a valid YOLO obb object
    print(obb_conf)
    # Move the tensor to CPU if it's on GPU for efficient NumPy conversion
    if obb_conf.is_cuda:
        obb_conf = obb_conf.cpu()

    # Detach the tensor from the computational graph (optional for memory efficiency)
    obb_conf = obb_conf.detach()

    # Convert the tensor to a NumPy list
    obb_conf_list = obb_conf.numpy().tolist()

    print("Extracted Oriented Bounding Boxes Conf:")
    print(obb_conf_list)
    
    print("-" * 40)  # Separator for each result
    # results.show()  # display to screen

    # # Display the processed frame
    # cv2.imshow('Processed Frame', frame)
    # # Exit if 'q' is pressed
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #      break
    
    # Write the processed frame to the output video
    if(save): video_writer.write(frame)

# Release resources
if(save): video_writer.release()