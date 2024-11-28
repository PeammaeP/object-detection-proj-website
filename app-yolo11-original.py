import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np

# Load the model
model = YOLO('MU_Model_V4/MU_YOLO11_100epochS_V4.pt')

# Define a function for object detection
def detect_objects(source):
    # Perform detection
    results = model(source)[0]  # Get the first (and only) result

    # Plot the frame with detections
    annotated_frame = results.plot()

    # Prepare detection details
    detection_info = []
    for box in results.boxes:
        # Get class name
        cls = results.names[int(box.cls[0])]

        # Get confidence
        conf = float(box.conf[0])

        # Get bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Format detection info
        detection_info.append(f"Class: {cls}, Confidence: {conf:.2f}, Bbox: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")

    # Combine detection info into a single string
    detection_text = "\n".join(detection_info) if detection_info else "No detections"

    return annotated_frame, detection_text


# Define the Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Input", sources=["webcam", "upload"])

        with gr.Column():
            output_img = gr.Image(label="Detected Image")
            detection_output = gr.Textbox(label="Detection Details")

    # Bind the detection function to the input image
    input_img.stream(
        detect_objects,
        inputs=input_img,
        outputs=[output_img, detection_output],
        time_limit=0.1,
        stream_every=0.05,
        concurrency_limit=60
    )

# Launch the interface
demo.launch()