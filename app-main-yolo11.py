import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter
import tempfile
import os

# Load the model
model = YOLO('MU_Model_V4/MU_YOLO11_100epochS_V4.pt')


# Define a function for object detection
def detect_objects(source, is_video=False):
    # If it's a video path, process the video
    if is_video:
        # Create a temporary output directory for frames
        output_dir = tempfile.mkdtemp()

        # Open the input video
        cap = cv2.VideoCapture(source)

        # Tracking overall detections
        total_class_counts = Counter()
        frame_images = []

        # Process video frames
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform detection on the frame
            results = model(frame)[0]

            # Plot the frame with detections
            annotated_frame = results.plot()

            # Convert BGR to RGB (for Gradio)
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Track class counts for this frame
            frame_class_counts = Counter()

            for box in results.boxes:
                # Get class name
                cls = results.names[int(box.cls[0])]

                # Track class counts
                frame_class_counts[cls] += 1
                total_class_counts[cls] += 1

            # Save frame as image
            frame_path = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
            cv2.imwrite(frame_path, annotated_frame)

            # Append frame to list
            frame_images.append(annotated_frame_rgb)

            frame_count += 1

        # Release video resources
        cap.release()

        # Prepare detection details
        class_count_details = []
        total_objects = sum(total_class_counts.values())

        for cls, count in total_class_counts.items():
            percentage = (count / total_objects) * 100
            class_count_details.append(f"{cls}: {count} ({percentage:.2f}%)")

        # Combine detection details
        detection_text = "Video Detections:\n"
        detection_text += f"\nTotal Frames Processed: {frame_count}"
        detection_text += "\n\nClass Counts:\n" + "\n".join(class_count_details)
        detection_text += f"\n\nTotal Objects Detected: {total_objects}"

        return frame_images, detection_text

    # For image detection (same as before)
    else:
        # Perform detection
        results = model(source)[0]  # Get the first (and only) result

        # Plot the frame with detections
        annotated_frame = results.plot()

        # Prepare detection details
        detection_info = []
        class_counts = Counter()

        for box in results.boxes:
            # Get class name
            cls = results.names[int(box.cls[0])]

            # Get confidence
            conf = float(box.conf[0])

            # Track class counts
            class_counts[cls] += 1

            # Format individual detection info
            detection_info.append(f"Class: {cls}, Confidence: {conf:.2f}")

        # Prepare class count details
        class_count_details = []
        total_objects = sum(class_counts.values())

        for cls, count in class_counts.items():
            percentage = (count / total_objects) * 100
            class_count_details.append(f"{cls}: {count} ({percentage:.2f}%)")

        # Combine detection details
        detection_text = "Detections:\n"
        if detection_info:
            detection_text += "\nIndividual Objects:\n" + "\n".join(detection_info)
            detection_text += "\n\nClass Counts:\n" + "\n".join(class_count_details)
            detection_text += f"\n\nTotal Objects Detected: {total_objects}"
        else:
            detection_text = "No detections"

        return annotated_frame, detection_text

# Define the Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Open Your Webcam", sources=["webcam"])
            upload_img = gr.Image(label="Upload Your Image", sources=["upload"])
            upload_video = gr.Video(label="Upload Your Video", sources=["upload"])

        with gr.Column():
            output_img = gr.Image(label="Detected Webcam Image")
            output_upload_img = gr.Image(label="Image Output")
            output_upload_video = gr.Gallery(label="Video Frames Output", columns=3, rows=3, height="auto")
            detection_output = gr.Textbox(label="Detection Details", lines=10)

    # Webcam detection
    input_img.stream(
        detect_objects,
        inputs=input_img,
        outputs=[output_img, detection_output],
        time_limit=0.1,
        stream_every=0.05,
        concurrency_limit=60
    )

    # Image upload detection
    upload_img_button = gr.Button("Upload Image")
    upload_img_button.click(
        detect_objects,
        inputs=upload_img,
        outputs=[output_upload_img, detection_output],
    )

# Launch the interface
demo.launch()