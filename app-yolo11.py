import gradio as gr
import cv2
from ultralytics import YOLO
import numpy as np

# Load YOLO model
model = YOLO('MU_Model_V4/MU_YOLO11_100epochS_V4.pt')

def detect_objects_upload_image(image):
    """
    Detect objects in a single image and return annotated image with label counts
    """
    if image is None:
        return None, {}

    # Perform detection
    results = model(image)

    # Initialize label counts
    label_counts = {}

    # Annotate the image and count labels
    for result in results:
        for box in result.boxes.data:
            label = int(box[5].item())
            class_name = result.names[label]
            label_counts[class_name] = label_counts.get(class_name, 0) + 1

    # Add total count
    label_counts["Total"] = sum(label_counts.values())

    # Plot annotated image
    annotated_image = results[0].plot()

    return annotated_image, label_counts

def detect_objects_camera_streaming(image):
    """
    Detect objects in streaming camera input
    """
    if image is None:
        return None, {}

    # Perform detection
    results = model(image)

    # Initialize label counts
    label_counts = {}

    # Annotate the image and count labels
    for result in results:
        for box in result.boxes.data:
            label = int(box[5].item())
            class_name = result.names[label]
            label_counts[class_name] = label_counts.get(class_name, 0) + 1

    # Add total count
    label_counts["Total"] = sum(label_counts.values())

    # Plot annotated image
    annotated_image = results[0].plot()

    return annotated_image, label_counts

def detect_objects_from_video(video):
    """
    Detect objects in a video and return last frame with label counts
    """
    if video is None:
        return None, {}

    # Initialize label counts
    label_counts = {}
    output_frames = []

    # Process video frame-by-frame
    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection on the current frame
        results = model(frame)
        for result in results:
            for box in result.boxes.data:
                label = int(box[5].item())
                class_name = result.names[label]
                label_counts[class_name] = label_counts.get(class_name, 0) + 1

        # Add annotated frame
        annotated_frame = results[0].plot()
        output_frames.append(annotated_frame)

    cap.release()

    # Create summary of detections
    label_counts["Total"] = sum(label_counts.values())

    # Return the last frame and detection summary
    last_frame = output_frames[-1] if output_frames else None
    return last_frame, label_counts

# Define the Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", sources=["webcam", "upload"])
            input_streaming_image = gr.Image(label="Open Camera", sources=["webcam"])
            input_video = gr.Video(label="Upload Video", sources=["upload", "webcam"])

        with gr.Column():
            output_image = gr.Image(label="Processed Image")
            output_streaming_image = gr.Image(label="Camera Stream Result")
            output_video = gr.Image(label="Processed Video Frame")
            output_summary = gr.JSON(label="Detection Summary")

    # Image detection
    detect_image_btn = gr.Button("Detect in Upload Image")
    detect_image_btn.click(
        detect_objects_upload_image,
        inputs=input_image,
        outputs=[output_image, output_summary]
    )

    # Streaming camera detection
    input_streaming_image.stream(
        detect_objects_camera_streaming,
        inputs=input_streaming_image,
        outputs=[output_streaming_image, output_summary],
        time_limit=0.1,
        stream_every=0.05,
        concurrency_limit=120,
        show_api=True
    )

    # Video detection
    detect_video_btn = gr.Button("Detect in Video")
    detect_video_btn.click(
        detect_objects_from_video,
        inputs=input_video,
        outputs=[output_video, output_summary]
    )

# Launch the interface
demo.launch(
    server_port=7860,
)