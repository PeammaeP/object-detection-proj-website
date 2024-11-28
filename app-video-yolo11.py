import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter

# Load the model
model = YOLO('MU_Model_V4/MU_YOLO11_100epochS_V4.pt')


def detect_video_objects(video):
    # Open the input video
    cap = cv2.VideoCapture(video)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Tracking overall detections
    total_class_counts = Counter()
    timestamp_results = []

    # Process video frames at 1-second intervals
    current_time = 0
    while cap.isOpened():
        # Set frame position for 1-second interval
        frame_position = int(current_time * fps)

        # Break if we've exceeded total frames
        if frame_position >= total_frames:
            break

        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)

        # Read the frame
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection on the frame
        results = model(frame)[0]

        # Plot the frame with detections
        annotated_frame = results.plot()

        # Convert BGR to RGB (for Gradio)
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Track class counts for this timestamp
        timestamp_class_counts = Counter()
        timestamp_detection_info = []

        for box in results.boxes:
            # Get class name
            cls = results.names[int(box.cls[0])]

            # Get confidence
            conf = float(box.conf[0])

            # Track class counts
            timestamp_class_counts[cls] += 1
            total_class_counts[cls] += 1

            # Format individual detection info
            timestamp_detection_info.append(f"Class: {cls}, Confidence: {conf:.2f}")

        # Prepare timestamp detection details
        timestamp_detection_text = f"Timestamp: {current_time} seconds\n"
        if timestamp_detection_info:
            timestamp_detection_text += "\nIndividual Objects:\n" + "\n".join(timestamp_detection_info)

            # Calculate timestamp class counts
            timestamp_class_count_details = []
            timestamp_total_objects = sum(timestamp_class_counts.values())

            for cls, count in timestamp_class_counts.items():
                percentage = (count / timestamp_total_objects) * 100
                timestamp_class_count_details.append(f"{cls}: {count} ({percentage:.2f}%)")

            timestamp_detection_text += "\n\nTimestamp Class Counts:\n" + "\n".join(timestamp_class_count_details)
            timestamp_detection_text += f"\n\nObjects at this Timestamp: {timestamp_total_objects}"
        else:
            timestamp_detection_text += "No detections at this timestamp"

        # Store results for this timestamp
        timestamp_results.append({
            'time': current_time,
            'frame': annotated_frame_rgb,
            'details': timestamp_detection_text
        })

        # Move to next 1-second interval
        current_time += 1

    # Release video resources
    cap.release()

    return timestamp_results


def process_video(video):
    # Process the video and get timestamp results
    results = detect_video_objects(video)

    # If no results, return None
    if not results:
        return None, "No frames processed"

    # Return the results
    return results

# Create Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            upload_video = gr.Video(label="Upload Your Video", sources=["upload"])
            upload_button = gr.Button("Detect Objects")

        with gr.Column():
            # Slider to select timestamp
            timestamp_slider = gr.Slider(
                minimum=0,
                maximum=100,
                value=0,
                label="Select Timestamp",
                step=1
            )
            # Output components
            output_video = gr.Image(label="Video Frame Output")
            detection_output = gr.Textbox(label="Detection Details", lines=10)

    # Store results globally to access across different interactions
    results_state = gr.State([])

    # Video upload and detection
    upload_button.click(
        fn=process_video,
        inputs=upload_video,
        outputs=[results_state, detection_output]
    ).then(
        fn=lambda results: (
            results['frame'] if results else None,
            results['details'] if results else "No frames"
        ),
        inputs=results_state,
        outputs=[output_video, detection_output]
    )

    # Timestamp slider interaction
    timestamp_slider.change(
        fn=lambda slider, results:
        (results[slider]['frame'], results[slider]['details'])
        if results and 0 <= slider < len(results)
        else (None, "Invalid timestamp"),
        inputs=[timestamp_slider, results_state],
        outputs=[output_video, detection_output]
    )

# Launch the interface
demo.launch()