import gradio as gr
import cv2
from ultralytics import YOLO
import numpy as np
import torch

# Load YOLO model
model = YOLO('../MU_Model_V4/MU_50epochOBB11s_V1.2.pt')

def safe_detect_objects(input_data, is_video=False):
    """
    Safe object detection function with comprehensive error handling
    """
    if input_data is None:
        return None, {}

    # Initialize label counts
    label_counts = {}
    output_frames = []

    try:
        # Handle video input differently
        if is_video:
            cap = cv2.VideoCapture(input_data)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Perform detection with explicit handling
                results = model(frame)

                # Debug: Print results type and content
                print(f"Results type: {type(results)}")
                print(f"Results content: {results}")

                # Check if results is valid and not None
                if results is not None:
                    # Ensure results is a list
                    if not isinstance(results, list):
                        results = [results]

                    for result in results:
                        # Check if result has detection information
                        if result is not None and hasattr(result, 'boxes') and result is not None:
                            # Count labels from boxes
                            # for box in result.boxes:
                            #     try:
                            #         # Safely extract label
                            #         label = box.cls[0].item()
                            #         class_name = result.names.get(label, f'Unknown_{label}')
                            #         label_counts[class_name] = label_counts.get(class_name, 0) + 1
                            #     except Exception as e:
                            #         print(f"Individual box processing error: {e}")
                            for obb in result.obb:
                                try:
                                    # Safely extract label
                                    label = obb.cls[0].item()
                                    print("LABEL OBB" + label)
                                    class_name = result.names.get(label, f'Unknown_{label}')
                                    print("CLASS NAME" + class_name)
                                    label_counts[class_name] = label_counts.get(class_name, 0) + 1
                                    print("LABEL COUNTs" + label_counts[class_name])
                                except Exception as e:
                                    print(f"Individual box processing error: {e}")
                            # Annotate frame
                            try:
                                annotated_frame = result.plot()
                                output_frames.append(annotated_frame)
                            except Exception as plot_error:
                                print(f"Frame annotation error: {plot_error}")
                                output_frames.append(frame)

            cap.release()
            last_frame = output_frames[-1] if output_frames else None
            return last_frame, label_counts

        else:
            # Handle single image input
            results = model(input_data)

            # Debug: Print results type and content
            print(f"Results type: {type(results)}")
            print(f"Results content: {results}")

            # Check if results is valid and not None
            if results is not None:
                # Ensure results is a list
                if not isinstance(results, list):
                    results = [results]

                for result in results:
                    # Check if result has detection information
                    if result is not None and hasattr(result, 'boxes') and result.boxes is not None:
                        # Count labels from boxes
                        # for box in result.boxes:
                        #     try:
                        #         # Safely extract label
                        #         label = box.cls[0].item()
                        #         class_name = result.names.get(label, f'Unknown_{label}')
                        #         label_counts[class_name] = label_counts.get(class_name, 0) + 1
                        #     except Exception as e:
                        #         print(f"Individual box processing error: {e}")
                        for obb in result.obb:
                            try:
                                # Safely extract label
                                label = obb.cls[0].item()
                                print("LABEL OBB" + label)
                                class_name = result.names.get(label, f'Unknown_{label}')
                                print("CLASS NAME" + class_name)

                                label_counts[class_name] = label_counts.get(class_name, 0) + 1
                                print("LABEL COUNTs" + label_counts[class_name])
                            except Exception as e:
                                print(f"Individual box processing error: {e}")
                        # Annotate image
                        try:
                            annotated_image = result.plot()
                            myObb = result.obb
                            print("MY_OBB : " + myObb)
                            return annotated_image, label_counts
                        except Exception as plot_error:
                            print(f"Image annotation error: {plot_error}")
                            return input_data, label_counts

        # If no detection occurs
        return input_data , label_counts

    except Exception as e:
        print(f"Comprehensive detection error: {e}")
        return input_data , label_counts


def detect_objects_upload_image(image):
    """
    Detect objects in a single image
    """
    return safe_detect_objects(image)


def detect_objects_camera_streaming(image):
    """
    Detect objects in streaming camera input
    """
    return safe_detect_objects(image)


def detect_objects_from_video(video):
    """
    Detect objects in a video
    """
    return safe_detect_objects(video, is_video=True)

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