import gradio as gr
from ultralytics import YOLO
import cv2

# Now load the model
model = YOLO('MU_Model_V4/MU_YOLO11_100epochS_V4.pt')

# Define a function for object detection
def detect_objects(source):
    # Perform detection
    for results in model(source, stream=True):  # generator of Results objects
        # Extract frame from results
        frame = results.plot()  # Assuming single image output
        return frame

# Define the Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Input", sources=["webcam" , "upload"])
        with gr.Column():
            output_img = gr.Image(label="Output")

        input_img.stream(detect_objects, input_img, output_img, time_limit=0.1, stream_every=0.05, concurrency_limit=60 , show_api = True)

# Launch the interface on localhost
demo.launch()