import gradio as gr
import torch
from PIL import Image, ImageDraw
import numpy as np

from ultralytics.nn.tasks import DetectionModel  # Import the class used in the model

torch.serialization.add_safe_globals([DetectionModel])  # Allow the specific class

# Now load the model
model = torch.load('MU_100epochS_V2.3.pt')

# Define a function for object detection
def detect_objects(image):
    # Convert the PIL image to a tensor
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()  # Convert image to tensor
    ])
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        results = model(input_tensor)

    # Assuming your model outputs bounding boxes, class indices, and confidences
    detections = results[0]  # Adjust based on your model's output structure

    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(image)
    for box in detections['boxes']:  # Assuming detections is a dictionary
        xmin, ymin, xmax, ymax = box.tolist()
        confidence = detections['scores'][i].item()
        class_id = int(detections['labels'][i].item())
        class_name = f"Class {class_id}"  # Replace with your class names if available
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
        draw.text((xmin, ymin), f"{class_name} {confidence:.2f}", fill="red")

    return image  # Return the annotated image

# Define the Gradio interface
interface = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="pil"),  # Accept PIL Image as input
    outputs=gr.Image(type="pil"),  # Return PIL Image as output
    title="Custom Model Object Detection",
    description="Upload an image, and the custom model will detect objects in it."
)

# Launch the interface on localhost
interface.launch()
