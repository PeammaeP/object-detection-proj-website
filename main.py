from fastapi import FastAPI
import gradio as gr

from model import demo

app = FastAPI()

@app.get("/")
async def root():
    return "Parking Lot Website in running on /model" , 200

app = gr.mount_gradio_app(app , demo , path="/model")