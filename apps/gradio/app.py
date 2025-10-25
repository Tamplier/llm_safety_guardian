import gradio as gr
from src.scripts.model_load import predict_with_proba as model

def predict(text):
    result = model([text]).to_dict(orient='records')
    return result[0]

app = gr.Interface(fn=predict, inputs="text", outputs="json")
