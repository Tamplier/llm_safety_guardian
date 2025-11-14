import gradio as gr
from src.scripts.model_load import predict_with_proba as model

def predict(text):
    result = model([text]).to_dict(orient='records')
    return result[0]

with gr.Blocks() as app:
    gr.Markdown("""
    ### An attempt to identify suicidal mood in messages in natural English.
    *The classification accuracy is usually higher for messages longer than 15 characters.*
    """)

    with gr.Row():
        with gr.Column():
            txt = gr.Textbox(lines=3, placeholder="Enter your message...")
            char_count = gr.Textbox(label="Character counter", interactive=False, max_lines=1)
        with gr.Column():
            result_out = gr.JSON(label="Output")
            submit_btn = gr.Button("Submit")

    txt.change(len, txt, char_count)
    submit_btn.click(predict, txt, result_out)
