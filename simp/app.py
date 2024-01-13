import gradio as gr
from main import remove_bg

with gr.Blocks() as demo:
    dropdown = gr.Dropdown(["Auto Remove Background", "Select and Remove Background", "Auto Remove Object", "Select and Remove Object"])
    interface = gr.Interface(inputs=gr.Image(type="pil"), outputs="image")
    if dropdown.value == "Auto Remove Background":
        interface.fn = remove_bg
    # elif dropdown.value == "Select and Remove Background":
    #     interface.fn = 
if __name__ == "__main__":
    demo.launch()