import argparse
import gradio as gr
import re
import os
from backend.rag import RAG

class DummyRAG:
    def __init__(self, config_path=None):
        pass
    def generate_response(self, chat_history, model, domain, retrieval_method, top_k):
        chat_history.append({"role": "assistant", "content": f"Dummy response with model={model}, domain={domain}, method={retrieval_method}, top_k={top_k}"})
        source_text = ""
        context_text = "- [1] **Source Title 1** : This is the content of source 1.\n- [2] **Source Title 2** : This is the content of source 2."
        return chat_history, source_text, context_text
    def clear_chat(self):
        return [], "", ""

def main(config_path):
    rag = RAG(config_path)
    gradio_app(rag)


def user_input(user_message, chat_history):
    """
    Handle user input and add it to the chat history.
    :param user_message: The message input by the user.
    :param chat_history: The current chat history.
    :return: Updated chat history with the new user message.
    """

    chat_history.append({"role":"user","content":user_message})
    return "", chat_history


def parse_sources(raw_text):
    pattern = r"- \[(\d+)\] \*\*(.*?)\*\* \(Source=(.*?), Pos=(.*?)\): (.*?)(?=\n- |\Z)"
    matches = re.findall(pattern, raw_text, re.DOTALL)
    sources = []
    for num, title, source_id, pos, content in matches:
        sources.append({
            "num": num.strip(),
            "title": title.strip(),
            "source_id": source_id.strip(),
            "position": pos.strip(),
            "content": content.strip()
        })
    return sources


def render_sources(raw_text):
    sources = parse_sources(raw_text)   # convert string -> list of dicts
    if not sources:
        return raw_text
    out = ""
    for src in sources:
        out += f"""
<details class="source-card">
  <summary class="source-title">
    📖 [{src['num']}] ({src['source_id']}) {src['title']}
  </summary>
  <div class="source-content">
    <i>[Chunk position: {src['position']}]</i>
    <br>{src['content']}
  </div>
</details>
"""
    return out

# ==================== GRADIO APP ====================
def gradio_app(rag):
    with gr.Blocks(css_paths="interfaz/style.css", theme=gr.themes.Soft()) as demo:
        gr.HTML(f"<center id='title'><h1>💬 Demostrador RAG</h1></center>")
        with gr.Row(equal_height=True, elem_id="main_row"):
            with gr.Column(scale=5):
                chatbot = gr.Chatbot(
                    elem_id = "chatbot",
                    type="messages",
                    avatar_images=None,
                    height="60vh"
                )
        
                with gr.Row(equal_height=True):
                    clear_button = gr.Button("🔁 Nuevo chat", scale=1)
                    msg = gr.Textbox(
                        placeholder="Escribe tu pregunta...",
                        lines=1,
                        show_label=False,
                        submit_btn=True,
                        stop_btn=True,
                        scale=5
                    )

                with gr.Row():
                    model = gr.Dropdown(
                        ["Model1", "Model2"], label="Modelo", info="Elegir un modelo"
                    )
                    domain = gr.Dropdown(
                        ["Noticias", "Boletines"], label="Dominio", info="Elegir un dominio"
                    )
                    retrieval_method = gr.Dropdown(
                        ["BM25", "Embeddings"], label="Método de recuperación", info="Elegir un método de recuperación"
                    )
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        label="Top K",
                        info="Seleccionar el número de resultados a devolver"
                    )

            with gr.Column(scale=2):   
                with gr.Column(variant="panel", elem_id="sources_column"):
                    with gr.Accordion("📚 Fuentes:", open=True):
                        source_context = gr.Markdown(show_label=False)
                        context_evaluation = gr.Markdown(
                            visible=False,
                            show_copy_button=True
                        )
                        sources_md = gr.Markdown()
                        context_evaluation.change(render_sources, inputs=context_evaluation, outputs=sources_md)


        msg.submit(user_input, [msg, chatbot], [msg, chatbot], queue=False).then(
            rag.generate_response,
            [chatbot, model, domain, retrieval_method, top_k],
            [chatbot, source_context, context_evaluation]
        )
        clear_button.click(rag.clear_chat, [], [chatbot, source_context, context_evaluation], queue=False)

        demo.launch(share=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lanzar demo con base de datos")
    parser.add_argument("config_path", help="Path al fichero de configuracion")
    args = parser.parse_args()
    main(args.config_path)
