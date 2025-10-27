import argparse
import gradio as gr
import re
import os
from backend.rag import RAG

class DummyRAG:
    def __init__(self, config_path=None):
        self.config = type('obj', (object,), {
            'retrieval_models': {"BM25": None, "BGE-M3": None},
            'reranker_models': {"None": None, "BGE-M3-Reranker": None},
            'generation_models': {"Salamandra-7B": None},
            'num_docs_retrieval': 10,
            'num_docs_reranker': 5
        })
        self.active_retriever_name = "BM25"
        self.active_reranker_name = "None"
        self.active_generator_name = "Salamandra-7B"
        
    def generate_response(self, chat_history, retriever, reranker, generator, num_retrieval, num_reranker):
        chat_history.append({
            "role": "assistant", 
            "content": f"Dummy response with retriever={retriever}, reranker={reranker}, generator={generator}, num_retrieval={num_retrieval}, num_reranker={num_reranker}"
        })
        source_text = ""
        context_text = "- [1] **Source Title 1** (Source=src1, Pos=1): This is the content of source 1.\n- [2] **Source Title 2** (Source=src2, Pos=2): This is the content of source 2."
        return chat_history, source_text, context_text
    
    def clear_chat(self):
        return [], "", ""
    
    def update_retriever(self, retriever_name):
        self.active_retriever_name = retriever_name
        return f"✓ Retriever cambiado a: {retriever_name}"
    
    def update_reranker(self, reranker_name):
        self.active_reranker_name = reranker_name
        return f"✓ Reranker cambiado a: {reranker_name}"
    
    def update_num_docs(self, num_retrieval, num_reranker):
        return f"✓ Documentos actualizados: {num_retrieval} retrieval, {num_reranker} reranker"


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
    chat_history.append({"role": "user", "content": user_message})
    return "", chat_history


def parse_sources(raw_text):
    """Parse sources from markdown-formatted text."""
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
    """Render sources as HTML details/summary elements."""
    sources = parse_sources(raw_text)
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


def update_retriever_handler(rag, retriever_name):
    """Handle retriever model change."""
    try:
        rag.switch_retriever(retriever_name)
        return f"✓ Retriever actualizado: {retriever_name}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def update_reranker_handler(rag, reranker_name):
    """Handle reranker model change."""
    try:
        rag.switch_reranker(reranker_name)
        return f"✓ Reranker actualizado: {reranker_name}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def update_num_docs_handler(rag, num_retrieval, num_reranker):
    """Handle number of documents change."""
    try:
        # Update the configuration
        rag.config.num_docs_retrieval = int(num_retrieval)
        rag.config.num_docs_reranker = int(num_reranker)
        
        # Reinitialize retriever with new settings
        #rag.retriever = rag._RAG__initialize_retriever()
        
        return f"✓ Documentos actualizados: {num_retrieval} retrieval, {num_reranker} reranker"
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ==================== GRADIO APP ====================
def gradio_app(rag):
    # Get available models from config
    retriever_choices = list(rag.config.retrieval_models.keys())
    reranker_choices = list(rag.config.reranker_models.keys())
    generator_choices = list(rag.config.generation_models.keys())
    
    with gr.Blocks(css_paths="interfaz/style.css", theme=gr.themes.Soft()) as demo:
        gr.HTML(f"<center id='title'><h1>💬 Demostrador RAG</h1></center>")
        
        with gr.Row(equal_height=True, elem_id="main_row"):
            # Main chat column
            with gr.Column(scale=5):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    type="messages",
                    avatar_images=None,
                    height="60vh"
                )
        
                with gr.Row(equal_height=True):
                    clear_button = gr.Button("🔄 Nuevo chat", scale=1)
                    msg = gr.Textbox(
                        placeholder="Escribe tu pregunta...",
                        lines=1,
                        show_label=False,
                        submit_btn=True,
                        stop_btn=True,
                        scale=5
                    )

                # Configuration controls
                gr.Markdown("### ⚙️ Configuración del Sistema")
                
                with gr.Row():
                    retriever_dropdown = gr.Dropdown(
                        choices=retriever_choices,
                        value=rag.active_retriever_name,
                        label="Retriever",
                        info="Modelo de recuperación"
                    )
                    reranker_dropdown = gr.Dropdown(
                        choices=reranker_choices,
                        value=rag.active_reranker_name,
                        label="Reranker",
                        info="Modelo de reordenación"
                    )
                    generator_dropdown = gr.Dropdown(
                        choices=generator_choices,
                        value=rag.active_generator_name,
                        label="Generator",
                        info="Modelo de generación",
                        interactive=False  # No podemos cambiar el generador en tiempo real fácilmente
                    )
                
                with gr.Row():
                    num_retrieval_slider = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=rag.config.num_docs_retrieval,
                        step=1,
                        label="Documentos Retrieval",
                        info="Número de documentos a recuperar inicialmente"
                    )
                    num_reranker_slider = gr.Slider(
                        minimum=1,
                        maximum=min(10, rag.config.num_docs_retrieval),
                        value=rag.config.num_docs_reranker,
                        step=1,
                        label="Documentos Reranker",
                        info="Número de documentos tras reordenación"
                    )
                
                # Status message for configuration changes
                config_status = gr.Markdown("", visible=False)

            # Sources column
            with gr.Column(scale=2):   
                with gr.Column(variant="panel", elem_id="sources_column"):
                    gr.Markdown("### 📚 Fuentes")
                    
                    # Current configuration display
                    with gr.Accordion("ℹ️ Configuración Actual", open=False):
                        current_config = gr.Markdown(
                            f"""
**Retriever:** {rag.active_retriever_name}
- Model: {rag.retriever_config.model_name}
- Index: {rag.retriever_config.elastic_index}

**Reranker:** {rag.active_reranker_name}
- Model: {rag.reranker_config.model_name}

**Generator:** {rag.active_generator_name}
- Model: {rag.generator_config.model_name}

**Docs Retrieval:** {rag.config.num_docs_retrieval}
**Docs Reranker:** {rag.config.num_docs_reranker}
                            """
                        )
                    
                    with gr.Accordion("📄 Documentos Recuperados", open=True):
                        source_context = gr.Markdown(show_label=False)
                        context_evaluation = gr.Markdown(
                            visible=False,
                            show_copy_button=True
                        )
                        sources_md = gr.HTML()
                        context_evaluation.change(
                            render_sources, 
                            inputs=context_evaluation, 
                            outputs=sources_md
                        )

        # Event handlers for configuration changes
        def update_config_display(retriever, reranker, generator, num_ret, num_rerank):
            """Update the configuration display."""
            retriever_cfg = rag.config.retrieval_models[retriever]
            reranker_cfg = rag.config.reranker_models[reranker]
            generator_cfg = rag.config.generation_models[generator]
            
            return f"""
**Retriever:** {retriever}
- Model: {retriever_cfg.model_name}
- Index: {retriever_cfg.elastic_index}

**Reranker:** {reranker}
- Model: {reranker_cfg.model_name}

**Generator:** {generator}
- Model: {generator_cfg.model_name}

**Docs Retrieval:** {int(num_ret)}
**Docs Reranker:** {int(num_rerank)}
            """

        # Handle retriever change
        retriever_dropdown.change(
            fn=lambda x: update_retriever_handler(rag, x),
            inputs=[retriever_dropdown],
            outputs=[config_status]
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=[config_status]
        ).then(
            fn=update_config_display,
            inputs=[retriever_dropdown, reranker_dropdown, generator_dropdown, 
                   num_retrieval_slider, num_reranker_slider],
            outputs=[current_config]
        )

        # Handle reranker change
        reranker_dropdown.change(
            fn=lambda x: update_reranker_handler(rag, x),
            inputs=[reranker_dropdown],
            outputs=[config_status]
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=[config_status]
        ).then(
            fn=update_config_display,
            inputs=[retriever_dropdown, reranker_dropdown, generator_dropdown, 
                   num_retrieval_slider, num_reranker_slider],
            outputs=[current_config]
        )

        # Handle number of documents change
        def handle_num_docs_change(num_ret, num_rerank):
            status = update_num_docs_handler(rag, num_ret, num_rerank)
            config_display = update_config_display(
                rag.active_retriever_name, 
                rag.active_reranker_name, 
                rag.active_generator_name,
                num_ret, 
                num_rerank
            )
            return status, config_display

        num_retrieval_slider.change(
            fn=handle_num_docs_change,
            inputs=[num_retrieval_slider, num_reranker_slider],
            outputs=[config_status, current_config]
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=[config_status]
        )

        num_reranker_slider.change(
            fn=handle_num_docs_change,
            inputs=[num_retrieval_slider, num_reranker_slider],
            outputs=[config_status, current_config]
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=[config_status]
        )

        # Chat interaction handlers
        msg.submit(
            user_input, 
            [msg, chatbot], 
            [msg, chatbot], 
            queue=False
        ).then(
            rag.generate_response,
            [chatbot, generator_dropdown, retriever_dropdown, reranker_dropdown, num_reranker_slider],
            [chatbot, source_context, context_evaluation]
        )
        
        clear_button.click(
            rag.clear_chat, 
            [], 
            [chatbot, source_context, context_evaluation], 
            queue=False
        )

        demo.launch(share=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lanzar demo con base de datos")
    parser.add_argument("config_path", help="Path al fichero de configuracion (general_config.json)")
    args = parser.parse_args()
    main(args.config_path)