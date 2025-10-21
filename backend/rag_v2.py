from sentence_transformers import SentenceTransformer
import torch
from backend.ConfigLoader_v2 import ConfigLoader
from backend.retriever.Reranker import Reranker
from backend.retriever.Retriever_v2 import Retriever
from backend.llm_handler_v2 import LLMHandler
from typing import Dict
from enum import Enum
import pprint

class RAG:
    def __init__(self, config_file, retriever_name=None, reranker_name=None, generator_name=None):
        """
        Initialize the RAG system.
        
        Args:
            config_file: Path to general_config.json
            retriever_name: Optional retriever name to override default
            reranker_name: Optional reranker name to override default
            generator_name: Optional generator name to override default
        """
        print("Loading configuration...")
        self.config = ConfigLoader.load(config_file)
        
        # Override defaults if provided
        self.active_retriever_name = retriever_name or self.config.default_retriever
        self.active_reranker_name = reranker_name or self.config.default_reranker
        self.active_generator_name = generator_name or self.config.default_generator
        
        # Get active model configurations
        self.retriever_config = self.config.retrieval_models[self.active_retriever_name]
        self.reranker_config = self.config.reranker_models[self.active_reranker_name]
        self.generator_config = self.config.generation_models[self.active_generator_name]
        
        # Initialize retriever
        print(f"Initializing retriever: {self.active_retriever_name}...")
        self.retriever = self.__initialize_retriever()
        
        # Initialize LLM
        print(f"Initializing LLM: {self.active_generator_name}...")
        self.llm = LLMHandler(self.config, self.generator_config)
        self.messages = self.initialize_conversation()
        
        print("RAG system initialized successfully.")
    
    def __initialize_retriever(self):
        """Initialize the retriever based on active configuration"""
        
        # Determine if we need embeddings
        if self.retriever_config.embeddings and self.retriever_config.model_name:
            print(f"Loading embedding model: {self.retriever_config.model_name}...")
            embedding_model = SentenceTransformer(
                self.retriever_config.model_name, 
                cache_folder=self.config.hf_cache_dir
            )
            embedding_model.to('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            embedding_model = None

        # Define query functions
        def bm25_query(search_query: str) -> Dict:
            return {
                "query": {
                    "match": {
                        "text": search_query,
                    },
                },
            }
        
        def vector_query(search_query: str) -> Dict:
            vector = embedding_model.encode(search_query).tolist()
            return {
                "knn": {
                    "field": "text_embedding",
                    "query_vector": vector,
                }
            }
        
        # # Connect to Elasticsearch
        # es_client = Elasticsearch(
        #     hosts=[self.config.elastic_config.endpoint],
        #     basic_auth=(self.config.elastic_config.username, self.config.elastic_config.password)
        # )
        
        # # Create the vectorstore retriever
        # vectorstore_retriever = ElasticsearchRetriever(
        #     es_client=es_client,
        #     index_name=self.retriever_config.elastic_index,
        #     content_field="text",
        #     body_func=bm25_query if not self.retriever_config.embeddings else vector_query,
        # )
        
        search_url = F"{self.config.elastic_config.endpoint}/{self.retriever_config.elastic_index}/_search?size={self.config.num_docs_retrieval}"
        # Initialize reranker if needed
        reranker = None
        if self.reranker_config.model_name:  # Si model_name no está vacío
            print(f"Loading reranker: {self.reranker_config.model_name}...")
            reranker = Reranker(
                model_name = self.reranker_config.model_name,
                hf_cache_dir = self.config.hf_cache_dir,
                use_fp16 = True,
                normalize = True
            )

        return Retriever(
            search_url = search_url,
            search_func = bm25_query if not self.retriever_config.embeddings else vector_query,
            es_user = self.config.elastic_config.username,
            es_password = self.config.elastic_config.password,
            reranker = reranker,
            num_docs_retrieval = self.config.num_docs_retrieval,
            num_docs_reranker = self.config.num_docs_reranker
        )

    def retrieve_contexts(self, user_query: str):        
        # Retrieve relevant documents
        initial_docs, final_docs = self.retriever.invoke(user_query)
        
        # Format context from retrieved and reranked documents
        context = "\n\n".join([f"Documento {i+1}: {doc.text}" for i, (doc,_) in enumerate(final_docs)])
        
        # Store source information
        source_info = []
        initial_docs_info = []
        
        for i, (doc, score) in enumerate(final_docs):
            source_data = {
                "id": i+1,
                "score": score,
                "content": doc.page_content,
                "metadata": doc.metadata if hasattr(doc, "metadata") else {}
            }
            source_info.append(source_data)
        
        for i, (doc, score) in enumerate(initial_docs):
            source_data = {
                "id": i+1,
                "score": score,
                "content": doc.page_content,
                "metadata": doc.metadata if hasattr(doc, "metadata") else {}
            }
            initial_docs_info.append(source_data)

        return source_info, initial_docs_info
    
    def generate_response(self, chat_history, model, domain, retrieval_method, top_k):
        
        # Get the latest user message
        user_message = chat_history[-1]['content']
        
        # Retrieve relevant documents
        _, sources = self.retrieve_contexts(user_message)
        
        # Format the context for the response
        context = "\n\n".join([f"Documento {src['id']}: {src['content']}" for src in sources[:top_k]])

        # Generate response
        prompt = self.llm.default_system_prompt_pre + context + self.llm.default_system_prompt_post + user_message
        
        use_chat_memory = False
        if use_chat_memory: 
            self.messages.append({"role": "user", "content": prompt})
        else:
            self.messages = [
                {"role": "system", "content": self.llm.default_system_prompt},
                {"role": "user", "content": prompt},
            ]

        # Fix: ensure alternating roles to prevent TemplateError
        if len(self.messages) >= 2 and self.messages[-1]["role"] == self.messages[-2]["role"]:
            print("[Warning] Fixing consecutive roles:", self.messages[-2]["role"])
            self.messages.pop(-2)

        # Generate response
        if self.llm.dummy:
            response_content = f"[DUMMY RESPONSE] Model disabled. Messages: {user_message[:100]}..."
        else:
            response_content = self.llm.generate(self.messages)

        # Append assistant response to maintain alternation
        self.messages.append({"role": "assistant", "content": response_content})
        
        # Append the assistant's response to the chat history
        chat_history.append({"role": "assistant", "content": response_content})
        
        # Prepare source text and context text for display
        source_text = ""
        context_text = ""
        for src in sources[:top_k]:
            meta = src["metadata"].get("_source", {})
            source_id = meta.get("source_id", f"Praza-{meta.get('published_on')}")
            title = meta.get("title", meta.get("headline", "Untitled"))
            position = meta.get("relative_chunk_id", "-1")
            content = src["content"].replace("\n", " ")
            context_text += f"- [{src['id']}] **{title}** (Source={source_id}, Pos={position}): {content}\n"
        
        return chat_history, source_text, context_text
    
    def clear_chat(self):
        return [], "", ""

    def initialize_conversation(self):
        messages = [
            {
                "role": "system",
                "content": self.config.prompts.retrieval_system_prompt
            }
        ]
        return messages
    
    def switch_retriever(self, retriever_name: str):
        """Switch to a different retriever model"""
        if retriever_name not in self.config.retrieval_models:
            raise ValueError(f"Retriever '{retriever_name}' not found in configuration")
        
        print(f"Switching retriever from {self.active_retriever_name} to {retriever_name}...")
        self.active_retriever_name = retriever_name
        self.retriever_config = self.config.retrieval_models[retriever_name]
        self.retriever = self.__initialize_retriever()
        print("Retriever switched successfully.")
    
    def switch_reranker(self, reranker_name: str):
        """Switch to a different reranker model"""
        if reranker_name not in self.config.reranker_models:
            raise ValueError(f"Reranker '{reranker_name}' not found in configuration")
        
        print(f"Switching reranker from {self.active_reranker_name} to {reranker_name}...")
        self.active_reranker_name = reranker_name
        self.reranker_config = self.config.reranker_models[reranker_name]
        self.retriever = self.__initialize_retriever()
        print("Reranker switched successfully.")


if __name__ == "__main__":
    # Print the configuration for debugging
    pp = pprint.PrettyPrinter(indent=4)
    
    # Initialize RAG with new config structure
    rag = RAG("general_config.json")
    
    print("\n" + "=" * 60)
    print("CONFIGURACIÓN ACTIVA")
    print("=" * 60)
    print(f"Retriever: {rag.active_retriever_name}")
    print(f"  - Model: {rag.retriever_config.model_name}")
    print(f"  - Index: {rag.retriever_config.elastic_index}")
    print(f"Reranker: {rag.active_reranker_name}")
    print(f"  - Model: {rag.reranker_config.model_name}")
    print(f"Generator: {rag.active_generator_name}")
    print(f"  - Model: {rag.generator_config.model_name}")
    print("=" * 60)
    
    print("\nAsistente RAG en Galego (escriba 'sair' para rematar)")
    print("Comandos especiales:")
    print("  - 'cambiar retriever <nombre>' - Cambiar modelo de retrieval")
    print("  - 'cambiar reranker <nombre>' - Cambiar modelo de reranking")
    print("  - 'modelos' - Ver modelos disponibles")
    
    while True:
        user_input = input("\nUsuario: ")
        
        if user_input.lower() in ["sair", "quit", "exit"]:
            print("Grazas por usar o asistente!")
            break
        
        # Handle special commands
        if user_input.lower().startswith("cambiar retriever "):
            retriever_name = user_input[18:].strip()
            try:
                rag.switch_retriever(retriever_name)
            except ValueError as e:
                print(f"Error: {e}")
            continue
        
        if user_input.lower().startswith("cambiar reranker "):
            reranker_name = user_input[17:].strip()
            try:
                rag.switch_reranker(reranker_name)
            except ValueError as e:
                print(f"Error: {e}")
            continue
        
        if user_input.lower() == "modelos":
            print("\nModelos disponibles:")
            print("Retrievers:", list(rag.config.retrieval_models.keys()))
            print("Rerankers:", list(rag.config.reranker_models.keys()))
            print("Generators:", list(rag.config.generation_models.keys()))
            continue
            
        # Get response from RAG system
        sources, initial_docs_info = rag.retrieve_contexts(user_input)
        
        print_sources = True
        if print_sources:
            print("\n--- Fragmentos empregados ---")
            for source in sources:
                source_id = source["id"]
                content = source["content"]
                metadata = source["metadata"]["_source"]
                source_file = metadata.get("source_id", f"Praza-{metadata.get('published_on')}")
                
                print(f"\nFragmento {source_id} - {source_file}")
                print("-" * 40)
                print(content)
                print("-" * 40)