from dataclasses import dataclass
from typing import Optional, Dict, Any
import yaml
import json
import os

@dataclass
class RetrievalModelConfig:
    embeddings: bool
    model_name: str
    elastic_index: str

@dataclass
class RerankerModelConfig:
    model_name: str

@dataclass
class GenerationModelConfig:
    openai_base_url: str
    openai_api_key: str
    model_name: str
    quantization: bool

@dataclass
class PromptsConfig:
    retrieval_system_prompt: str
    non_retrieval_system_prompt: str

@dataclass
class ElasticConfig:
    username: str
    password: str
    url: str
    endpoint: str

@dataclass
class Config:
    # Diccionarios con todos los modelos disponibles
    retrieval_models: Dict[str, RetrievalModelConfig]
    reranker_models: Dict[str, RerankerModelConfig]
    generation_models: Dict[str, GenerationModelConfig]
    
    # Modelos por defecto seleccionados
    default_retriever: str
    default_reranker: str
    default_generator: str
    
    # Configuraciones generales
    num_docs_retrieval: int
    num_docs_reranker: int
    elastic_config_path: str
    hf_cache_dir: str
    prompts: PromptsConfig
    
    # Configuración de Elasticsearch (cargada desde archivo)
    elastic_config: Optional[ElasticConfig] = None


class ConfigLoader:
    @staticmethod
    def load(config_path: str) -> Config:
        """
        Carga la configuración desde un archivo JSON.
        
        Args:
            config_path: Ruta al archivo general_config.json
            
        Returns:
            Config: Objeto de configuración completo
        """
        with open(config_path, 'r') as file:
            config_dict = json.load(file)
        
        # Parsear modelos de retrieval
        retrieval_models = {}
        for name, model_data in config_dict['RETRIEVAL_MODELS'].items():
            retrieval_models[name] = RetrievalModelConfig(
                embeddings=model_data['embeddings'].lower() == 'true',
                model_name=model_data['model_name'],
                elastic_index=model_data['elastic_index']
            )
        
        # Parsear modelos de reranking
        reranker_models = {}
        for name, model_data in config_dict['RERANKER_MODELS'].items():
            reranker_models[name] = RerankerModelConfig(
                model_name=model_data['model_name']
            )
        
        # Parsear modelos de generación
        generation_models = {}
        for name, model_data in config_dict['GENERATION_MODELS'].items():
            generation_models[name] = GenerationModelConfig(
                openai_base_url=model_data.get('openai_base_url', ''),
                openai_api_key=model_data.get('openai_api_key', ''),
                model_name=model_data['model_name'],
                quantization=model_data.get('quantization', 'false').lower() == 'true'
            )
        
        # Parsear prompts
        prompts = PromptsConfig(
            retrieval_system_prompt=config_dict['PROMPTS']['retrieval_system_prompt'],
            non_retrieval_system_prompt=config_dict['PROMPTS']['non_retrieval_system_prompt']
        )
        
        # Crear el objeto de configuración
        config = Config(
            retrieval_models=retrieval_models,
            reranker_models=reranker_models,
            generation_models=generation_models,
            default_retriever=config_dict['DEFAULT_RETRIEVER'],
            default_reranker=config_dict['DEFAULT_RERANKER'],
            default_generator=config_dict['DEFAULT_GENERATOR'],
            num_docs_retrieval=config_dict['NUM_DOCS_RETIEVAL'],  # Nota: mantiene el typo del JSON original
            num_docs_reranker=config_dict['NUM_DOCS_RERANKER'],
            elastic_config_path=config_dict['ELASTIC_CONFIG'],
            hf_cache_dir=config_dict['HF_CACHE_DIR'],
            prompts=prompts
        )
        
        # Cargar configuración de Elasticsearch si existe el archivo
        if os.path.exists(config.elastic_config_path):
            config.elastic_config = ConfigLoader.load_elastic(config.elastic_config_path)
        
        return config
    
    @staticmethod
    def load_elastic(config_path: str) -> ElasticConfig:
        """
        Carga la configuración de Elasticsearch desde un archivo YAML.
        
        Args:
            config_path: Ruta al archivo config_elastic.yaml
            
        Returns:
            ElasticConfig: Configuración de Elasticsearch
        """
        with open(config_path, 'r') as file:
            config_dict = yaml.safe_load(file)

        return ElasticConfig(
            username=config_dict['username'],
            password=config_dict['password'],
            url=config_dict['elastic_url'],
            endpoint=config_dict['api_endpoint']
        )


if __name__ == "__main__":
    # Ejemplo de uso
    config_path = "configs/general_config.json"
    config = ConfigLoader.load(config_path)
    
    print("=" * 50)
    print("CONFIGURACIÓN CARGADA")
    print("=" * 50)
    
    print(f"\nHF Cache Dir: {config.hf_cache_dir}")
    print(f"Elastic Config Path: {config.elastic_config_path}")
    print(f"Num Docs Retrieval: {config.num_docs_retrieval}")
    print(f"Num Docs Reranker: {config.num_docs_reranker}")
    
    print("\n" + "=" * 50)
    print("MODELOS DISPONIBLES")
    print("=" * 50)
    print("\nRetrievers:", list(config.retrieval_models.keys()))
    print("Rerankers:", list(config.reranker_models.keys()))
    print("Generators:", list(config.generation_models.keys()))

    

