from dataclasses import dataclass
from typing import Optional
import yaml
import os

@dataclass
class GeneralConfig:
    hf_cache_dir: str

@dataclass
class DatabaseConfig:
    elastic_index: str
    elastic_config_file: str
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None

@dataclass
class RetrieverConfig:
    retrieval_strategy: str
    initial_retrieve_count: int
    query_top_k: int
    embedding_model: Optional[str] = None

@dataclass
class RerankerConfig:
    use_reranking: bool
    reranker_model: Optional[str] = None

@dataclass
class ElasticConfig:
    username: str
    password: str
    url: str
    endpoint: str

@dataclass
class GeneratorConfig:
    model_name: str
    quantization: Optional[bool] = False

@dataclass
class Config:
    general_config: GeneralConfig
    database: DatabaseConfig
    retriever: RetrieverConfig
    reranker: RerankerConfig
    generator: GeneratorConfig
    

class ConfigLoader:
    @staticmethod
    def load(config_path) -> Config:
        with open(config_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        
        return Config(
            general_config=GeneralConfig(**config_dict['general_config']),
            database=DatabaseConfig(**config_dict['database']),
            retriever=RetrieverConfig(**config_dict['retriever']),
            reranker=RerankerConfig(**config_dict['reranker']),
            generator=GeneratorConfig(**config_dict['generator'])
        )
    @staticmethod
    def load_elastic(config_path) -> Config:
        with open(config_path, 'r') as file:
            config_dict = yaml.safe_load(file)

        return ElasticConfig(
            username=config_dict['username'],
            password=config_dict['password'],
            url=config_dict['elastic_url'],
            endpoint=config_dict['api_endpoint']
        )

if __name__ == "__main__":
    config_path = "config.yaml"
    config = ConfigLoader.load(config_path)
    print("General Config:", config.general_config)
    print("Database Config:", config.database)
    print("Retriever Config:", config.retriever)
    print("Reranker Config:", config.reranker)
    print("Generator Config:", config.generator)