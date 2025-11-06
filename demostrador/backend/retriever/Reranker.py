from FlagEmbedding import FlagReranker
from typing import List, Any
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch

class SentenceTransformerReranker(SentenceTransformer):
    def __init__(self, model_name: str, cache_dir: str = None, use_fp16: bool = True):
        """
        Args:
            model_name: Name of the SentenceTransformer model to use
            use_fp16: Whether to use half-precision for inference
            normalize: Whether to normalize scores to 0-1 range using sigmoid
        """
        super().__init__(model_name, cache_folder=cache_dir, device='cuda' if use_fp16 else 'cpu')

    def compute_scores(self, query: str, passages: List[str], normalize) -> List[float]:
        # Encode query and passages
        query_embedding = self.encode(query, convert_to_tensor=True)
        passage_embeddings = self.encode(passages, convert_to_tensor=True)
        
        # Compute cosine similarities
        scores = (query_embedding @ passage_embeddings.T).cpu().numpy().tolist()
        
        if normalize:
            scores = [1 / (1 + np.exp(-score)) for score in scores]  # Sigmoid normalization
        return scores

class FlagEmbeddingReranker(FlagReranker):
    def __init__(self, model_name: str, cache_dir: str = None, use_fp16: bool = True):
        """
        Args:
            model_name: Name of the FlagEmbedding model to use
            cache_dir: Directory to cache the model files
            use_fp16: Whether to use half-precision for inference
        """
        super().__init__(model_name, cache_dir=cache_dir, use_fp16=use_fp16)

    def compute_scores(self, query: str, passages: List[str], normalize) -> List[float]:
        # Create pairs of [query, passage] for each passage
        pairs = [[query, passage] for passage in passages]
        scores = self.compute_score(pairs, normalize=normalize) #Using FlagEmbedding's compute_score method
        return scores

class Qwen3Reranker:
    def __init__(self, model_name: str, cache_dir: str = None, use_fp16: bool = True):
        #Official code adapted from https://github.com/QwenLM/Qwen3-Embedding
        """
        Args:
            model_name: Name of the Qwen3 model to use
            cache_dir: Directory to cache the model files
            use_fp16: Whether to use half-precision for inference
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype='auto', device_map='auto').eval()
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = 8192
        
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
        
        self.task = 'Given a web search query, retrieve relevant passages that answer the query'

    def format_instruction(self, instruction, query, doc):
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
        return output

    def process_inputs(self, pairs):
        inputs = self.tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs

    @torch.no_grad()
    def compute_logits(self, inputs, **kwargs):
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def compute_scores(self, query: str, passages: List[str], normalize) -> List[float]:
        pairs = [self.format_instruction(self.task, query, passage) for passage in passages]
        inputs = self.process_inputs(pairs)
        scores = self.compute_logits(inputs)
        if normalize:
            scores = [1 / (1 + np.exp(-score)) for score in scores]  # Sigmoid normalization
        return scores

class JinaReranker:
    def __init__(self, model_name: str, cache_dir: str = None, use_fp16: bool = True):
        self.model = AutoModel.from_pretrained(
            model_name,
            dtype="auto",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        self.model.to('cuda' if use_fp16 else 'cpu')
        self.model.eval()
    
    def compute_scores(self, query: str, passages: List[str], normalize) -> List[float]:
        results = self.model.rerank(query, passages) #Jina model's has its own rerank method
        return results

class Reranker:
    def __init__(self, model_name, hf_cache_dir, use_fp16=True, normalize=True):
        """
        Args:
            model_name: Name of the reranker model to use
            use_fp16: Whether to use half-precision for inference
            normalize: Whether to normalize scores to 0-1 range using sigmoid
        """
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.normalize = normalize
        if model_name in ["BAAI/bge-reranker-v2-m3"]:
            # Use FlagEmbedding for BGE reranker
            self.reranker = FlagEmbeddingReranker(model_name, cache_dir=hf_cache_dir, use_fp16=self.use_fp16)
        elif model_name.startswith("Qwen"):
            # Use Qwen3 reranker
            self.reranker = Qwen3Reranker(model_name, cache_dir=hf_cache_dir, use_fp16=self.use_fp16)
        elif model_name.startswith("jina"):
            # Use Jina reranker
            self.reranker = JinaReranker(model_name, cache_dir=hf_cache_dir, use_fp16=self.use_fp16)
        else:
            # Use SentenceTransformer for other models
            self.reranker = SentenceTransformerReranker(model_name, cache_dir=hf_cache_dir, use_fp16=self.use_fp16)
        
    def compute_scores(self, query: str, passages: List[str]) -> List[float]:
        # Create pairs of [query, passage] for each passage
        scores = self.reranker.compute_scores(query, passages, normalize=self.normalize)
        return scores
    
    def rerank(self, query: str, docs: List[Any]) -> List[Any]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: User query
            docs: List of documents to rerank
            
        Returns:
            List of reranked documents
        """
        # Extract text content from documents
        passages = [doc.get("text") for doc in docs]
        scores = self.compute_scores(query, passages)
        
        scored_docs = []
        # JinaReranker special handling. Jina already returns sorted results, with original indices
        if self.reranker .__class__.__name__ == "JinaReranker":
            for result in scores:
                original_doc = docs[result['index']]
                scored_docs.append((original_doc, float(result['relevance_score'])))
        # Create (doc, score) pairs and sort by score in descending order
        else:
            scored_docs = list(zip(docs, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs