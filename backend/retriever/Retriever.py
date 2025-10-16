class Retriever:
    """A retriever class that handles document retrieval with optional reranking."""
    
    def __init__(self, vectorstore, reranker=None, num_docs_retrieval=10, num_docs_reranker=5):
        """
        Initialize the document retriever.
        
        Args:
            vectorstore: The ElasticSearch vectorstore
            reranker: Optional reranker to use (BGEReranker instance)
            num_docs_retrieval: Number of documents to retrieve initially
            num_docs_reranker: Number of documents to rerank
        """
        self.vectorstore = vectorstore
        self.num_docs_retrieval = num_docs_retrieval
        self.num_docs_reranker = num_docs_reranker
        self.reranker = reranker
        self.base_retriever = vectorstore

    def invoke(self, query):
        """
        Retrieve documents for the given query with optional reranking.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (initial_docs, final_docs)
                - retrieved_docs: List of documents retrieved before reranking
                - reranked_docs: List of documents after reranking (or top_k if no reranker)
        """
        # Get initial results
        retrieved_docs = self.base_retriever.invoke(query)
        # Apply reranking if available
        if self.reranker:
            reranked_docs = self.reranker.rerank(query, retrieved_docs)
            final_reranked_docs = reranked_docs[:self.num_docs_reranker]
        else:
            # If no reranker but we retrieved more than top_k, trim the results and add fake scores
            reranked_docs = [(doc, "N/A") for doc in retrieved_docs]
            final_reranked_docs = final_reranked_docs = [(doc, "N/A") for doc in retrieved_docs[:self.num_docs_reranker]]
        return reranked_docs, final_reranked_docs