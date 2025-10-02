class Retriever:
    """A retriever class that handles document retrieval with optional reranking."""
    
    def __init__(self, vectorstore, top_k=5, reranker=None, initial_retrieve_count=None):
        """
        Initialize the document retriever.
        
        Args:
            vectorstore: The ElasticSearch vectorstore
            top_k: Number of documents to return after retrieval/reranking
            reranker: Optional reranker to use (BGEReranker instance)
            initial_retrieve_count: How many documents to retrieve initially (defaults to top_k if no reranker)
        """
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.reranker = reranker
        
        # If no initial_retrieve_count specified, use top_k or top_k*3 when reranking
        if initial_retrieve_count is None:
            self.initial_retrieve_count = top_k * 3 if reranker else top_k
        else:
            self.initial_retrieve_count = initial_retrieve_count
        
        # Create base retriever
        # self.base_retriever = vectorstore.as_retriever(
        #     search_kwargs={
        #         "k": self.initial_retrieve_count,
        #         "include_metadata": True}
        # )
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
            reranked_docs = self.reranker.rerank(query, retrieved_docs, top_k=self.top_k)
            final_reranked_docs = reranked_docs[:self.top_k]
        else:
            # If no reranker but we retrieved more than top_k, trim the results and add fake scores
            reranked_docs = [(doc, "N/A") for doc in retrieved_docs]
            final_reranked_docs = final_reranked_docs = [(doc, "N/A") for doc in retrieved_docs[:self.top_k]]
        return reranked_docs, final_reranked_docs