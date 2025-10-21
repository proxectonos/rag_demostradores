import requests

class Retriever:
    """A retriever class that handles document retrieval with optional reranking."""
    
    def __init__(self, search_url, es_user, es_password, search_func, reranker=None, num_docs_retrieval=10, num_docs_reranker=5):
        """
        Initialize the document retriever.
        
        Args:
            vectorstore: The ElasticSearch vectorstore
            reranker: Optional reranker to use (BGEReranker instance)
            num_docs_retrieval: Number of documents to retrieve initially
            num_docs_reranker: Number of documents to rerank
        """
        self.num_docs_retrieval = num_docs_retrieval
        self.num_docs_reranker = num_docs_reranker
        self.reranker = reranker
        self.search_url = search_url
        self.search_func = search_func
        self.es_user = es_user
        self.es_password = es_password

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
        retrieved_docs = self.search_documents(query)
        # Apply reranking if available
        if self.reranker:
            reranked_docs = self.reranker.rerank(query, retrieved_docs)
            final_reranked_docs = reranked_docs[:self.num_docs_reranker]
        else:
            # If no reranker but we retrieved more than top_k, trim the results and add fake scores
            reranked_docs = [(doc, "N/A") for doc in retrieved_docs]
            final_reranked_docs = final_reranked_docs = [(doc, "N/A") for doc in retrieved_docs[:self.num_docs_reranker]]
        return reranked_docs, final_reranked_docs
    
    def search_documents(self, query):
        """
        Search documents in ElasticSearch for the given query.
        
        Args:
            query: User query
        Returns:
            List of retrieved documents
        """
        try:
            # Make the request to ElasticSearch
            response = requests.post(
                self.search_url, 
                auth=(self.es_user, self.es_password), 
                json=self.search_func(query)
            )
            response.raise_for_status()
            
            # Parse the results
            hits = response.json().get('hits', {}).get('hits', [])
            documents = [hit['_source'] for hit in hits]
        except requests.exceptions.RequestException as e:
            print(f"Error during ElasticSearch request: {e}")
        print(f"Retrieved {len(documents)} documents from ElasticSearch.")
        return documents