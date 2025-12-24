import logging, requests, numpy as np, time
from typing import List, Union, Optional

logger = logging.getLogger("moonlight_rag_provider")

class RAGProviderException(Exception):
    """
    Custom exception for RAGProvider errors.
    """
    pass

class RAGProvider:
    """
    Connects to llama.cpp server to provide embedding and re-ranking capabilities.
    OPTIMIZED VERSION with connection pooling and batching.
    """
    
    def __init__(
        self,
        embedding_url: str = "http://host.docker.internal:8080",
        reranking_url: str = "http://host.docker.internal:9696"
    ):
        """
        Initialize the client with the provided URLs for embedding and reranking.
        
        Args:
            embedding_url (str): URL for the embedding service.
            reranking_url (str): URL for the reranking service.
        """
        
        self.embedding_url = embedding_url
        self.reranking_url = reranking_url
        
        self.max_length: int = 8192  # Max context length of BGE-M3 Model
        self.timeout: int = 120      # Timeout for requests in seconds
        
        # OPTIMIZATION: Connection pooling
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=20,
            pool_maxsize=20,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
    def _wait_for_server(
        self,
        max_retries: int = 5,
        delay: int = 2
    ):
        """
        Wait for the server to be ready by checking the health endpoint.
        
        Args:
            max_retries (int): Maximum number of retries to check server health.
            delay (int): Delay between retries in seconds.
        
        Raises:
            RAGProviderException: If the server is not ready after max retries.
        """

        for _ in range(max_retries):
            embedding_ready = self._check_health(
                url=self.embedding_url,
                service_name="Embedding Service"
            )
            
            reranking_ready = self._check_health(
                url=self.reranking_url,
                service_name="Reranking Service"
            )
            
            if embedding_ready and reranking_ready:
                logger.info("RAGProvider is ready.")
                return
            
            logger.warning(f"Servers are not ready. Embedding: {embedding_ready}, Reranking: {reranking_ready}. Retrying in {delay} seconds...")
            time.sleep(delay)
            
        raise RAGProviderException("RAGProvider is not ready after maximum retries. Check the service URLs or status.")
        
    
    def _check_health(
        self,
        url: str,
        service_name: str
    ) -> bool:
        """
        Check the health of the service by sending a GET request to the health endpoint.
        
        Args:
            url (str): URL of the service to check.
            service_name (str): Name of the service for logging purposes.
        
        Returns:
            bool: True if the service is healthy, False otherwise.
        """
        
        try:
            response = self.session.get(
                url=f"{url}/health",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("status") == "ok":
                    return True
                
            return False
        
        except Exception as e:
            logger.debug(f"Error checking health of {service_name}: {e}")
            return False
        
        
    def embed(
        self,
        texts: Union[str, List[str]]
    ) -> np.ndarray:
        """
        Generate embeddings for the provided texts.
        OPTIMIZED: Handles both single and batch embedding efficiently.
        
        Args:
            texts (Union[str, List[str]]): Text or list of texts to embed. (max 8192 tokens each)
            
        Returns:
            np.ndarray: Normalized embeddings as a numpy array.
        """
        
        # Convert single string input to list for consistent processing
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False

        # OPTIMIZATION: Batch all texts in one request
        payload = {
            "input": texts,
            "encoding_format": "float"
        }
        
        try:
            start_time = time.perf_counter()
            response = self.session.post(
                url=f"{self.embedding_url}/v1/embeddings",
                json=payload,
                timeout=self.timeout
            )
            logger.debug(f"Embedding request took {time.perf_counter() - start_time:.3f}s for {len(texts)} texts")
            
            response.raise_for_status()
            
            data = response.json()
            
            # Extract embeddings from response
            embeddings = []
            for item in data.get("data", []):
                embedding = item.get("embedding")
                if embedding:
                    embeddings.append(embedding)
                else:
                    logger.warning("Received empty embedding for one of the texts.")
                
            if not embeddings:
                raise RAGProviderException("No valid embeddings received from the service")
                
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Return single embedding if single input, otherwise return array
            if single_input:
                return embeddings_array[0] if len(embeddings_array) > 0 else np.array([])
            
            return embeddings_array
        
        except requests.exceptions.RequestException as e:
            raise RAGProviderException(f"Error during embedding request: {e}")
        except Exception as e:
            raise RAGProviderException(f"Unexpected error during embedding: {e}")
        
    def rerank(
        self,
        query: str,
        chunks: List[str],
        top_n: Optional[int] = None
    ) -> dict:
        """
        Reranks documents based on their relevance to the query.

        Args:
            query (str): The query to rerank chunks against.
            chunks (List[str]): List of chunks to be reranked.
            top_n (Optional[int]): Number of top chunks to return. If None, returns all.

        Returns:
            dict: Dictionary with 'scores' and 'chunks' keys containing reranked results.
        """
        
        if not chunks:
            return {
                "scores": [],
                "chunks": []
            }
        
        # Craft the payload for reranking
        payload = {
            "query": query,
            "documents": chunks,
        }
        
        # Add top_n to payload if specified
        if top_n is not None:
            payload["top_n"] = top_n
        
        # Send the request to the reranking service
        try:
            start_time = time.perf_counter()
            response = self.session.post(
                url=f"{self.reranking_url}/v1/rerank",
                json=payload,
                timeout=self.timeout
            )
            logger.debug(f"Reranking request took {time.perf_counter() - start_time:.3f}s for {len(chunks)} chunks")
            
            response.raise_for_status()
            
            data = response.json()
            
            # Extract scores and reordered chunks
            results = data.get("results", [])
            
            if not results:
                logger.warning("No results returned from reranking service")
                return {
                    "scores": [],
                    "chunks": []
                }
            
            # Sort by relevance score (descending)
            sorted_results = sorted(
                results,
                key=lambda x: x.get("relevance_score", 0),
                reverse=True
            )
            
            # Apply top_n filter if specified
            if top_n is not None:
                sorted_results = sorted_results[:top_n]
                
            scores = [item.get("relevance_score", 0) for item in sorted_results]
            sorted_chunks = [chunks[item["index"]] for item in sorted_results if item["index"] < len(chunks)]

            # Normalize scores
            normalized_scores = [self._sigmoid(score) for score in scores]
            
            return {
                "scores": normalized_scores,
                "chunks": sorted_chunks
            }
            
        except requests.exceptions.RequestException as e:
            raise RAGProviderException(f"Error during reranking request: {e}")
        except Exception as e:
            raise RAGProviderException(f"Unexpected error during reranking: {e}")
    
    def _sigmoid(self, x: float) -> float:
        """
        Apply sigmoid function to normalize score to [0,1].
        """
        try:
            return 1 / (1 + np.exp(-x))
        except (OverflowError, FloatingPointError):
            # Handle extreme values
            return 0.0 if x < 0 else 1.0
    
    def __del__(self):
        """Clean up session on destruction"""
        try:
            if hasattr(self, 'session'):
                self.session.close()
        except:
            pass