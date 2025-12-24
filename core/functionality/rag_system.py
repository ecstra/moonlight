######################################################################################
#  Need to make this async                                                           #
######################################################################################

import os, uuid, logging, time, asyncio

from hashlib import md5
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from ...src.helpers import log_warning
from ..processors import FileProcessor
from ..providers.rag_provider import RAGProvider
from ..processors.chunk_processor import DocumentChunker

logger = logging.getLogger("moonlight_rag")

QDRANT_URL = "http://host.docker.internal:6333"

class RAGException(Exception):
    pass

class RAGResult(BaseModel):
    """
    Structured RAG query result
    """
    question: str
    matches: List[Dict[str, Any]]
    total_matches: int
    sources_used: int
    success: bool
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class RAGSystem:
    def __init__(
        self, 
        namespace: str = "", 
        max_sources: int = 3,
        chunk_min_size: int = 800,
        chunk_max_size: int = 1200,
        top_k: int = 15,
        top_n_rerank: int = 5,
        metadata: Optional[Dict[str, Any]] = None
    ):

        if not namespace:
            log_warning("No namespace provided. Using default namespace of 'temporary'.")
            self.namespace = "temporary"
        else:
            self.namespace = namespace
        
        self.max_sources = max_sources
        self.top_k = top_k
        self.top_n_rerank = top_n_rerank
        self.metadata = metadata or {}
        
        # Initialize components
        self.client = self._init_qdrant_client()
        self.rag_provider = RAGProvider()
        self.chunker = DocumentChunker(
            min_size=chunk_min_size,
            max_size=chunk_max_size,
        )
        
        self.text_embedding_dim = 1024 # BGE-M3 Dimension, change if using a different model
        
        # Set dynamic collection names
        self.text_collection = f"{self.namespace}_text"
        
        self._ensure_collections_exist()

    def _init_qdrant_client(self) -> QdrantClient:
        for attempt in range(3):
            try:
                return QdrantClient(QDRANT_URL, timeout=10.0)
            except Exception as e:
                if attempt == 2:
                    raise
                logger.warning(f"Failed to connect to Qdrant (attempt {attempt + 1}): {e}")
                time.sleep(2 ** attempt)

    def _ensure_collections_exist(self):
        try:
            # Text collection for documents and temporary text
            if not self.client.collection_exists(self.text_collection):
                logger.info(f"Creating collection '{self.text_collection}' with dimension {self.text_embedding_dim}")
                self.client.create_collection(
                    collection_name=self.text_collection,
                    vectors_config=models.VectorParams(
                        size=self.text_embedding_dim,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info("Collection created successfully")
            else:
                # Verify dimension of existing collection
                collection_info = self.client.get_collection(self.text_collection)
                existing_dim = collection_info.config.params.vectors.size
                if existing_dim != self.text_embedding_dim:
                    raise RAGException(
                        f"Collection dimension mismatch: expected {self.text_embedding_dim}, got {existing_dim}"
                    )
        except Exception as e:
            raise RAGException(f"Failed to initialize collections: {str(e)}")

    def _get_file_hash(
            self, 
            file_path: str
        ) -> str:
        """
        Generate unique hash for file versioning
        """
        with open(file_path, 'rb') as f:
            return md5(f.read()).hexdigest()

    def clear_rag(self) -> None:
        """
        Delete all collections associated with this named RAG instance.
        """
        try:
            if self.client.collection_exists(self.text_collection):
                self.client.delete_collection(self.text_collection)
        except Exception as e:
            logger.error(f"Failed to clear RAG collections: {e}")
            pass

    def add_text(
            self, 
            text: str, 
            metadata: Optional[Dict[str, Any]] = None
        ) -> None:
        
        """
        Add text to RAG system using the new chunker - OPTIMIZED with batch embedding
        """
        
        self._ensure_collections_exist()
        
        # Use the new chunker
        try:
            chunk_result = self.chunker.chunk(text)
            chunks = chunk_result['chunks']
            logger.info(f"Generated {len(chunks)} chunks using DocumentChunker")
        except Exception as e:
            raise RAGException(f"Chunking failed: {e}")
        
        if not chunks:
            raise RAGException("No chunks generated from text.")
        
        # OPTIMIZATION: Batch embed all chunks at once
        try:
            start_time = time.perf_counter()
            embeddings = self.rag_provider.embed(chunks)  # Batch embedding
            logger.debug(f"Batch embedding took {time.perf_counter() - start_time:.3f}s for {len(chunks)} chunks")
            
            # Ensure embeddings is 2D
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
                
        except Exception as e:
            raise RAGException(f"Batch embedding failed: {e}")
        
        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            try:
                if len(embedding) != self.text_embedding_dim:
                    logger.error(f"Embedding dimension mismatch: got {len(embedding)}, expected {self.text_embedding_dim}")
                    continue
                    
                payload = {
                    "content": chunk,
                    "source_type": "text",
                    "chunk_index": idx,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Add custom metadata if provided
                if metadata:
                    payload.update(metadata)
                
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"text_{idx}_{datetime.now().timestamp()}"))
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload=payload
                    )
                )
            except Exception as e:
                logger.error(f"Failed to process chunk {idx}: {e}")
                continue
        
        if not points:
            raise RAGException("No valid points generated from text")
            
        try:
            self.client.upsert(
                collection_name=self.text_collection,
                points=points
            )
            logger.info(f"Successfully added {len(points)} text chunks to RAG")
        except (ResponseHandlingException, UnexpectedResponse) as e:
            logger.error(f"Failed to add text to RAG: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error adding text to RAG: {e}")
            raise
    
    def add_document(
            self, 
            file_path: str, 
            metadata: Optional[Dict[str, Any]] = None
        ) -> None:
        """
        Add document to RAG system using the new chunker - OPTIMIZED with batch embedding
        """
        self._ensure_collections_exist()
        file_path = str(Path(file_path).resolve())
        logger.info(f"Adding document: {file_path}")
        
        file_hash = self._get_file_hash(file_path)
        last_modified = os.path.getmtime(file_path)
        
        # Check if document already exists
        existing = self.client.scroll(
            collection_name=self.text_collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_type",
                        match=models.MatchValue(value="document")
                    ),
                    models.FieldCondition(
                        key="file_path",
                        match=models.MatchValue(value=file_path)
                    ),
                    models.FieldCondition(
                        key="file_hash",
                        match=models.MatchValue(value=file_hash)
                    )
                ]
            )
        )
        
        if len(existing[0]) > 0:
            logger.info("Document already exists with same hash, skipping")
            return

        # Process document
        try:
            text = FileProcessor([file_path]).content[file_path]
        except Exception as e:
            raise RAGException(f"Error processing file: {str(e)}")

        # Use the new chunker
        try:
            chunk_result = self.chunker.chunk(text)
            chunks = chunk_result['chunks']
            logger.info(f"Generated {len(chunks)} chunks using DocumentChunker")
            logger.info(f"Chunk statistics: {chunk_result['statistics']}")
        except Exception as e:
            raise RAGException(f"Chunking failed: {e}")

        if not chunks:
            raise RAGException("No chunks generated from document")

        # OPTIMIZATION: Batch embed all chunks at once
        try:
            start_time = time.perf_counter()
            embeddings = self.rag_provider.embed(chunks)  # Batch embedding
            logger.debug(f"Batch embedding took {time.perf_counter() - start_time:.3f}s for {len(chunks)} chunks")
            
            # Ensure embeddings is 2D
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
                
        except Exception as e:
            raise RAGException(f"Batch embedding failed: {e}")

        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            try:
                if len(embedding) != self.text_embedding_dim:
                    logger.error(f"Chunk {idx}: Embedding dimension mismatch - got {len(embedding)}, expected {self.text_embedding_dim}")
                    continue
                
                # Create payload
                payload = {
                    "content": chunk,
                    "source_type": "document",
                    "file_path": file_path,
                    "file_hash": file_hash,
                    "last_modified": last_modified,
                    "chunk_index": idx,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Add custom metadata if provided
                if metadata:
                    payload.update(metadata)
                
                # Generate a valid UUID using uuid5 from file hash & chunk index
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{file_hash}_{idx}"))
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload=payload
                    )
                )
                    
            except Exception as e:
                logger.error(f"Failed to process chunk {idx}: {e}")
                continue

        if not points:
            raise RAGException("No valid points generated from document")

        try:
            logger.info(f"Upserting {len(points)} points to collection {self.text_collection}")
            self.client.upsert(
                collection_name=self.text_collection,
                points=points
            )
            logger.info("Document added successfully")
        except Exception as e:
            logger.error(f"Failed to add document to RAG: {e}")
            raise RAGException(f"Failed to add document to RAG: {e}")

    def query(
            self, 
            question: str
        ) -> RAGResult:
        """
        OPTIMIZED query using vector search with Qdrant similarity scores directly
        """
        try:
            start_time = time.perf_counter()
            
            # Get query embedding
            query_embedding = self.rag_provider.embed(question)
            if query_embedding.ndim > 1:
                query_embedding = query_embedding.reshape(-1)
            
            embed_time = time.perf_counter() - start_time
            logger.debug(f"Query embedding took {embed_time:.3f}s")
            
            # Vector search to get candidate documents
            search_start = time.perf_counter()
            results = self.client.search(
                collection_name=self.text_collection,
                query_vector=query_embedding.tolist(),
                limit=self.top_k,
                score_threshold=0.0,
                with_payload=True,
                with_vectors=False  # OPTIMIZATION: Don't fetch vectors unless needed
            )
            search_time = time.perf_counter() - search_start
            logger.debug(f"Vector search took {search_time:.3f}s")

            if not results:
                return RAGResult(
                    question=question,
                    matches=[],
                    total_matches=0,
                    sources_used=0,
                    success=True,
                    metadata={"message": "No relevant information found."}
                )

            # Extract documents and metadata
            documents = []
            metadata_list = []
            similarity_scores = []
            
            for result in results:
                payload = result.payload
                if not payload:
                    continue
                    
                chunk_text = payload["content"]
                documents.append(chunk_text)
                
                # OPTIMIZATION: Use Qdrant's similarity score directly
                similarity_scores.append(float(result.score))
                
                source = Path(payload.get("file_path", "temporary")).name if payload.get("file_path") else "temporary"
                metadata_list.append({
                    "source": source,
                    "chunk_index": payload.get("chunk_index", 0),
                    "source_type": payload.get("source_type", "unknown"),
                    "timestamp": payload.get("timestamp"),
                    "custom_metadata": {k: v for k, v in payload.items() 
                                     if k not in ["content", "source_type", "chunk_index", "file_path", "file_hash", "last_modified", "timestamp"]}
                })

            if not documents:
                return RAGResult(
                    question=question,
                    matches=[],
                    total_matches=0,
                    sources_used=0,
                    success=True,
                    metadata={"message": "No valid documents found for processing."}
                )
            
            # Get top candidates for reranking based on Qdrant scores
            top_indices = list(range(min(self.top_n_rerank, len(documents))))
            top_documents = [documents[i] for i in top_indices]
            top_metadata = [metadata_list[i] for i in top_indices]
            top_similarities = [similarity_scores[i] for i in top_indices]
            
            # Second pass: Rerank documents using the provider
            rerank_start = time.perf_counter()
            try:
                reranked = self.rag_provider.rerank(
                    query=question, 
                    chunks=top_documents, 
                    top_n=self.max_sources
                )
                rerank_time = time.perf_counter() - rerank_start
                logger.debug(f"Reranking took {rerank_time:.3f}s")
                
                # Format results using reranked order
                matches = []
                for i, (score, doc) in enumerate(zip(reranked["scores"], reranked["chunks"])):
                    if len(matches) >= self.max_sources:
                        break
                    
                    # Find the original metadata for this document
                    doc_idx = top_documents.index(doc)
                    meta = top_metadata[doc_idx]
                    original_similarity = top_similarities[doc_idx]
                    
                    matches.append({
                        "content": doc,
                        "rerank_score": float(score),
                        "similarity": original_similarity,
                        "source": meta["source"],
                        "chunk_index": meta["chunk_index"],
                        "source_type": meta["source_type"],
                        "timestamp": meta["timestamp"],
                        "custom_metadata": meta["custom_metadata"]
                    })

            except Exception as rerank_error:
                logger.warning(f"Reranking failed, using similarity scores: {rerank_error}")
                # Fallback to similarity-based ranking
                matches = []
                for i in range(min(self.max_sources, len(top_indices))):
                    idx = top_indices[i]
                    matches.append({
                        "content": documents[idx],
                        "rerank_score": None,
                        "similarity": similarity_scores[idx],
                        "source": metadata_list[idx]["source"],
                        "chunk_index": metadata_list[idx]["chunk_index"],
                        "source_type": metadata_list[idx]["source_type"],
                        "timestamp": metadata_list[idx]["timestamp"],
                        "custom_metadata": metadata_list[idx]["custom_metadata"]
                    })

            total_time = time.perf_counter() - start_time
            logger.debug(f"Total query time: {total_time:.3f}s")

            return RAGResult(
                question=question,
                matches=matches,
                total_matches=len(results),
                sources_used=len(matches),
                success=True,
                metadata={
                    "namespace": self.namespace,
                    "embedding_dimension": self.text_embedding_dim,
                    "top_k": self.top_k,
                    "reranking_used": 'reranked' in locals(),
                    "timing": {
                        "embedding": embed_time,
                        "search": search_time,
                        "rerank": rerank_time if 'rerank_time' in locals() else 0,
                        "total": total_time
                    }
                }
            )

        except Exception as e:
            logger.error(f"Query failed: {str(e)}", exc_info=True)
            return RAGResult(
                question=question,
                matches=[],
                total_matches=0,
                sources_used=0,
                success=False,
                error_message=str(e)
            )

    @staticmethod
    def get_all_namespaces() -> list:
        """Return namespaces whose '_text' collection exists and has data."""
        client = QdrantClient(QDRANT_URL)
        collections = client.get_collections().collections
        namespaces = set()
        for coll in collections:
            name = coll.name
            if name.endswith("_text"):
                coll_info = client.get_collection(name)
                if getattr(coll_info, "points_count", 0) > 0:
                    namespaces.add(name[:-5])
        return list(namespaces)

    @staticmethod
    def clear_all_namespaces() -> None:
        """Clear all namespaces"""
        log_warning("All namespaces will be deleted, please proceed with caution.")
        client = QdrantClient(QDRANT_URL)
        all_namespaces = RAGSystem.get_all_namespaces()
        for ns in all_namespaces:
            text_coll = f"{ns}_text"
            if client.collection_exists(text_coll):
                client.delete_collection(text_coll)
    
    def __del__(self):
        try:
            if hasattr(self, 'client'):
                self.client.close()
        except:
            pass