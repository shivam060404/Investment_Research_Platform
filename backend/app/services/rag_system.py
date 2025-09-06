from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import asyncio
import numpy as np
from loguru import logger
from dataclasses import dataclass
from enum import Enum
import json
import re
from pathlib import Path

# Vector database imports
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None
    logger.warning("ChromaDB not available, using mock implementation")

try:
    import pinecone
except ImportError:
    pinecone = None
    logger.warning("Pinecone not available, using mock implementation")

# Embedding and NLP imports
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    logger.warning("SentenceTransformers not available, using mock embeddings")

try:
    import faiss
except ImportError:
    faiss = None
    logger.warning("FAISS not available, using alternative similarity search")

from ..core.config import settings

class DocumentType(Enum):
    """Types of financial documents"""
    SEC_FILING = "sec_filing"
    EARNINGS_REPORT = "earnings_report"
    RESEARCH_REPORT = "research_report"
    NEWS_ARTICLE = "news_article"
    ECONOMIC_DATA = "economic_data"
    MARKET_DATA = "market_data"
    COMPANY_PROFILE = "company_profile"

class RetrievalMethod(Enum):
    """Retrieval methods for RAG system"""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    TEMPORAL = "temporal"
    ENTITY_BASED = "entity_based"

@dataclass
class Document:
    """Represents a document in the RAG system"""
    id: str
    content: str
    title: str
    doc_type: DocumentType
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    timestamp: datetime = None
    entities: List[str] = None
    keywords: List[str] = None

@dataclass
class RetrievalResult:
    """Result from document retrieval"""
    document: Document
    score: float
    relevance_explanation: str
    retrieval_method: RetrievalMethod

class DocumentProcessor:
    """Processes and chunks documents for RAG system"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.financial_entities = [
            "revenue", "profit", "earnings", "ebitda", "cash flow", "debt",
            "assets", "liabilities", "equity", "dividend", "share price",
            "market cap", "pe ratio", "roe", "roa", "gross margin"
        ]
    
    def chunk_document(self, content: str) -> List[str]:
        """Split document into overlapping chunks"""
        if len(content) <= self.chunk_size:
            return [content]
        
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(content):
                # Look for sentence endings near the chunk boundary
                for i in range(end, max(start + self.chunk_size - 100, start), -1):
                    if content[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = content[start:end]
            chunks.append(chunk.strip())
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= len(content):
                break
        
        return chunks
    
    def extract_entities(self, content: str) -> List[str]:
        """Extract financial entities from content"""
        entities = []
        content_lower = content.lower()
        
        for entity in self.financial_entities:
            if entity in content_lower:
                entities.append(entity)
        
        # Extract ticker symbols (simple pattern)
        ticker_pattern = r'\b[A-Z]{1,5}\b'
        tickers = re.findall(ticker_pattern, content)
        entities.extend([t for t in tickers if len(t) <= 5])
        
        return list(set(entities))
    
    def extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content"""
        # Simple keyword extraction
        financial_keywords = [
            "growth", "decline", "increase", "decrease", "bullish", "bearish",
            "buy", "sell", "hold", "upgrade", "downgrade", "target", "forecast",
            "guidance", "outlook", "risk", "opportunity", "merger", "acquisition"
        ]
        
        keywords = []
        content_lower = content.lower()
        
        for keyword in financial_keywords:
            if keyword in content_lower:
                keywords.append(keyword)
        
        return keywords
    
    def process_document(self, doc: Document) -> List[Document]:
        """Process document into chunks with metadata"""
        chunks = self.chunk_document(doc.content)
        processed_docs = []
        
        for i, chunk in enumerate(chunks):
            chunk_doc = Document(
                id=f"{doc.id}_chunk_{i}",
                content=chunk,
                title=f"{doc.title} (Part {i+1})",
                doc_type=doc.doc_type,
                metadata={
                    **doc.metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "parent_doc_id": doc.id
                },
                timestamp=doc.timestamp,
                entities=self.extract_entities(chunk),
                keywords=self.extract_keywords(chunk)
            )
            processed_docs.append(chunk_doc)
        
        return processed_docs

class EmbeddingService:
    """Service for generating document embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.embedding_dim = 384  # Default for MiniLM
    
    async def initialize(self):
        """Initialize embedding model"""
        try:
            if SentenceTransformer:
                self.model = SentenceTransformer(self.model_name)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                logger.info(f"Initialized embedding model: {self.model_name}")
            else:
                logger.warning("Using mock embeddings - install sentence-transformers for real embeddings")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            self.model = None
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        if self.model:
            return self.model.encode(text)
        else:
            # Mock embedding for testing
            return np.random.normal(0, 1, self.embedding_dim)
    
    def generate_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for batch of texts"""
        if self.model:
            return self.model.encode(texts)
        else:
            # Mock embeddings for testing
            return np.random.normal(0, 1, (len(texts), self.embedding_dim))

class VectorStore:
    """Abstract vector store interface"""
    
    def __init__(self, collection_name: str = "financial_documents"):
        self.collection_name = collection_name
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize vector store"""
        raise NotImplementedError
    
    async def add_documents(self, documents: List[Document]):
        """Add documents to vector store"""
        raise NotImplementedError
    
    async def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[Document, float]]:
        """Search for similar documents"""
        raise NotImplementedError
    
    async def delete_document(self, doc_id: str):
        """Delete document from vector store"""
        raise NotImplementedError

class ChromaVectorStore(VectorStore):
    """ChromaDB vector store implementation"""
    
    def __init__(self, collection_name: str = "financial_documents"):
        super().__init__(collection_name)
        self.client = None
        self.collection = None
    
    async def initialize(self):
        """Initialize ChromaDB"""
        try:
            if chromadb:
                self.client = chromadb.Client(Settings(
                    persist_directory=settings.CHROMA_PERSIST_DIRECTORY
                ))
                
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"description": "Financial documents for RAG system"}
                )
                
                self.is_initialized = True
                logger.info(f"Initialized ChromaDB collection: {self.collection_name}")
            else:
                logger.warning("ChromaDB not available, using mock vector store")
                self.is_initialized = True
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
    
    async def add_documents(self, documents: List[Document]):
        """Add documents to ChromaDB"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            if self.collection:
                ids = [doc.id for doc in documents]
                embeddings = [doc.embedding.tolist() for doc in documents if doc.embedding is not None]
                metadatas = [{
                    "title": doc.title,
                    "doc_type": doc.doc_type.value,
                    "timestamp": doc.timestamp.isoformat() if doc.timestamp else None,
                    **doc.metadata
                } for doc in documents]
                documents_content = [doc.content for doc in documents]
                
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents_content
                )
                
                logger.info(f"Added {len(documents)} documents to ChromaDB")
            else:
                logger.info(f"Mock: Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
    
    async def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[Document, float]]:
        """Search ChromaDB for similar documents"""
        try:
            if self.collection:
                results = self.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=top_k
                )
                
                documents_with_scores = []
                for i in range(len(results['ids'][0])):
                    doc = Document(
                        id=results['ids'][0][i],
                        content=results['documents'][0][i],
                        title=results['metadatas'][0][i].get('title', ''),
                        doc_type=DocumentType(results['metadatas'][0][i].get('doc_type', 'research_report')),
                        metadata=results['metadatas'][0][i]
                    )
                    score = 1 - results['distances'][0][i]  # Convert distance to similarity
                    documents_with_scores.append((doc, score))
                
                return documents_with_scores
            else:
                # Mock search results
                mock_docs = []
                for i in range(min(top_k, 3)):
                    doc = Document(
                        id=f"mock_doc_{i}",
                        content=f"Mock financial document content {i}",
                        title=f"Mock Document {i}",
                        doc_type=DocumentType.RESEARCH_REPORT,
                        metadata={"source": "mock"}
                    )
                    score = 0.9 - (i * 0.1)
                    mock_docs.append((doc, score))
                return mock_docs
                
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            return []
    
    async def delete_document(self, doc_id: str):
        """Delete document from ChromaDB"""
        try:
            if self.collection:
                self.collection.delete(ids=[doc_id])
                logger.info(f"Deleted document {doc_id} from ChromaDB")
        except Exception as e:
            logger.error(f"Error deleting document from ChromaDB: {e}")

class ReRanker:
    """Re-ranks retrieved documents based on relevance"""
    
    def __init__(self):
        self.financial_terms_weight = 1.5
        self.recency_weight = 1.2
        self.entity_match_weight = 2.0
    
    def calculate_relevance_score(self, doc: Document, query: str, 
                                query_entities: List[str] = None) -> float:
        """Calculate relevance score for document"""
        base_score = 1.0
        
        # Financial terms boost
        financial_terms = ["revenue", "profit", "earnings", "growth", "market"]
        query_lower = query.lower()
        content_lower = doc.content.lower()
        
        financial_matches = sum(1 for term in financial_terms 
                              if term in query_lower and term in content_lower)
        financial_boost = financial_matches * 0.1 * self.financial_terms_weight
        
        # Entity matching boost
        entity_boost = 0
        if query_entities and doc.entities:
            entity_matches = len(set(query_entities) & set(doc.entities))
            entity_boost = entity_matches * 0.2 * self.entity_match_weight
        
        # Recency boost
        recency_boost = 0
        if doc.timestamp:
            days_old = (datetime.now() - doc.timestamp).days
            if days_old < 30:  # Recent documents get boost
                recency_boost = (30 - days_old) / 30 * 0.1 * self.recency_weight
        
        # Document type relevance
        type_boost = {
            DocumentType.SEC_FILING: 0.2,
            DocumentType.EARNINGS_REPORT: 0.15,
            DocumentType.RESEARCH_REPORT: 0.1,
            DocumentType.NEWS_ARTICLE: 0.05
        }.get(doc.doc_type, 0)
        
        total_score = base_score + financial_boost + entity_boost + recency_boost + type_boost
        return min(total_score, 2.0)  # Cap at 2.0
    
    def rerank_documents(self, documents_with_scores: List[Tuple[Document, float]], 
                        query: str, query_entities: List[str] = None) -> List[RetrievalResult]:
        """Re-rank documents based on multiple factors"""
        reranked_results = []
        
        for doc, original_score in documents_with_scores:
            relevance_score = self.calculate_relevance_score(doc, query, query_entities)
            final_score = original_score * relevance_score
            
            # Generate explanation
            explanation = self._generate_explanation(doc, query, relevance_score, original_score)
            
            result = RetrievalResult(
                document=doc,
                score=final_score,
                relevance_explanation=explanation,
                retrieval_method=RetrievalMethod.HYBRID
            )
            reranked_results.append(result)
        
        # Sort by final score
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        return reranked_results
    
    def _generate_explanation(self, doc: Document, query: str, 
                            relevance_score: float, original_score: float) -> str:
        """Generate explanation for ranking"""
        explanations = []
        
        if relevance_score > 1.2:
            explanations.append("High relevance due to financial content match")
        
        if doc.entities:
            explanations.append(f"Contains {len(doc.entities)} relevant entities")
        
        if doc.timestamp and (datetime.now() - doc.timestamp).days < 30:
            explanations.append("Recent document (recency boost)")
        
        if doc.doc_type in [DocumentType.SEC_FILING, DocumentType.EARNINGS_REPORT]:
            explanations.append("High-quality financial document type")
        
        return "; ".join(explanations) if explanations else "Standard relevance matching"

class QueryProcessor:
    """Processes and expands queries for better retrieval"""
    
    def __init__(self):
        self.financial_synonyms = {
            "profit": ["earnings", "income", "net income"],
            "revenue": ["sales", "turnover", "income"],
            "growth": ["increase", "expansion", "rise"],
            "decline": ["decrease", "fall", "drop"]
        }
    
    def expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms"""
        expanded_terms = [query]
        query_lower = query.lower()
        
        for term, synonyms in self.financial_synonyms.items():
            if term in query_lower:
                expanded_terms.extend(synonyms)
        
        return " ".join(expanded_terms)
    
    def extract_query_entities(self, query: str) -> List[str]:
        """Extract entities from query"""
        # Simple entity extraction
        entities = []
        
        # Extract ticker symbols
        ticker_pattern = r'\b[A-Z]{1,5}\b'
        tickers = re.findall(ticker_pattern, query)
        entities.extend(tickers)
        
        # Extract financial terms
        financial_terms = ["revenue", "profit", "earnings", "growth", "debt"]
        query_lower = query.lower()
        
        for term in financial_terms:
            if term in query_lower:
                entities.append(term)
        
        return entities

class RAGSystem:
    """Main RAG system orchestrator"""
    
    def __init__(self):
        self.document_processor = DocumentProcessor(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        self.embedding_service = EmbeddingService()
        self.vector_store = ChromaVectorStore()
        self.reranker = ReRanker()
        self.query_processor = QueryProcessor()
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize RAG system components"""
        try:
            logger.info("Initializing RAG system")
            
            await self.embedding_service.initialize()
            await self.vector_store.initialize()
            
            self.is_initialized = True
            logger.info("RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            raise
    
    async def add_document(self, doc: Document):
        """Add document to RAG system"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Process document into chunks
            processed_docs = self.document_processor.process_document(doc)
            
            # Generate embeddings
            for processed_doc in processed_docs:
                embedding = self.embedding_service.generate_embedding(processed_doc.content)
                processed_doc.embedding = embedding
            
            # Add to vector store
            await self.vector_store.add_documents(processed_docs)
            
            logger.info(f"Added document {doc.id} with {len(processed_docs)} chunks")
            
        except Exception as e:
            logger.error(f"Error adding document to RAG system: {e}")
    
    async def add_documents_batch(self, documents: List[Document]):
        """Add multiple documents in batch"""
        for doc in documents:
            await self.add_document(doc)
    
    async def retrieve(self, query: str, top_k: int = 10, 
                     retrieval_method: RetrievalMethod = RetrievalMethod.HYBRID) -> List[RetrievalResult]:
        """Retrieve relevant documents for query"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Process query
            expanded_query = self.query_processor.expand_query(query)
            query_entities = self.query_processor.extract_query_entities(query)
            
            # Generate query embedding
            query_embedding = self.embedding_service.generate_embedding(expanded_query)
            
            # Retrieve documents
            documents_with_scores = await self.vector_store.search(query_embedding, top_k * 2)
            
            # Re-rank documents
            reranked_results = self.reranker.rerank_documents(
                documents_with_scores, query, query_entities
            )
            
            # Return top results
            return reranked_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    async def generate_context(self, query: str, max_context_length: int = 4000) -> str:
        """Generate context for query from retrieved documents"""
        try:
            # Retrieve relevant documents
            results = await self.retrieve(query, top_k=settings.TOP_K_RETRIEVAL)
            
            if not results:
                return "No relevant documents found."
            
            # Build context from top results
            context_parts = []
            current_length = 0
            
            for result in results:
                doc_content = f"Source: {result.document.title}\n{result.document.content}\n"
                
                if current_length + len(doc_content) > max_context_length:
                    # Truncate if needed
                    remaining_space = max_context_length - current_length
                    if remaining_space > 100:  # Only add if meaningful space left
                        doc_content = doc_content[:remaining_space] + "..."
                        context_parts.append(doc_content)
                    break
                
                context_parts.append(doc_content)
                current_length += len(doc_content)
            
            context = "\n\n".join(context_parts)
            
            # Add metadata about sources
            source_info = f"\n\nContext generated from {len(context_parts)} relevant documents."
            
            return context + source_info
            
        except Exception as e:
            logger.error(f"Error generating context: {e}")
            return "Error generating context from documents."
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        try:
            # In a real implementation, get actual stats from vector store
            stats = {
                "is_initialized": self.is_initialized,
                "vector_store_type": type(self.vector_store).__name__,
                "embedding_model": self.embedding_service.model_name,
                "embedding_dimension": self.embedding_service.embedding_dim,
                "chunk_size": self.document_processor.chunk_size,
                "chunk_overlap": self.document_processor.chunk_overlap,
                "timestamp": datetime.now().isoformat()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {"error": str(e)}

# Factory function to create RAG system
def create_rag_system() -> RAGSystem:
    """Create and return configured RAG system"""
    return RAGSystem()

# Sample documents for testing
def create_sample_documents() -> List[Document]:
    """Create sample financial documents for testing"""
    documents = [
        Document(
            id="aapl_q4_2024",
            content="Apple Inc. reported record Q4 2024 revenue of $95.3 billion, up 8% year-over-year. iPhone revenue was $46.2 billion, Services revenue reached $22.3 billion. The company's gross margin improved to 46.2% from 43.3% in the prior year. Apple's cash position remains strong at $162.1 billion.",
            title="Apple Q4 2024 Earnings Report",
            doc_type=DocumentType.EARNINGS_REPORT,
            metadata={"company": "AAPL", "quarter": "Q4", "year": 2024},
            timestamp=datetime(2024, 11, 1)
        ),
        Document(
            id="tsla_analysis_2024",
            content="Tesla's vehicle deliveries in Q3 2024 reached 462,890 units, exceeding analyst expectations of 455,000. The company's energy storage deployments grew 40% year-over-year. Tesla's automotive gross margin excluding regulatory credits was 16.9%. The company maintains its target of 50% average annual growth in vehicle deliveries.",
            title="Tesla Q3 2024 Delivery Analysis",
            doc_type=DocumentType.RESEARCH_REPORT,
            metadata={"company": "TSLA", "quarter": "Q3", "year": 2024},
            timestamp=datetime(2024, 10, 15)
        ),
        Document(
            id="fed_rate_decision_2024",
            content="The Federal Reserve decided to maintain the federal funds rate at 5.25%-5.50% range in its November 2024 meeting. The Fed cited ongoing concerns about inflation, which remains above the 2% target at 3.2%. Economic growth continues to show resilience with GDP expanding at 2.8% annualized rate in Q3.",
            title="Federal Reserve Rate Decision November 2024",
            doc_type=DocumentType.ECONOMIC_DATA,
            metadata={"source": "Federal Reserve", "type": "monetary_policy"},
            timestamp=datetime(2024, 11, 7)
        )
    ]
    
    return documents