#!/usr/bin/env python3
"""
Enhanced Retrieval-Augmented Generation (RAG) Pipeline
Advanced document retrieval, embedding, and context generation for educational AI
Integrated with IntelliLearn AI's open-source model ecosystem
"""

import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Any, Optional, Union
import json
import time
import logging
from dataclasses import dataclass, field
import requests
from config import Config
import hashlib
import pickle
from pathlib import Path
import faiss
import re
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Enhanced imports for open-source integration
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    from bertopic import BERTopic
    import chromadb
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    ENHANCED_RAG_AVAILABLE = True
except ImportError:
    ENHANCED_RAG_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class EnhancedDocument:
    """Enhanced document chunk with comprehensive metadata and features"""
    content: str
    source: str
    doc_type: str  # 'topic', 'concept', 'example', 'analysis', 'chapter', 'keypoint'
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    
    # Enhanced attributes
    chunk_id: str = field(default_factory=lambda: str(time.time()))
    difficulty_level: str = "intermediate"
    educational_tags: List[str] = field(default_factory=list)
    confidence_score: float = 1.0
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    user_ratings: List[float] = field(default_factory=list)
    
    # Content analysis
    word_count: int = 0
    readability_score: float = 0.0
    key_concepts: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize computed fields"""
        self.word_count = len(self.content.split())
        self.readability_score = self._calculate_readability()
        self.key_concepts = self._extract_key_concepts()
    
    def _calculate_readability(self) -> float:
        """Calculate readability score using Flesch Reading Ease"""
        try:
            import textstat
            return textstat.flesch_reading_ease(self.content)
        except ImportError:
            # Simple approximation based on sentence and word length
            sentences = self.content.count('.') + self.content.count('!') + self.content.count('?')
            words = len(self.content.split())
            if sentences == 0:
                return 50.0
            avg_sentence_length = words / sentences
            return max(0, min(100, 100 - (avg_sentence_length * 1.5)))
    
    def _extract_key_concepts(self) -> List[str]:
        """Extract key concepts from content"""
        # Simple keyword extraction - can be enhanced with NLP models
        common_words = {'the', 'is', 'are', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
        words = re.findall(r'\b[A-Z][a-z]+\b', self.content)  # Capitalized words
        concepts = [word for word in words if word.lower() not in common_words]
        return list(set(concepts))[:5]  # Top 5 unique concepts
    
    def update_access_stats(self):
        """Update access statistics"""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def add_rating(self, rating: float):
        """Add user rating (0-5 scale)"""
        if 0 <= rating <= 5:
            self.user_ratings.append(rating)
    
    @property
    def average_rating(self) -> float:
        """Get average user rating"""
        return np.mean(self.user_ratings) if self.user_ratings else 0.0

class AdvancedRAGPipeline:
    """
    Advanced RAG pipeline with educational optimization and open-source models
    Features: FAISS indexing, topic modeling, difficulty-aware retrieval, caching
    """
    
    def __init__(self, 
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 use_faiss: bool = True,
                 cache_dir: str = "cache/rag",
                 difficulty_level: str = "intermediate"):
        
        self.config = Config()
        self.embedding_model_name = embedding_model_name
        self.embedding_model = self._load_embedding_model(embedding_model_name)
        self.difficulty_level = difficulty_level
        
        # Document storage
        self.documents: List[EnhancedDocument] = []
        self.document_index: Dict[str, EnhancedDocument] = {}
        
        # Indexing
        self.use_faiss = use_faiss and ENHANCED_RAG_AVAILABLE
        self.faiss_index = None
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.is_indexed = False
        
        # Caching
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.query_cache: Dict[str, Tuple[List[Tuple[str, float]], datetime]] = {}
        self.cache_ttl = timedelta(hours=1)
        
        # Enhanced features
        self.topic_model = None
        self.text_splitter = None
        self.vector_store = None
        self.feedback_scores: Dict[str, Dict[str, float]] = {}
        
        # Performance tracking
        self.retrieval_stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'avg_retrieval_time': 0.0,
            'feedback_count': 0
        }
        
        # Initialize enhanced components
        self._initialize_enhanced_components()
    
    def _initialize_enhanced_components(self):
        """Initialize enhanced RAG components"""
        if ENHANCED_RAG_AVAILABLE:
            try:
                # Initialize text splitter for better chunking
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50,
                    separators=["\n\n", "\n", ". ", "! ", "? ", " "]
                )
                
                # Initialize topic modeling
                self.topic_model = BERTopic(
                    embedding_model=self.embedding_model,
                    verbose=False,
                    calculate_probabilities=True
                )
                
                # Initialize vector store if available
                try:
                    client = chromadb.Client()
                    self.vector_store = client.create_collection(
                        name="intellilearn_knowledge",
                        get_or_create=True
                    )
                except Exception as e:
                    logger.warning(f"ChromaDB not available: {e}")
                
                logger.info("✅ Enhanced RAG components initialized")
                
            except Exception as e:
                logger.warning(f"Some enhanced features not available: {e}")
    
    @st.cache_resource
    def _load_embedding_model(_self, model_name: str):
        """Load and cache the embedding model with error handling"""
        try:
            model = SentenceTransformer(model_name)
            logger.info(f"✅ Loaded embedding model: {model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to load {model_name}, falling back to default")
            return SentenceTransformer('all-MiniLM-L6-v2')
    
    def ingest_knowledge_base_enhanced(self, knowledge_base: Dict[str, Any]) -> int:
        """
        Enhanced knowledge base ingestion with intelligent chunking and metadata
        """
        self.documents = []
        self.document_index = {}
        doc_count = 0
        
        try:
            # Process chapters with enhanced chunking
            for chapter_name, chapter_data in knowledge_base.get('chapters', {}).items():
                doc_count += self._process_chapter(chapter_name, chapter_data)
            
            # Process standalone topics
            for topic_name, topic_data in knowledge_base.get('topics', {}).items():
                doc_count += self._process_topic(topic_name, topic_data)
            
            # Process concepts with educational context
            for i, concept in enumerate(knowledge_base.get('concepts', [])):
                doc = self._create_concept_document(concept, i)
                self._add_document(doc)
                doc_count += 1
            
            # Process examples with categorization
            for i, example in enumerate(knowledge_base.get('examples', [])):
                doc = self._create_example_document(example, i)
                self._add_document(doc)
                doc_count += 1
            
            # Process AI analysis with enhanced metadata
            if knowledge_base.get('grok_analysis'):
                doc = self._create_analysis_document(knowledge_base['grok_analysis'])
                self._add_document(doc)
                doc_count += 1
            
            # Run topic modeling on all documents
            if self.topic_model and len(self.documents) > 5:
                self._run_topic_modeling()
            
            logger.info(f"✅ Enhanced ingestion: {doc_count} documents processed")
            return doc_count
            
        except Exception as e:
            logger.error(f"❌ Enhanced ingestion failed: {e}")
            return 0
    
    def _process_chapter(self, chapter_name: str, chapter_data: Dict) -> int:
        """Process a chapter with intelligent chunking"""
        doc_count = 0
        
        # Main chapter content with chunking
        content = chapter_data.get('content', '')
        if content and self.text_splitter:
            chunks = self.text_splitter.split_text(content)
            
            for i, chunk in enumerate(chunks):
                doc = EnhancedDocument(
                    content=f"Chapter: {chapter_name}\n\n{chunk}",
                    source=f"chapter_{chapter_name}_chunk_{i}",
                    doc_type="chapter",
                    difficulty_level=chapter_data.get('difficulty', 'intermediate').lower(),
                    metadata={
                        "chapter_name": chapter_name,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "difficulty": chapter_data.get('difficulty', 'intermediate'),
                        "estimated_time": chapter_data.get('estimated_reading_time', 0),
                        "word_count": chapter_data.get('word_count', 0)
                    }
                )
                
                # Add educational tags
                doc.educational_tags = self._extract_educational_tags(chunk, chapter_data)
                
                self._add_document(doc)
                doc_count += 1
        
        # Process topics within chapter
        for topic_name, topic_data in chapter_data.get('topics', {}).items():
            doc_count += self._process_topic(topic_name, topic_data, chapter_name)
        
        return doc_count
    
    def _process_topic(self, topic_name: str, topic_data: Dict, chapter_name: str = None) -> int:
        """Process a topic with comprehensive metadata"""
        doc_count = 0
        
        # Main topic content
        if topic_data.get('content'):
            doc = EnhancedDocument(
                content=f"Topic: {topic_name}\n\n{topic_data['content']}",
                source=f"topic_{topic_name}",
                doc_type="topic",
                difficulty_level=topic_data.get('difficulty', 'intermediate').lower(),
                metadata={
                    "topic_name": topic_name,
                    "chapter_name": chapter_name,
                    "difficulty": topic_data.get('difficulty', 'intermediate'),
                    "estimated_time": topic_data.get('estimated_time', 0),
                    "prerequisites": topic_data.get('prerequisites', [])
                }
            )
            
            # Enhanced educational tags
            doc.educational_tags = self._extract_educational_tags(topic_data['content'], topic_data)
            
            self._add_document(doc)
            doc_count += 1
        
        # Key points as focused documents
        for i, point in enumerate(topic_data.get('key_points', [])):
            doc = EnhancedDocument(
                content=f"Key Point about {topic_name}: {point}",
                source=f"keypoint_{topic_name}_{i}",
                doc_type="keypoint",
                difficulty_level=topic_data.get('difficulty', 'intermediate').lower(),
                metadata={
                    "topic_name": topic_name,
                    "chapter_name": chapter_name,
                    "point_index": i,
                    "difficulty": topic_data.get('difficulty', 'intermediate')
                }
            )
            
            self._add_document(doc)
            doc_count += 1
        
        # Topic summary
        if topic_data.get('summary'):
            doc = EnhancedDocument(
                content=f"Summary of {topic_name}: {topic_data['summary']}",
                source=f"summary_{topic_name}",
                doc_type="summary",
                difficulty_level=topic_data.get('difficulty', 'intermediate').lower(),
                metadata={
                    "topic_name": topic_name,
                    "chapter_name": chapter_name,
                    "difficulty": topic_data.get('difficulty', 'intermediate')
                }
            )
            
            self._add_document(doc)
            doc_count += 1
        
        return doc_count
    
    def _create_concept_document(self, concept: str, index: int) -> EnhancedDocument:
        """Create an enhanced concept document"""
        return EnhancedDocument(
            content=f"Important Concept: {concept}",
            source=f"concept_{index}",
            doc_type="concept",
            difficulty_level=self._infer_difficulty_from_content(concept),
            metadata={
                "concept_index": index,
                "concept_type": self._classify_concept(concept)
            },
            educational_tags=["concept", "definition", "important"]
        )
    
    def _create_example_document(self, example: str, index: int) -> EnhancedDocument:
        """Create an enhanced example document"""
        return EnhancedDocument(
            content=f"Example: {example}",
            source=f"example_{index}",
            doc_type="example",
            difficulty_level=self._infer_difficulty_from_content(example),
            metadata={
                "example_index": index,
                "example_type": self._classify_example(example)
            },
            educational_tags=["example", "practical", "application"]
        )
    
    def _create_analysis_document(self, analysis: str) -> EnhancedDocument:
        """Create an enhanced AI analysis document"""
        return EnhancedDocument(
            content=f"AI Content Analysis: {analysis}",
            source="ai_analysis",
            doc_type="analysis",
            difficulty_level="advanced",
            metadata={
                "analysis_type": "comprehensive",
                "generated_by": "AI"
            },
            educational_tags=["analysis", "insights", "overview"]
        )
    
    def _extract_educational_tags(self, content: str, metadata: Dict = None) -> List[str]:
        """Extract educational tags from content"""
        tags = []
        content_lower = content.lower()
        
        # Content type tags
        if any(word in content_lower for word in ['definition', 'define', 'meaning']):
            tags.append('definition')
        if any(word in content_lower for word in ['example', 'instance', 'case']):
            tags.append('example')
        if any(word in content_lower for word in ['formula', 'equation', 'calculate']):
            tags.append('mathematical')
        if any(word in content_lower for word in ['process', 'step', 'procedure']):
            tags.append('procedural')
        if any(word in content_lower for word in ['theory', 'principle', 'concept']):
            tags.append('theoretical')
        
        # Difficulty indicators
        if metadata:
            difficulty = metadata.get('difficulty', '').lower()
            if difficulty:
                tags.append(f'difficulty_{difficulty}')
        
        return tags
    
    def _infer_difficulty_from_content(self, content: str) -> str:
        """Infer difficulty level from content characteristics"""
        word_count = len(content.split())
        
        # Simple heuristics - can be enhanced with ML models
        if word_count < 20:
            return "beginner"
        elif word_count > 100:
            return "advanced"
        
        # Check for complexity indicators
        complex_words = ['analysis', 'synthesis', 'evaluation', 'optimization', 'methodology']
        if any(word in content.lower() for word in complex_words):
            return "advanced"
        
        return "intermediate"
    
    def _classify_concept(self, concept: str) -> str:
        """Classify concept type"""
        concept_lower = concept.lower()
        
        if any(word in concept_lower for word in ['principle', 'law', 'theory']):
            return "theoretical"
        elif any(word in concept_lower for word in ['method', 'technique', 'process']):
            return "procedural"
        elif any(word in concept_lower for word in ['tool', 'instrument', 'device']):
            return "practical"
        else:
            return "general"
    
    def _classify_example(self, example: str) -> str:
        """Classify example type"""
        example_lower = example.lower()
        
        if any(word in example_lower for word in ['real-world', 'industry', 'application']):
            return "real_world"
        elif any(word in example_lower for word in ['exercise', 'problem', 'practice']):
            return "practice"
        elif any(word in example_lower for word in ['case study', 'scenario']):
            return "case_study"
        else:
            return "general"
    
    def _add_document(self, doc: EnhancedDocument):
        """Add document to storage and index"""
        self.documents.append(doc)
        self.document_index[doc.source] = doc
    
    def _run_topic_modeling(self):
        """Run topic modeling on documents for better organization"""
        if not self.topic_model:
            return
        
        try:
            # Prepare documents for topic modeling
            docs = [doc.content for doc in self.documents if len(doc.content) > 50]
            
            if len(docs) < 5:
                return
            
            # Fit topic model
            topics, probabilities = self.topic_model.fit_transform(docs)
            
            # Assign topics to documents
            for i, doc in enumerate(self.documents):
                if i < len(topics):
                    doc.metadata['topic_id'] = int(topics[i])
                    doc.metadata['topic_probability'] = float(probabilities[i].max()) if probabilities is not None else 0.0
            
            logger.info(f"✅ Topic modeling completed: {len(set(topics))} topics identified")
            
        except Exception as e:
            logger.warning(f"Topic modeling failed: {e}")
    
    def build_advanced_index(self) -> bool:
        """Build advanced index with FAISS for better performance"""
        if not self.documents:
            logger.warning("No documents to index")
            return False
        
        try:
            # Generate embeddings
            texts = [doc.content for doc in self.documents]
            embeddings = self._generate_embeddings_batch(texts)
            
            if embeddings is None:
                return False
            
            # Store embeddings in documents
            for i, doc in enumerate(self.documents):
                doc.embedding = embeddings[i]
            
            # Build FAISS index if available
            if self.use_faiss and ENHANCED_RAG_AVAILABLE:
                self._build_faiss_index(embeddings)
            else:
                self.embeddings_matrix = embeddings
            
            # Store in vector database if available
            if self.vector_store:
                self._store_in_vector_db()
            
            self.is_indexed = True
            logger.info(f"✅ Advanced index built: {len(self.documents)} documents")
            
            # Cache the index
            self._save_index_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Index building failed: {e}")
            return False
    
    def _generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> Optional[np.ndarray]:
        """Generate embeddings in batches for better performance"""
        try:
            if not texts:
                return None
            
            embeddings = []
            
            with st.spinner(f"Generating embeddings for {len(texts)} documents..."):
                progress_bar = st.progress(0)
                
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    batch_embeddings = self.embedding_model.encode(
                        batch,
                        show_progress_bar=False,
                        batch_size=min(batch_size, len(batch))
                    )
                    embeddings.append(batch_embeddings)
                    
                    # Update progress
                    progress = (i + len(batch)) / len(texts)
                    progress_bar.progress(progress)
                
                progress_bar.progress(1.0)
            
            return np.vstack(embeddings)
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None
    
    def _build_faiss_index(self, embeddings: np.ndarray):
        """Build FAISS index for fast similarity search"""
        try:
            import faiss
            
            # Normalize embeddings for better cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            
            # Use IndexFlatIP for exact cosine similarity
            self.faiss_index = faiss.IndexFlatIP(dimension)
            self.faiss_index.add(embeddings.astype('float32'))
            
            logger.info(f"✅ FAISS index created with {embeddings.shape[0]} vectors")
            
        except Exception as e:
            logger.warning(f"FAISS index creation failed: {e}")
            self.embeddings_matrix = embeddings
    
    def _store_in_vector_db(self):
        """Store documents in vector database"""
        if not self.vector_store:
            return
        
        try:
            # Prepare data for ChromaDB
            documents = []
            metadatas = []
            ids = []
            
            for doc in self.documents:
                documents.append(doc.content)
                metadatas.append({
                    **doc.metadata,
                    'doc_type': doc.doc_type,
                    'difficulty_level': doc.difficulty_level,
                    'educational_tags': ','.join(doc.educational_tags)
                })
                ids.append(doc.source)
            
            # Add to collection
            self.vector_store.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info("✅ Documents stored in vector database")
            
        except Exception as e:
            logger.warning(f"Vector database storage failed: {e}")
    
    def retrieve_advanced(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.3,
        doc_types: Optional[List[str]] = None,
        difficulty_filter: Optional[str] = None,
        use_cache: bool = True
    ) -> List[Tuple[EnhancedDocument, float]]:
        """
        Advanced retrieval with caching, filtering, and performance optimization
        """
        start_time = time.time()
        self.retrieval_stats['total_queries'] += 1
        
        # Check cache first
        if use_cache:
            cached_result = self._get_cached_result(query, top_k, similarity_threshold)
            if cached_result:
                self.retrieval_stats['cache_hits'] += 1
                return cached_result
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Retrieve using FAISS or standard similarity
            if self.faiss_index is not None:
                similarities, indices = self._faiss_search(query_embedding, top_k * 2)
            else:
                similarities, indices = self._standard_search(query_embedding, top_k * 2)
            
            # Filter and rank results
            results = self._filter_and_rank_results(
                indices, similarities, query,
                doc_types, difficulty_filter, similarity_threshold, top_k
            )
            
            # Update document access stats
            for doc, _ in results:
                doc.update_access_stats()
            
            # Cache results
            if use_cache:
                self._cache_result(query, top_k, similarity_threshold, results)
            
            # Update performance stats
            retrieval_time = time.time() - start_time
            self._update_performance_stats(retrieval_time)
            
            logger.info(f"Retrieved {len(results)} documents in {retrieval_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"Advanced retrieval failed: {e}")
            return []
    
    def _faiss_search(self, query_embedding: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search using FAISS index"""
        # Normalize query embedding
        query_embedding = query_embedding.astype('float32')
        import faiss
        faiss.normalize_L2(query_embedding)
        
        # Search
        similarities, indices = self.faiss_index.search(query_embedding, k)
        return similarities[0], indices[0]
    
    def _standard_search(self, query_embedding: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Standard similarity search using numpy"""
        if self.embeddings_matrix is None:
            return np.array([]), np.array([])
        
        # Calculate similarities
        similarities = np.dot(query_embedding, self.embeddings_matrix.T).flatten()
        
        # Get top k indices
        indices = np.argsort(similarities)[::-1][:k]
        return similarities[indices], indices
    
    def _filter_and_rank_results(
        self,
        indices: np.ndarray,
        similarities: np.ndarray,
        query: str,
        doc_types: Optional[List[str]],
        difficulty_filter: Optional[str],
        similarity_threshold: float,
        top_k: int
    ) -> List[Tuple[EnhancedDocument, float]]:
        """Filter and rank search results"""
        
        results = []
        
        for idx, similarity in zip(indices, similarities):
            if similarity < similarity_threshold:
                continue
            
            if idx >= len(self.documents):
                continue
            
            doc = self.documents[idx]
            
            # Apply document type filter
            if doc_types and doc.doc_type not in doc_types:
                continue
            
            # Apply difficulty filter
            if difficulty_filter and doc.difficulty_level != difficulty_filter.lower():
                continue
            
            # Calculate enhanced score
            enhanced_score = self._calculate_enhanced_score(doc, similarity, query)
            
            results.append((doc, enhanced_score))
            
            if len(results) >= top_k:
                break
        
        # Sort by enhanced score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _calculate_enhanced_score(self, doc: EnhancedDocument, base_similarity: float, query: str) -> float:
        """Calculate enhanced scoring with multiple factors"""
        
        score = base_similarity
        
        # Boost score based on user ratings
        if doc.average_rating > 0:
            rating_boost = (doc.average_rating - 2.5) * 0.1  # Center around 2.5
            score += rating_boost
        
        # Boost score based on access frequency (popularity)
        if doc.access_count > 0:
            popularity_boost = min(0.1, doc.access_count * 0.01)
            score += popularity_boost
        
        # Boost score for difficulty match
        if doc.difficulty_level == self.difficulty_level:
            score += 0.05
        
        # Boost score for recent content
        days_since_access = (datetime.now() - doc.last_accessed).days
        if days_since_access < 7:
            recency_boost = 0.02 * (7 - days_since_access) / 7
            score += recency_boost
        
        # Apply feedback scores if available
        feedback_score = self.feedback_scores.get(query, {}).get(doc.source, 0)
        score += feedback_score * 0.1
        
        return score
    
    def hybrid_retrieve_enhanced(
        self,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.2,
        educational_weight: float = 0.1
    ) -> List[Tuple[EnhancedDocument, float]]:
        """
        Enhanced hybrid retrieval combining multiple scoring methods
        """
        
        # Get semantic results
        semantic_results = self.retrieve_advanced(query, top_k=top_k * 2, similarity_threshold=0.1)
        
        if not semantic_results:
            return []
        
        # Calculate keyword scores
        query_words = set(query.lower().split())
        keyword_scores = {}
        
        for doc, _ in semantic_results:
            doc_words = set(doc.content.lower().split())
            overlap = len(query_words.intersection(doc_words))
            keyword_score = overlap / len(query_words) if query_words else 0
            keyword_scores[doc.source] = keyword_score
        
        # Calculate educational relevance scores
        educational_scores = {}
        for doc, _ in semantic_results:
            edu_score = self._calculate_educational_relevance(doc, query)
            educational_scores[doc.source] = edu_score
        
        # Combine all scores
        combined_results = []
        for doc, semantic_score in semantic_results:
            keyword_score = keyword_scores.get(doc.source, 0)
            edu_score = educational_scores.get(doc.source, 0)
            
            combined_score = (
                semantic_weight * semantic_score +
                keyword_weight * keyword_score +
                educational_weight * edu_score
            )
            
            combined_results.append((doc, combined_score))
        
        # Sort and return top results
        combined_results.sort(key=lambda x: x[1], reverse=True)
        return combined_results[:top_k]
    
    def _calculate_educational_relevance(self, doc: EnhancedDocument, query: str) -> float:
        """Calculate educational relevance score"""
        
        score = 0.0
        query_lower = query.lower()
        
        # Check for educational question patterns
        question_patterns = ['what is', 'how to', 'why does', 'explain', 'define', 'describe']
        if any(pattern in query_lower for pattern in question_patterns):
            # Boost definitions and explanatory content
            if doc.doc_type in ['concept', 'topic', 'summary']:
                score += 0.3
            if 'definition' in doc.educational_tags:
                score += 0.2
        
        # Check for practical application queries
        practical_patterns = ['example', 'application', 'use case', 'practice']
        if any(pattern in query_lower for pattern in practical_patterns):
            if doc.doc_type == 'example':
                score += 0.4
            if 'practical' in doc.educational_tags:
                score += 0.2
        
        # Boost based on educational tags matching query intent
        for tag in doc.educational_tags:
            if tag in query_lower:
                score += 0.1
        
        return min(1.0, score)
    
    def get_contextual_response(
        self,
        query: str,
        max_context_length: int = 2000,
        include_metadata: bool = True,
        difficulty_level: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive contextual response with enhanced features
        """
        
        # Use provided difficulty or default
        target_difficulty = difficulty_level or self.difficulty_level
        
        # Retrieve relevant documents with difficulty filtering
        retrieved_docs = self.hybrid_retrieve_enhanced(
            query, 
            top_k=5
        )
        
        # Filter by difficulty if specified
        if target_difficulty:
            retrieved_docs = [
                (doc, score) for doc, score in retrieved_docs
                if doc.difficulty_level == target_difficulty.lower()
            ]
        
        if not retrieved_docs:
            return {
                'context': f"No relevant content found for {target_difficulty} level.",
                'sources': [],
                'confidence': 0.0,
                'suggestions': self._get_alternative_suggestions(query)
            }
        
        # Build enhanced context
        context_parts = []
        sources = []
        current_length = 0
        total_confidence = 0
        
        for doc, score in retrieved_docs:
            # Prepare document text with educational formatting
            doc_text = self._format_document_for_context(doc, include_metadata)
            
            # Check length constraints
            if current_length + len(doc_text) > max_context_length:
                remaining_space = max_context_length - current_length - 50
                if remaining_space > 100:
                    doc_text = doc_text[:remaining_space] + "..."
                    context_parts.append(doc_text)
                    sources.append(self._create_source_info(doc, score))
                break
            
            context_parts.append(doc_text)
            sources.append(self._create_source_info(doc, score))
            current_length += len(doc_text)
            total_confidence += score
        
        # Calculate average confidence
        avg_confidence = total_confidence / len(retrieved_docs) if retrieved_docs else 0
        
        # Compile context
        context = "\n\n".join(context_parts)
        
        # Add educational guidance
        educational_note = self._generate_educational_note(target_difficulty, len(sources))
        context += f"\n\n{educational_note}"
        
        return {
            'context': context,
            'sources': sources,
            'confidence': avg_confidence,
            'difficulty_level': target_difficulty,
            'document_types': list(set(doc.doc_type for doc, _ in retrieved_docs)),
            'suggestions': self._get_related_queries(query, retrieved_docs)
        }
    
    def _format_document_for_context(self, doc: EnhancedDocument, include_metadata: bool) -> str:
        """Format document for educational context"""
        
        content = doc.content
        
        if include_metadata:
            metadata_parts = []
            
            # Add source information
            if doc.doc_type == "topic":
                metadata_parts.append(f"Topic: {doc.metadata.get('topic_name', 'Unknown')}")
            elif doc.doc_type == "chapter":
                metadata_parts.append(f"Chapter: {doc.metadata.get('chapter_name', 'Unknown')}")
            elif doc.doc_type == "concept":
                metadata_parts.append("Important Concept")
            elif doc.doc_type == "example":
                metadata_parts.append("Example")
            
            # Add difficulty level
            if doc.difficulty_level:
                metadata_parts.append(f"Level: {doc.difficulty_level.title()}")
            
            # Add educational tags
            if doc.educational_tags:
                metadata_parts.append(f"Tags: {', '.join(doc.educational_tags[:3])}")
            
            if metadata_parts:
                content = f"[{' | '.join(metadata_parts)}]\n{content}"
        
        return content
    
    def _create_source_info(self, doc: EnhancedDocument, score: float) -> Dict[str, Any]:
        """Create detailed source information"""
        return {
            'source_id': doc.source,
            'doc_type': doc.doc_type,
            'confidence': score,
            'difficulty_level': doc.difficulty_level,
            'educational_tags': doc.educational_tags,
            'average_rating': doc.average_rating,
            'access_count': doc.access_count,
            'word_count': doc.word_count,
            'key_concepts': doc.key_concepts
        }
    
    def _generate_educational_note(self, difficulty_level: str, source_count: int) -> str:
        """Generate educational guidance note"""
        
        notes = {
            'beginner': f"📚 This {difficulty_level} level explanation uses {source_count} source{'s' if source_count > 1 else ''} to provide a clear, foundational understanding.",
            'intermediate': f"🎯 This {difficulty_level} level content draws from {source_count} source{'s' if source_count > 1 else ''} to build upon basic concepts.",
            'advanced': f"🚀 This {difficulty_level} level analysis combines {source_count} source{'s' if source_count > 1 else ''} for comprehensive understanding."
        }
        
        return notes.get(difficulty_level, f"📖 Information compiled from {source_count} relevant source{'s' if source_count > 1 else ''}.")
    
    def _get_alternative_suggestions(self, query: str) -> List[str]:
        """Get alternative query suggestions when no results found"""
        suggestions = []
        
        # Extract key terms from query
        query_words = query.lower().split()
        
        # Find documents with partial matches
        partial_matches = []
        for doc in self.documents[:20]:  # Check first 20 documents
            doc_words = set(doc.content.lower().split())
            overlap = len(set(query_words).intersection(doc_words))
            if overlap > 0:
                partial_matches.append((doc, overlap))
        
        # Sort by overlap and suggest related topics
        partial_matches.sort(key=lambda x: x[1], reverse=True)
        
        for doc, _ in partial_matches[:3]:
            if doc.doc_type == 'topic' and doc.metadata.get('topic_name'):
                suggestions.append(f"Try asking about: {doc.metadata['topic_name']}")
            elif doc.key_concepts:
                suggestions.append(f"Related concept: {doc.key_concepts[0]}")
        
        return suggestions[:3]
    
    def _get_related_queries(self, original_query: str, retrieved_docs: List[Tuple[EnhancedDocument, float]]) -> List[str]:
        """Generate related query suggestions"""
        suggestions = []
        
        # Extract concepts from retrieved documents
        all_concepts = []
        for doc, _ in retrieved_docs:
            all_concepts.extend(doc.key_concepts)
        
        # Generate suggestions based on concepts
        unique_concepts = list(set(all_concepts))[:3]
        for concept in unique_concepts:
            suggestions.append(f"What is {concept}?")
            suggestions.append(f"How does {concept} work?")
        
        return suggestions[:5]
    
    # Caching methods
    def _get_cached_result(self, query: str, top_k: int, threshold: float) -> Optional[List[Tuple[EnhancedDocument, float]]]:
        """Get cached retrieval result"""
        cache_key = f"{query}_{top_k}_{threshold}"
        
        if cache_key in self.query_cache:
            cached_data, timestamp = self.query_cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                # Convert cached data back to document objects
                results = []
                for source_id, score in cached_data:
                    if source_id in self.document_index:
                        results.append((self.document_index[source_id], score))
                return results
            else:
                # Remove expired cache entry
                del self.query_cache[cache_key]
        
        return None
    
    def _cache_result(self, query: str, top_k: int, threshold: float, results: List[Tuple[EnhancedDocument, float]]):
        """Cache retrieval result"""
        cache_key = f"{query}_{top_k}_{threshold}"
        
        # Store only source IDs and scores to reduce memory usage
        cached_data = [(doc.source, score) for doc, score in results]
        self.query_cache[cache_key] = (cached_data, datetime.now())
        
        # Limit cache size
        if len(self.query_cache) > 100:
            # Remove oldest entries
            oldest_key = min(self.query_cache.keys(), key=lambda k: self.query_cache[k][1])
            del self.query_cache[oldest_key]
    
    def _save_index_cache(self):
        """Save index to disk cache"""
        try:
            cache_file = self.cache_dir / "rag_index.pkl"
            
            cache_data = {
                'documents': self.documents,
                'embeddings_matrix': self.embeddings_matrix,
                'is_indexed': self.is_indexed,
                'embedding_model_name': self.embedding_model_name,
                'created_at': datetime.now().isoformat()
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"✅ Index cached to {cache_file}")
            
        except Exception as e:
            logger.warning(f"Failed to cache index: {e}")
    
    def _load_index_cache(self) -> bool:
        """Load index from disk cache"""
        try:
            cache_file = self.cache_dir / "rag_index.pkl"
            
            if not cache_file.exists():
                return False
            
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Verify cache is compatible
            if cache_data.get('embedding_model_name') != self.embedding_model_name:
                logger.info("Embedding model changed, cache invalid")
                return False
            
            # Load cached data
            self.documents = cache_data['documents']
            self.embeddings_matrix = cache_data['embeddings_matrix']
            self.is_indexed = cache_data['is_indexed']
            
            # Rebuild document index
            self.document_index = {doc.source: doc for doc in self.documents}
            
            logger.info(f"✅ Index loaded from cache: {len(self.documents)} documents")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load cached index: {e}")
            return False
    
    def add_feedback_enhanced(self, query: str, doc_source: str, helpful: bool, rating: float = None):
        """Enhanced feedback system with ratings"""
        
        # Add basic feedback
        if query not in self.feedback_scores:
            self.feedback_scores[query] = {}
        
        feedback_value = 1.0 if helpful else -0.5
        self.feedback_scores[query][doc_source] = feedback_value
        
        # Add rating if provided
        if rating is not None and doc_source in self.document_index:
            self.document_index[doc_source].add_rating(rating)
        
        self.retrieval_stats['feedback_count'] += 1
        
        logger.info(f"Added enhanced feedback: {query[:30]}... -> {doc_source} = {helpful} (rating: {rating})")
    
    def _update_performance_stats(self, retrieval_time: float):
        """Update performance statistics"""
        total_queries = self.retrieval_stats['total_queries']
        current_avg = self.retrieval_stats['avg_retrieval_time']
        
        # Update running average
        new_avg = ((current_avg * (total_queries - 1)) + retrieval_time) / total_queries
        self.retrieval_stats['avg_retrieval_time'] = new_avg
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive RAG pipeline statistics"""
        
        doc_types = {}
        difficulty_distribution = {}
        educational_tags = {}
        
        for doc in self.documents:
            # Document types
            doc_types[doc.doc_type] = doc_types.get(doc.doc_type, 0) + 1
            
            # Difficulty distribution
            difficulty_distribution[doc.difficulty_level] = difficulty_distribution.get(doc.difficulty_level, 0) + 1
            
            # Educational tags
            for tag in doc.educational_tags:
                educational_tags[tag] = educational_tags.get(tag, 0) + 1
        
        # Calculate average ratings
        rated_docs = [doc for doc in self.documents if doc.user_ratings]
        avg_rating = np.mean([doc.average_rating for doc in rated_docs]) if rated_docs else 0
        
        return {
            'total_documents': len(self.documents),
            'is_indexed': self.is_indexed,
            'embedding_dimension': self.embeddings_matrix.shape[1] if self.embeddings_matrix is not None else 0,
            'document_types': doc_types,
            'difficulty_distribution': difficulty_distribution,
            'educational_tags': dict(list(educational_tags.items())[:10]),  # Top 10 tags
            'performance_stats': self.retrieval_stats.copy(),
            'caching_stats': {
                'cache_size': len(self.query_cache),
                'cache_hit_rate': self.retrieval_stats['cache_hits'] / max(1, self.retrieval_stats['total_queries'])
            },
            'quality_metrics': {
                'average_document_length': np.mean([doc.word_count for doc in self.documents]),
                'average_rating': avg_rating,
                'total_ratings': sum(len(doc.user_ratings) for doc in self.documents),
                'most_accessed_doc_type': max(doc_types, key=doc_types.get) if doc_types else None
            },
            'enhanced_features': {
                'faiss_enabled': self.faiss_index is not None,
                'topic_modeling_enabled': self.topic_model is not None,
                'vector_store_enabled': self.vector_store is not None,
                'total_topics': len(set(doc.metadata.get('topic_id', -1) for doc in self.documents if doc.metadata.get('topic_id') is not None))
            }
        }

# Enhanced utility functions

def create_advanced_rag_pipeline(
    knowledge_base: Dict[str, Any], 
    embedding_model: str = "all-MiniLM-L6-v2",
    difficulty_level: str = "intermediate",
    use_cache: bool = True
) -> AdvancedRAGPipeline:
    """Create and initialize advanced RAG pipeline"""
    
    pipeline = AdvancedRAGPipeline(
        embedding_model_name=embedding_model,
        difficulty_level=difficulty_level
    )
    
    # Try to load from cache first
    if use_cache and pipeline._load_index_cache():
        st.success(f"✅ Loaded cached RAG index with {len(pipeline.documents)} documents")
        return pipeline
    
    # Build new index
    with st.spinner("🔄 Building advanced RAG index..."):
        doc_count = pipeline.ingest_knowledge_base_enhanced(knowledge_base)
        
        if doc_count > 0:
            success = pipeline.build_advanced_index()
            if success:
                st.success(f"✅ Advanced RAG index built: {doc_count} documents indexed")
            else:
                st.error("❌ Failed to build RAG index")
        else:
            st.warning("⚠️ No documents found in knowledge base")
    
    return pipeline

@st.cache_resource
def get_global_rag_pipeline(_knowledge_base_hash: str, _embedding_model: str, _difficulty: str) -> AdvancedRAGPipeline:
    """Get globally cached RAG pipeline"""
    return AdvancedRAGPipeline(
        embedding_model_name=_embedding_model,
        difficulty_level=_difficulty
    )

def calculate_knowledge_base_hash(knowledge_base: Dict[str, Any]) -> str:
    """Calculate hash of knowledge base for caching"""
    content_str = json.dumps(knowledge_base, sort_keys=True)
    return hashlib.md5(content_str.encode()).hexdigest()

# Integration with Streamlit interface

def display_rag_interface(pipeline: AdvancedRAGPipeline):
    """Display RAG pipeline interface in Streamlit"""
    
    st.subheader("🔍 Advanced Document Search & Retrieval")
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Enter your question:",
            placeholder="e.g., What is machine learning?",
            help="Ask any question about the uploaded content"
        )
    
    with col2:
        search_button = st.button("🔍 Search", type="primary")
    
    # Advanced search options
    with st.expander("⚙️ Advanced Search Options"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            top_k = st.slider("Number of results", 1, 10, 5)
            similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.3, 0.1)
        
        with col2:
            doc_types = st.multiselect(
                "Document types",
                options=['topic', 'concept', 'example', 'chapter', 'keypoint', 'summary'],
                default=None
            )
            
            difficulty_filter = st.selectbox(
                "Difficulty level",
                options=[None, 'beginner', 'intermediate', 'advanced'],
                index=0
            )
        
        with col3:
            use_hybrid = st.checkbox("Use hybrid search", value=True)
            include_metadata = st.checkbox("Include metadata", value=True)
    
    # Perform search
    if search_button and query:
        with st.spinner("🔍 Searching knowledge base..."):
            
            if use_hybrid:
                results = pipeline.hybrid_retrieve_enhanced(query, top_k=top_k)
            else:
                results = pipeline.retrieve_advanced(
                    query=query,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold,
                    doc_types=doc_types if doc_types else None,
                    difficulty_filter=difficulty_filter
                )
            
            if results:
                st.success(f"✅ Found {len(results)} relevant results")
                
                # Display results
                for i, (doc, score) in enumerate(results, 1):
                    with st.expander(f"📄 Result {i}: {doc.doc_type.title()} (Score: {score:.3f})"):
                        
                        # Document content
                        st.write(doc.content)
                        
                        # Metadata
                        if include_metadata:
                            st.markdown("**📊 Document Information:**")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.write(f"**Type:** {doc.doc_type}")
                                st.write(f"**Difficulty:** {doc.difficulty_level}")
                                st.write(f"**Word Count:** {doc.word_count}")
                            
                            with col2:
                                st.write(f"**Access Count:** {doc.access_count}")
                                st.write(f"**Average Rating:** {doc.average_rating:.1f}/5.0")
                                st.write(f"**Readability:** {doc.readability_score:.1f}")
                            
                            with col3:
                                if doc.educational_tags:
                                    st.write(f"**Tags:** {', '.join(doc.educational_tags[:3])}")
                                if doc.key_concepts:
                                    st.write(f"**Key Concepts:** {', '.join(doc.key_concepts[:3])}")
                        
                        # Feedback
                        col1, col2, col3 = st.columns([1, 1, 2])
                        
                        with col1:
                            if st.button("👍 Helpful", key=f"helpful_{i}"):
                                pipeline.add_feedback_enhanced(query, doc.source, True)
                                st.success("Thanks for your feedback!")
                        
                        with col2:
                            if st.button("👎 Not Helpful", key=f"not_helpful_{i}"):
                                pipeline.add_feedback_enhanced(query, doc.source, False)
                                st.success("Thanks for your feedback!")
                        
                        with col3:
                            rating = st.select_slider(
                                "Rate this result:",
                                options=[1, 2, 3, 4, 5],
                                value=3,
                                key=f"rating_{i}"
                            )
                            if st.button("Submit Rating", key=f"submit_rating_{i}"):
                                pipeline.add_feedback_enhanced(query, doc.source, True, rating)
                                st.success(f"Rating {rating}/5 submitted!")
            else:
                st.warning("❌ No relevant results found")
                
                # Get suggestions
                context_response = pipeline.get_contextual_response(query)
                if context_response.get('suggestions'):
                    st.markdown("**💡 Try these related queries:**")
                    for suggestion in context_response['suggestions']:
                        st.write(f"• {suggestion}")
    
    # Display pipeline statistics
    if st.checkbox("📊 Show Pipeline Statistics"):
        stats = pipeline.get_comprehensive_statistics()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📚 Content Statistics**")
            st.json({
                'Total Documents': stats['total_documents'],
                'Document Types': stats['document_types'],
                'Difficulty Distribution': stats['difficulty_distribution']
            })
        
        with col2:
            st.markdown("**⚡ Performance Statistics**")
            st.json({
                'Total Queries': stats['performance_stats']['total_queries'],
                'Cache Hit Rate': f"{stats['caching_stats']['cache_hit_rate']:.1%}",
                'Avg Retrieval Time': f"{stats['performance_stats']['avg_retrieval_time']:.3f}s"
            })

# Testing and validation

def test_advanced_rag_pipeline():
    """Comprehensive test for advanced RAG pipeline"""
    
    # Enhanced sample knowledge base
    sample_kb = {
        "chapters": {
            "Introduction to AI": {
                "content": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines. It involves various techniques such as machine learning, neural networks, and natural language processing.",
                "difficulty": "Beginner",
                "topics": {
                    "Machine Learning": {
                        "content": "Machine learning is a subset of AI that focuses on algorithms that can learn from data without explicit programming.",
                        "key_points": ["Supervised learning", "Unsupervised learning", "Reinforcement learning"],
                        "difficulty": "Intermediate"
                    },
                    "Neural Networks": {
                        "content": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information.",
                        "key_points": ["Neurons", "Layers", "Activation functions", "Backpropagation"],
                        "difficulty": "Advanced"
                    }
                }
            }
        },
        "concepts": ["Deep Learning", "Classification", "Regression", "Clustering"],
        "examples": [
            "Image recognition using convolutional neural networks",
            "Natural language processing with transformers",
            "Recommendation systems using collaborative filtering"
        ]
    }
    
    print("🧪 Testing Advanced RAG Pipeline...")
    
    # Create and test pipeline
    pipeline = AdvancedRAGPipeline()
    doc_count = pipeline.ingest_knowledge_base_enhanced(sample_kb)
    print(f"📄 Ingested {doc_count} documents")
    
    # Build index
    success = pipeline.build_advanced_index()
    print(f"🔍 Index built: {success}")
    
    # Test basic retrieval
    results = pipeline.retrieve_advanced("What is machine learning?", top_k=3)
    print(f"🔍 Basic retrieval: {len(results)} results")
    
    # Test hybrid retrieval
    hybrid_results = pipeline.hybrid_retrieve_enhanced("artificial intelligence examples", top_k=3)
    print(f"🔍 Hybrid retrieval: {len(hybrid_results)} results")
    
    # Test contextual response
    context_response = pipeline.get_contextual_response("Explain neural networks")
    print(f"📝 Context length: {len(context_response['context'])} characters")
    print(f"📊 Confidence: {context_response['confidence']:.3f}")
    
    # Test feedback
    if results:
        doc_source = results[0][0].source
        pipeline.add_feedback_enhanced("What is machine learning?", doc_source, True, 4.5)
        print("👍 Feedback added")
    
    # Get statistics
    stats = pipeline.get_comprehensive_statistics()
    print("\n📊 Pipeline Statistics:")
    print(f"   Total Documents: {stats['total_documents']}")
    print(f"   Document Types: {stats['document_types']}")
    print(f"   Enhanced Features: {stats['enhanced_features']}")
    
    print("✅ Advanced RAG Pipeline test completed!")

if __name__ == "__main__":
    test_advanced_rag_pipeline()
