#!/usr/bin/env python3
"""
Enhanced Utility Helper Functions for IntelliLearn AI Educational Assistant
Comprehensive functionality for open-source educational AI platform
Integrated with difficulty-aware learning and advanced analytics
"""

import os
import re
import json
import hashlib
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pickle
import base64

# Enhanced imports for open-source integration
try:
    import textstat
    from sentence_transformers import SentenceTransformer
    import spacy
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import plotly.graph_objects as go
    import plotly.express as px
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/intellilearn.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedDirectoryManager:
    """Enhanced directory management with caching and organization"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.required_dirs = [
            "data/uploaded_books",
            "data/processed_content", 
            "assets/generated_content",
            "assets/video_output",
            "assets/audio_output",
            "logs",
            "cache/models",
            "cache/embeddings",
            "cache/rag_index",
            "cache/user_data",
            "backup",
            "temp"
        ]
    
    def setup_directories(self):
        """Create enhanced directory structure"""
        created_dirs = []
        
        for directory in self.required_dirs:
            full_path = self.base_dir / directory
            try:
                full_path.mkdir(parents=True, exist_ok=True)
                created_dirs.append(str(full_path))
                logger.info(f"✅ Directory ensured: {full_path}")
            except Exception as e:
                logger.error(f"❌ Failed to create directory {full_path}: {e}")
        
        # Create .gitignore for sensitive directories
        self._create_gitignore()
        
        return created_dirs
    
    def _create_gitignore(self):
        """Create .gitignore for cache and sensitive directories"""
        gitignore_content = """
# IntelliLearn AI - Generated files
cache/
temp/
logs/
backup/
*.log
*.pkl
*.cache

# User data
data/uploaded_books/
data/processed_content/

# Generated assets
assets/generated_content/
assets/video_output/
assets/audio_output/

# Model files
models/
*.bin
*.safetensors
"""
        
        gitignore_path = self.base_dir / ".gitignore"
        try:
            with open(gitignore_path, 'w') as f:
                f.write(gitignore_content.strip())
            logger.info("✅ .gitignore created/updated")
        except Exception as e:
            logger.warning(f"Failed to create .gitignore: {e}")
    
    def get_directory_stats(self) -> Dict[str, Any]:
        """Get statistics about directory usage"""
        stats = {}
        
        for directory in self.required_dirs:
            dir_path = self.base_dir / directory
            if dir_path.exists():
                try:
                    file_count = len(list(dir_path.rglob('*')))
                    total_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                    stats[directory] = {
                        'exists': True,
                        'file_count': file_count,
                        'total_size_mb': round(total_size / (1024 * 1024), 2)
                    }
                except Exception as e:
                    stats[directory] = {'exists': True, 'error': str(e)}
            else:
                stats[directory] = {'exists': False}
        
        return stats

class AdvancedFileValidator:
    """Enhanced file validation with security and content analysis"""
    
    def __init__(self):
        self.allowed_extensions = {
            'documents': ['pdf', 'epub', 'txt', 'docx', 'rtf', 'odt'],
            'images': ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'],
            'audio': ['mp3', 'wav', 'ogg', 'aac', 'm4a'],
            'video': ['mp4', 'avi', 'mov', 'mkv', 'webm'],
            'data': ['csv', 'json', 'xlsx', 'xml']
        }
        
        self.max_file_sizes = {
            'documents': 50 * 1024 * 1024,  # 50MB
            'images': 10 * 1024 * 1024,     # 10MB
            'audio': 100 * 1024 * 1024,     # 100MB
            'video': 500 * 1024 * 1024,     # 500MB
            'data': 20 * 1024 * 1024        # 20MB
        }
    
    def validate_file_comprehensive(self, file_obj, filename: str = None) -> Dict[str, Any]:
        """Comprehensive file validation with security checks"""
        
        if filename is None:
            filename = getattr(file_obj, 'name', 'unknown_file')
        
        validation_result = {
            'valid': False,
            'filename': filename,
            'file_type': None,
            'category': None,
            'size': 0,
            'security_check': False,
            'content_preview': None,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Basic file info
            file_content = file_obj.read()
            file_obj.seek(0)  # Reset pointer
            
            validation_result['size'] = len(file_content)
            file_extension = filename.lower().split('.')[-1] if '.' in filename else ''
            validation_result['file_type'] = file_extension
            
            # Determine category
            for category, extensions in self.allowed_extensions.items():
                if file_extension in extensions:
                    validation_result['category'] = category
                    break
            
            if not validation_result['category']:
                validation_result['errors'].append(f"Unsupported file type: {file_extension}")
                return validation_result
            
            # Size validation
            max_size = self.max_file_sizes[validation_result['category']]
            if validation_result['size'] > max_size:
                validation_result['errors'].append(
                    f"File too large: {validation_result['size']} bytes "
                    f"(max: {max_size} bytes)"
                )
                return validation_result
            
            # Security checks
            security_result = self._perform_security_checks(file_content, filename)
            validation_result['security_check'] = security_result['passed']
            
            if not security_result['passed']:
                validation_result['errors'].extend(security_result['issues'])
            
            validation_result['warnings'].extend(security_result.get('warnings', []))
            
            # Content preview for text files
            if validation_result['category'] == 'documents':
                preview = self._generate_content_preview(file_content, file_extension)
                validation_result['content_preview'] = preview
            
            # Final validation
            validation_result['valid'] = (
                len(validation_result['errors']) == 0 and 
                validation_result['security_check']
            )
            
            logger.info(f"File validation: {filename} - {'✅ Valid' if validation_result['valid'] else '❌ Invalid'}")
            
        except Exception as e:
            validation_result['errors'].append(f"Validation error: {str(e)}")
            logger.error(f"File validation failed for {filename}: {e}")
        
        return validation_result
    
    def _perform_security_checks(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Perform security checks on file content"""
        
        security_result = {
            'passed': True,
            'issues': [],
            'warnings': []
        }
        
        # Check for suspicious file signatures
        suspicious_signatures = [
            b'\x4d\x5a',  # Windows executable
            b'\x7f\x45\x4c\x46',  # Linux executable
            b'\xca\xfe\xba\xbe',  # Java class file
            b'<?php',  # PHP script
            b'<script',  # JavaScript (in beginning)
        ]
        
        file_start = file_content[:100].lower()
        for signature in suspicious_signatures:
            if signature in file_start:
                security_result['passed'] = False
                security_result['issues'].append(f"Suspicious file signature detected")
                break
        
        # Check filename for suspicious patterns
        suspicious_patterns = [
            r'\.exe$', r'\.bat$', r'\.cmd$', r'\.scr$', r'\.pif$',
            r'\.com$', r'\.scf$', r'\.vbs$', r'\.js$', r'\.jar$'
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, filename, re.IGNORECASE):
                security_result['passed'] = False
                security_result['issues'].append(f"Suspicious file extension: {filename}")
                break
        
        # Size-based warnings
        if len(file_content) < 100:
            security_result['warnings'].append("File is very small, may be incomplete")
        
        return security_result
    
    def _generate_content_preview(self, file_content: bytes, file_extension: str) -> Optional[str]:
        """Generate preview of file content"""
        
        try:
            if file_extension in ['txt', 'csv', 'json']:
                # Text-based files
                content_str = file_content.decode('utf-8', errors='ignore')
                return content_str[:500] + "..." if len(content_str) > 500 else content_str
            
            elif file_extension == 'pdf':
                return f"PDF file with {len(file_content)} bytes of content"
            
            else:
                return f"{file_extension.upper()} file with {len(file_content)} bytes"
        
        except Exception:
            return "Binary content - preview not available"

class IntelligentTextProcessor:
    """Enhanced text processing with NLP and educational focus"""
    
    def __init__(self):
        self.nlp_model = None
        self.sentence_model = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Initialize NLP models if available
        if ENHANCED_FEATURES_AVAILABLE:
            self._initialize_nlp_models()
    
    def _initialize_nlp_models(self):
        """Initialize NLP models for enhanced processing"""
        try:
            import spacy
            # Try to load English model
            try:
                self.nlp_model = spacy.load("en_core_web_sm")
                logger.info("✅ spaCy model loaded")
            except OSError:
                logger.warning("spaCy English model not found - some features limited")
            
            # Load sentence transformer for embeddings
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Sentence transformer loaded")
            
        except Exception as e:
            logger.warning(f"NLP model initialization failed: {e}")
    
    def clean_text_enhanced(self, text: str, preserve_structure: bool = False) -> str:
        """Enhanced text cleaning with educational content preservation"""
        
        if not text:
            return ""
        
        # Remove excessive whitespace but preserve paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        
        if not preserve_structure:
            text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines to double
        
        # Remove page headers/footers and page numbers
        text = re.sub(r'^Page \d+.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^Chapter \d+.*$', '', text, flags=re.MULTILINE)
        
        # Clean up common PDF artifacts
        text = re.sub(r'(?<=\w)-\s*\n\s*(?=\w)', '', text)  # Remove hyphenation
        text = re.sub(r'\x0c', '\n', text)  # Form feed to newline
        
        # Preserve educational formatting
        if preserve_structure:
            # Preserve bullet points and numbering
            text = re.sub(r'^(\s*[•·▪▫◦‣⁃]\s*)', r'\1', text, flags=re.MULTILINE)
            text = re.sub(r'^(\s*\d+[\.\)]\s*)', r'\1', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def chunk_text_intelligent(self, text: str, 
                              chunk_size: int = 1000, 
                              overlap: int = 100,
                              preserve_sentences: bool = True) -> List[Dict[str, Any]]:
        """Intelligent text chunking with educational content awareness"""
        
        if not text or len(text) <= chunk_size:
            return [{'content': text, 'chunk_id': 0, 'metadata': {}}] if text else []
        
        chunks = []
        
        # Split by educational boundaries first (chapters, sections, etc.)
        educational_splits = self._split_by_educational_boundaries(text)
        
        for section_idx, section in enumerate(educational_splits):
            section_chunks = self._chunk_section(
                section, chunk_size, overlap, preserve_sentences
            )
            
            for chunk_idx, chunk_content in enumerate(section_chunks):
                chunk_metadata = {
                    'section_index': section_idx,
                    'chunk_index': chunk_idx,
                    'total_chunks': len(section_chunks),
                    'word_count': len(chunk_content.split()),
                    'estimated_reading_time': self._estimate_reading_time(chunk_content)
                }
                
                # Extract educational features
                if ENHANCED_FEATURES_AVAILABLE:
                    chunk_metadata.update(self._extract_educational_features(chunk_content))
                
                chunks.append({
                    'content': chunk_content,
                    'chunk_id': len(chunks),
                    'metadata': chunk_metadata
                })
        
        return chunks
    
    def _split_by_educational_boundaries(self, text: str) -> List[str]:
        """Split text by educational boundaries (chapters, sections, etc.)"""
        
        # Educational boundary patterns
        patterns = [
            r'\n\s*Chapter \d+.*?\n',
            r'\n\s*Section \d+.*?\n',
            r'\n\s*\d+\.\d+.*?\n',  # Numbered sections
            r'\n\s*[A-Z][A-Z\s]+\n',  # ALL CAPS headings
        ]
        
        # Find all boundaries
        boundaries = []
        for pattern in patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            boundaries.extend([(m.start(), m.end()) for m in matches])
        
        # Sort boundaries by position
        boundaries.sort()
        
        # Split text at boundaries
        sections = []
        last_pos = 0
        
        for start, end in boundaries:
            if start > last_pos:
                sections.append(text[last_pos:start])
            last_pos = end
        
        # Add final section
        if last_pos < len(text):
            sections.append(text[last_pos:])
        
        return [section.strip() for section in sections if section.strip()]
    
    def _chunk_section(self, text: str, chunk_size: int, overlap: int, preserve_sentences: bool) -> List[str]:
        """Chunk a single section intelligently"""
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary if requested
            if preserve_sentences and end < len(text):
                # Look for sentence endings
                sentence_end = text.rfind('.', start, end)
                question_end = text.rfind('?', start, end)
                exclaim_end = text.rfind('!', start, end)
                
                best_end = max(sentence_end, question_end, exclaim_end)
                
                if best_end != -1 and best_end > start + chunk_size // 2:
                    end = best_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start with overlap
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def _extract_educational_features(self, text: str) -> Dict[str, Any]:
        """Extract educational features from text chunk"""
        
        features = {}
        
        # Readability analysis
        if ENHANCED_FEATURES_AVAILABLE:
            features['readability_score'] = textstat.flesch_reading_ease(text)
            features['grade_level'] = textstat.flesch_kincaid_grade(text)
        
        # Educational content indicators
        features['has_definitions'] = bool(re.search(r'\b(is defined as|means|refers to|definition)\b', text, re.IGNORECASE))
        features['has_examples'] = bool(re.search(r'\b(for example|such as|instance|e\.g\.)\b', text, re.IGNORECASE))
        features['has_formulas'] = bool(re.search(r'[=+\-*/^(){}[\]]', text))
        features['has_lists'] = bool(re.search(r'^\s*[•·▪▫◦‣⁃\d+\.\)]\s*', text, re.MULTILINE))
        
        # Extract key terms using TF-IDF
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text])
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top 5 terms
            top_indices = scores.argsort()[-5:][::-1]
            features['key_terms'] = [feature_names[i] for i in top_indices if scores[i] > 0]
            
        except Exception:
            features['key_terms'] = self.extract_key_terms_simple(text, 5)
        
        return features
    
    def extract_key_terms_enhanced(self, text: str, max_terms: int = 20, 
                                 min_freq: int = 2, exclude_common: bool = True) -> List[Dict[str, Any]]:
        """Enhanced key term extraction with frequency and context analysis"""
        
        if not text:
            return []
        
        # Clean and prepare text
        clean = self.clean_text_enhanced(text).lower()
        
        # Enhanced stop words list
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'this', 'that', 'these', 'those', 'is', 'are', 
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 
            'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'cannot', 'it', 'its', 'he', 'she', 'they', 'we', 'you', 'i',
            'also', 'even', 'just', 'only', 'very', 'more', 'most', 'much',
            'many', 'some', 'all', 'any', 'each', 'every', 'such', 'same'
        }
        
        # Extract terms with different patterns
        term_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns/titles
            r'\b[a-z]{3,}\b',  # Regular words (3+ chars)
            r'\b[a-z]+(?:-[a-z]+)+\b'  # Hyphenated terms
        ]
        
        term_freq = {}
        term_contexts = {}
        
        for pattern in term_patterns:
            matches = re.findall(pattern, clean)
            
            for match in matches:
                term = match.strip().lower()
                
                if exclude_common and term in stop_words:
                    continue
                
                if len(term) < 3:
                    continue
                
                # Count frequency
                term_freq[term] = term_freq.get(term, 0) + 1
                
                # Store context (sentence containing the term)
                if term not in term_contexts:
                    # Find sentence containing this term
                    sentences = re.split(r'[.!?]+', text)
                    for sentence in sentences:
                        if term in sentence.lower():
                            term_contexts[term] = sentence.strip()[:100]
                            break
        
        # Filter by minimum frequency
        filtered_terms = {k: v for k, v in term_freq.items() if v >= min_freq}
        
        # Sort by frequency and create detailed results
        sorted_terms = sorted(filtered_terms.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for term, freq in sorted_terms[:max_terms]:
            results.append({
                'term': term,
                'frequency': freq,
                'context': term_contexts.get(term, ''),
                'importance_score': freq / len(text.split()) * 1000  # Normalized importance
            })
        
        return results
    
    def extract_key_terms_simple(self, text: str, max_terms: int = 20) -> List[str]:
        """Simple key term extraction (fallback method)"""
        
        # Clean text and convert to lowercase
        clean = self.clean_text_enhanced(text).lower()
        
        # Basic stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'this', 'that', 'these', 'those', 'is', 'are', 
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 
            'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'cannot', 'it', 'its', 'he', 'she', 'they', 'we', 'you', 'i'
        }
        
        # Extract words (2+ characters)
        words = re.findall(r'\b[a-z]{2,}\b', clean)
        word_freq = {}
        
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top terms
        key_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [term[0] for term in key_terms[:max_terms]]
    
    def _estimate_reading_time(self, text: str, wpm: int = 200) -> Dict[str, Any]:
        """Estimate reading time with difficulty adjustment"""
        
        word_count = len(text.split())
        base_time = max(1, round(word_count / wpm))
        
        # Adjust for text complexity
        complexity_adjustment = 1.0
        
        if ENHANCED_FEATURES_AVAILABLE:
            try:
                grade_level = textstat.flesch_kincaid_grade(text)
                if grade_level > 12:
                    complexity_adjustment = 1.3
                elif grade_level > 8:
                    complexity_adjustment = 1.1
                elif grade_level < 6:
                    complexity_adjustment = 0.9
            except:
                pass
        
        adjusted_time = max(1, round(base_time * complexity_adjustment))
        
        return {
            'base_minutes': base_time,
            'adjusted_minutes': adjusted_time,
            'word_count': word_count,
            'complexity_factor': complexity_adjustment
        }

class AdvancedCacheManager:
    """Enhanced caching system with intelligent invalidation"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'invalidations': 0,
            'total_size': 0
        }
        
        # TTL settings for different cache types
        self.default_ttl = {
            'text_analysis': timedelta(hours=24),
            'embeddings': timedelta(days=7),
            'user_data': timedelta(hours=6),
            'model_outputs': timedelta(hours=12),
            'file_hashes': timedelta(days=30)
        }
    
    def get_cache_key(self, data: Union[str, bytes, Dict], prefix: str = "") -> str:
        """Generate cache key from data"""
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True)
        elif isinstance(data, bytes):
            data_str = data.decode('utf-8', errors='ignore')
        else:
            data_str = str(data)
        
        hash_obj = hashlib.sha256(data_str.encode()).hexdigest()[:16]
        return f"{prefix}_{hash_obj}" if prefix else hash_obj
    
    def get_cached_result(self, cache_key: str, cache_type: str = 'default') -> Optional[Any]:
        """Retrieve cached result with TTL check"""
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            self.cache_stats['misses'] += 1
            return None
        
        try:
            # Load cache data
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check TTL
            cached_time = cache_data['timestamp']
            ttl = self.default_ttl.get(cache_type, timedelta(hours=1))
            
            if datetime.now() - cached_time > ttl:
                # Cache expired
                cache_file.unlink()
                self.cache_stats['invalidations'] += 1
                return None
            
            # Cache hit
            self.cache_stats['hits'] += 1
            return cache_data['data']
            
        except Exception as e:
            logger.warning(f"Cache read error for {cache_key}: {e}")
            # Remove corrupted cache file
            try:
                cache_file.unlink()
            except:
                pass
            self.cache_stats['misses'] += 1
            return None
    
    def save_to_cache(self, cache_key: str, data: Any, cache_type: str = 'default') -> bool:
        """Save data to cache with timestamp"""
        
        try:
            cache_data = {
                'data': data,
                'timestamp': datetime.now(),
                'cache_type': cache_type
            }
            
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            # Update stats
            self.cache_stats['total_size'] = self.get_cache_size()
            
            logger.debug(f"Cached data saved: {cache_key}")
            return True
            
        except Exception as e:
            logger.error(f"Cache save error for {cache_key}: {e}")
            return False
    
    def get_cache_size(self) -> int:
        """Get total cache size in bytes"""
        total_size = 0
        try:
            for cache_file in self.cache_dir.glob('*.pkl'):
                total_size += cache_file.stat().st_size
        except Exception as e:
            logger.warning(f"Cache size calculation error: {e}")
        
        return total_size
    
    def cleanup_cache(self, max_age_days: int = 7, max_size_mb: int = 100) -> Dict[str, int]:
        """Clean up old cache files"""
        
        cleanup_stats = {'files_removed': 0, 'space_freed': 0}
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        max_size_bytes = max_size_mb * 1024 * 1024
        
        try:
            cache_files = []
            
            # Collect cache file info
            for cache_file in self.cache_dir.glob('*.pkl'):
                try:
                    stat = cache_file.stat()
                    # Try to get cache timestamp
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                        timestamp = cache_data.get('timestamp', datetime.fromtimestamp(stat.st_mtime))
                    
                    cache_files.append({
                        'path': cache_file,
                        'size': stat.st_size,
                        'timestamp': timestamp
                    })
                except Exception:
                    # Remove corrupted files
                    cache_file.unlink()
                    cleanup_stats['files_removed'] += 1
            
            # Sort by timestamp (oldest first)
            cache_files.sort(key=lambda x: x['timestamp'])
            
            # Remove old files
            for file_info in cache_files:
                if file_info['timestamp'] < cutoff_date:
                    file_info['path'].unlink()
                    cleanup_stats['files_removed'] += 1
                    cleanup_stats['space_freed'] += file_info['size']
            
            # Remove files if total size exceeds limit
            remaining_files = [f for f in cache_files if f['path'].exists()]
            total_size = sum(f['size'] for f in remaining_files)
            
            if total_size > max_size_bytes:
                # Remove oldest files until under limit
                for file_info in remaining_files:
                    if total_size <= max_size_bytes:
                        break
                    
                    file_info['path'].unlink()
                    cleanup_stats['files_removed'] += 1
                    cleanup_stats['space_freed'] += file_info['size']
                    total_size -= file_info['size']
            
            logger.info(f"Cache cleanup completed: {cleanup_stats}")
            
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
        
        return cleanup_stats
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        
        hit_rate = (self.cache_stats['hits'] / 
                   max(1, self.cache_stats['hits'] + self.cache_stats['misses'])) * 100
        
        return {
            **self.cache_stats,
            'hit_rate_percent': round(hit_rate, 2),
            'total_size_mb': round(self.cache_stats['total_size'] / (1024 * 1024), 2),
            'file_count': len(list(self.cache_dir.glob('*.pkl')))
        }

class AdvancedResponseFormatter:
    """Enhanced response formatting with educational focus"""
    
    def __init__(self):
        self.citation_styles = {
            'simple': self._format_simple_citations,
            'academic': self._format_academic_citations,
            'educational': self._format_educational_citations
        }
    
    def format_response_with_enhanced_citations(self, 
                                              response: str, 
                                              sources: List[Dict[str, Any]], 
                                              style: str = 'educational') -> str:
        """Enhanced response formatting with multiple citation styles"""
        
        if not sources:
            return response
        
        formatter = self.citation_styles.get(style, self._format_educational_citations)
        return formatter(response, sources)
    
    def _format_simple_citations(self, response: str, sources: List[Dict[str, Any]]) -> str:
        """Simple citation format"""
        citation_text = "\n\n**Sources from textbook:**\n"
        
        for i, source in enumerate(sources[:3], 1):
            source_text = source.get('content', str(source))[:100]
            if len(source_text) == 100:
                source_text += "..."
            citation_text += f"{i}. {source_text}\n"
        
        return response + citation_text
    
    def _format_academic_citations(self, response: str, sources: List[Dict[str, Any]]) -> str:
        """Academic-style citations"""
        citation_text = "\n\n**References:**\n"
        
        for i, source in enumerate(sources[:5], 1):
            # Extract metadata
            doc_type = source.get('doc_type', 'Document')
            source_name = source.get('source', f'Source {i}')
            difficulty = source.get('difficulty_level', 'Unknown')
            
            citation_text += f"[{i}] {doc_type.title()}: {source_name} (Level: {difficulty})\n"
        
        return response + citation_text
    
    def _format_educational_citations(self, response: str, sources: List[Dict[str, Any]]) -> str:
        """Educational-focused citation format"""
        
        citation_text = "\n\n📚 **Learning Sources:**\n"
        
        source_types = {}
        for source in sources[:5]:
            doc_type = source.get('doc_type', 'content')
            if doc_type not in source_types:
                source_types[doc_type] = []
            source_types[doc_type].append(source)
        
        # Group by type
        for doc_type, type_sources in source_types.items():
            emoji_map = {
                'topic': '📖',
                'concept': '💡',
                'example': '🌟',
                'chapter': '📚',
                'keypoint': '🎯',
                'summary': '📝'
            }
            
            emoji = emoji_map.get(doc_type, '📄')
            citation_text += f"\n{emoji} **{doc_type.title()}s:**\n"
            
            for i, source in enumerate(type_sources[:2], 1):  # Max 2 per type
                source_name = source.get('source', f'{doc_type} {i}').replace('_', ' ').title()
                difficulty = source.get('difficulty_level', 'intermediate')
                
                # Add confidence indicator
                confidence = source.get('confidence', 0.5)
                confidence_indicator = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.5 else "🔴"
                
                citation_text += f"  • {source_name} ({difficulty} level) {confidence_indicator}\n"
        
        # Add study suggestions
        citation_text += "\n💡 **Study Tip:** Review the sources above for deeper understanding!"
        
        return response + citation_text

class KnowledgeBaseValidator:
    """Enhanced knowledge base validation with detailed reporting"""
    
    def __init__(self):
        self.validation_rules = {
            'structure': self._validate_structure,
            'content': self._validate_content,
            'metadata': self._validate_metadata,
            'completeness': self._validate_completeness,
            'consistency': self._validate_consistency
        }
    
    def validate_knowledge_base_comprehensive(self, kb: Dict[Any, Any]) -> Dict[str, Any]:
        """Comprehensive knowledge base validation"""
        
        validation_result = {
            'valid': False,
            'score': 0.0,
            'total_checks': 0,
            'passed_checks': 0,
            'errors': [],
            'warnings': [],
            'suggestions': [],
            'detailed_results': {}
        }
        
        # Run all validation rules
        for rule_name, rule_func in self.validation_rules.items():
            try:
                rule_result = rule_func(kb)
                validation_result['detailed_results'][rule_name] = rule_result
                
                validation_result['total_checks'] += rule_result.get('checks_performed', 1)
                validation_result['passed_checks'] += rule_result.get('checks_passed', 0)
                
                validation_result['errors'].extend(rule_result.get('errors', []))
                validation_result['warnings'].extend(rule_result.get('warnings', []))
                validation_result['suggestions'].extend(rule_result.get('suggestions', []))
                
            except Exception as e:
                error_msg = f"Validation rule '{rule_name}' failed: {str(e)}"
                validation_result['errors'].append(error_msg)
                logger.error(error_msg)
        
        # Calculate overall score
        if validation_result['total_checks'] > 0:
            validation_result['score'] = (validation_result['passed_checks'] / 
                                        validation_result['total_checks']) * 100
        
        # Determine if valid (score > 70% and no critical errors)
        critical_errors = [e for e in validation_result['errors'] if 'missing required field' in e.lower()]
        validation_result['valid'] = (validation_result['score'] > 70.0 and 
                                    len(critical_errors) == 0)
        
        logger.info(f"Knowledge base validation completed: Score {validation_result['score']:.1f}%")
        
        return validation_result
    
    def _validate_structure(self, kb: Dict[Any, Any]) -> Dict[str, Any]:
        """Validate knowledge base structure"""
        
        result = {
            'checks_performed': 0,
            'checks_passed': 0,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        # Required top-level fields
        required_fields = ['topics', 'concepts', 'examples']
        optional_fields = ['chapters', 'metadata', 'generated_content']
        
        for field in required_fields:
            result['checks_performed'] += 1
            if field not in kb:
                result['errors'].append(f"Missing required field: {field}")
            else:
                result['checks_passed'] += 1
        
        # Check field types
        field_types = {
            'topics': dict,
            'concepts': list,
            'examples': list,
            'chapters': dict
        }
        
        for field, expected_type in field_types.items():
            if field in kb:
                result['checks_performed'] += 1
                if isinstance(kb[field], expected_type):
                    result['checks_passed'] += 1
                else:
                    result['errors'].append(f"Field '{field}' must be {expected_type.__name__}, got {type(kb[field]).__name__}")
        
        # Suggest optional fields
        for field in optional_fields:
            if field not in kb:
                result['suggestions'].append(f"Consider adding optional field: {field}")
        
        return result
    
    def _validate_content(self, kb: Dict[Any, Any]) -> Dict[str, Any]:
        """Validate content quality and completeness"""
        
        result = {
            'checks_performed': 0,
            'checks_passed': 0,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        # Validate topics
        if 'topics' in kb and isinstance(kb['topics'], dict):
            for topic_name, topic_data in kb['topics'].items():
                result['checks_performed'] += 3
                
                # Check topic structure
                if isinstance(topic_data, dict):
                    result['checks_passed'] += 1
                else:
                    result['errors'].append(f"Topic '{topic_name}' must be a dictionary")
                    continue
                
                # Check content
                if 'content' in topic_data and topic_data['content'].strip():
                    result['checks_passed'] += 1
                    
                    # Check content length
                    content_length = len(topic_data['content'])
                    if content_length < 50:
                        result['warnings'].append(f"Topic '{topic_name}' has very short content ({content_length} chars)")
                    elif content_length > 5000:
                        result['warnings'].append(f"Topic '{topic_name}' has very long content ({content_length} chars)")
                else:
                    result['errors'].append(f"Topic '{topic_name}' missing or empty content")
                
                # Check key points
                if 'key_points' in topic_data and isinstance(topic_data['key_points'], list):
                    result['checks_passed'] += 1
                    
                    if len(topic_data['key_points']) == 0:
                        result['warnings'].append(f"Topic '{topic_name}' has no key points")
                    elif len(topic_data['key_points']) > 10:
                        result['suggestions'].append(f"Topic '{topic_name}' has many key points ({len(topic_data['key_points'])}), consider grouping")
                else:
                    result['warnings'].append(f"Topic '{topic_name}' missing key_points")
        
        # Validate concepts
        if 'concepts' in kb and isinstance(kb['concepts'], list):
            result['checks_performed'] += 1
            if len(kb['concepts']) > 0:
                result['checks_passed'] += 1
                
                # Check for duplicate concepts
                concepts = [c.lower() for c in kb['concepts']]
                if len(concepts) != len(set(concepts)):
                    result['warnings'].append("Duplicate concepts found")
            else:
                result['warnings'].append("No concepts defined")
        
        # Validate examples
        if 'examples' in kb and isinstance(kb['examples'], list):
            result['checks_performed'] += 1
            if len(kb['examples']) > 0:
                result['checks_passed'] += 1
                
                # Check example quality
                short_examples = [ex for ex in kb['examples'] if len(str(ex)) < 20]
                if short_examples:
                    result['warnings'].append(f"{len(short_examples)} examples are very short")
            else:
                result['suggestions'].append("Consider adding examples to illustrate concepts")
        
        return result
    
    def _validate_metadata(self, kb: Dict[Any, Any]) -> Dict[str, Any]:
        """Validate metadata and educational attributes"""
        
        result = {
            'checks_performed': 0,
            'checks_passed': 0,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        # Check for difficulty levels in topics
        if 'topics' in kb:
            difficulty_levels = set()
            
            for topic_name, topic_data in kb['topics'].items():
                result['checks_performed'] += 1
                
                if 'difficulty' in topic_data:
                    difficulty = topic_data['difficulty'].lower()
                    difficulty_levels.add(difficulty)
                    
                    if difficulty in ['beginner', 'intermediate', 'advanced']:
                        result['checks_passed'] += 1
                    else:
                        result['warnings'].append(f"Topic '{topic_name}' has non-standard difficulty: {difficulty}")
                else:
                    result['suggestions'].append(f"Topic '{topic_name}' missing difficulty level")
            
            # Check difficulty distribution
            if len(difficulty_levels) == 1:
                result['suggestions'].append("All topics have same difficulty level - consider adding variety")
            elif len(difficulty_levels) == 0:
                result['suggestions'].append("No difficulty levels specified - consider adding them")
        
        return result
    
    def _validate_completeness(self, kb: Dict[Any, Any]) -> Dict[str, Any]:
        """Validate knowledge base completeness"""
        
        result = {
            'checks_performed': 0,
            'checks_passed': 0,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        # Calculate completeness metrics
        metrics = {
            'topics_count': len(kb.get('topics', {})),
            'concepts_count': len(kb.get('concepts', [])),
            'examples_count': len(kb.get('examples', [])),
            'chapters_count': len(kb.get('chapters', {}))
        }
        
        # Check minimum content requirements
        result['checks_performed'] += 3
        
        if metrics['topics_count'] >= 3:
            result['checks_passed'] += 1
        else:
            result['warnings'].append(f"Few topics ({metrics['topics_count']}) - consider adding more")
        
        if metrics['concepts_count'] >= 5:
            result['checks_passed'] += 1
        else:
            result['suggestions'].append(f"Few concepts ({metrics['concepts_count']}) - consider adding more")
        
        if metrics['examples_count'] >= 3:
            result['checks_passed'] += 1
        else:
            result['suggestions'].append(f"Few examples ({metrics['examples_count']}) - consider adding more")
        
        # Check balance
        if metrics['topics_count'] > 0:
            concept_to_topic_ratio = metrics['concepts_count'] / metrics['topics_count']
            if concept_to_topic_ratio < 1:
                result['suggestions'].append("Consider adding more concepts per topic")
            elif concept_to_topic_ratio > 5:
                result['suggestions'].append("Very high concept-to-topic ratio - consider organizing better")
        
        return result
    
    def _validate_consistency(self, kb: Dict[Any, Any]) -> Dict[str, Any]:
        """Validate knowledge base consistency"""
        
        result = {
            'checks_performed': 0,
            'checks_passed': 0,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        # Check topic naming consistency
        if 'topics' in kb:
            topic_names = list(kb['topics'].keys())
            
            # Check for consistent naming patterns
            result['checks_performed'] += 1
            
            # Simple check: all titles should be title case or all sentence case
            title_case_count = sum(1 for name in topic_names if name.istitle())
            if 0 < title_case_count < len(topic_names):
                result['warnings'].append("Inconsistent topic name capitalization")
            else:
                result['checks_passed'] += 1
        
        # Check difficulty progression (if chapters exist)
        if 'chapters' in kb and 'topics' in kb:
            result['checks_performed'] += 1
            
            # This is a simplified check - in real implementation, 
            # you'd want more sophisticated difficulty progression analysis
            difficulty_order = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
            
            prev_max_difficulty = 0
            consistent_progression = True
            
            for chapter_data in kb['chapters'].values():
                if 'topics' in chapter_data:
                    chapter_difficulties = []
                    
                    for topic_data in chapter_data['topics'].values():
                        difficulty = topic_data.get('difficulty', 'intermediate').lower()
                        chapter_difficulties.append(difficulty_order.get(difficulty, 2))
                    
                    if chapter_difficulties:
                        max_difficulty = max(chapter_difficulties)
                        if max_difficulty < prev_max_difficulty:
                            consistent_progression = False
                        prev_max_difficulty = max(prev_max_difficulty, max_difficulty)
            
            if consistent_progression:
                result['checks_passed'] += 1
            else:
                result['suggestions'].append("Consider reviewing difficulty progression across chapters")
        
        return result

# Utility functions (enhanced versions of originals)

def setup_directories():
    """Enhanced directory setup"""
    dir_manager = EnhancedDirectoryManager()
    return dir_manager.setup_directories()

def validate_file_type(filename: str, allowed_extensions: List[str] = None) -> bool:
    """Enhanced file type validation"""
    validator = AdvancedFileValidator()
    
    if allowed_extensions is None:
        # Use all document types from validator
        allowed_extensions = validator.allowed_extensions['documents']
    
    file_extension = filename.lower().split('.')[-1]
    return file_extension in allowed_extensions

def calculate_file_hash(file_content: bytes) -> str:
    """Enhanced file hash calculation with SHA256 option"""
    return hashlib.sha256(file_content).hexdigest()

def clean_text(text: str) -> str:
    """Enhanced text cleaning (backward compatibility)"""
    processor = IntelligentTextProcessor()
    return processor.clean_text_enhanced(text)

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Enhanced text chunking (backward compatibility)"""
    processor = IntelligentTextProcessor()
    chunks = processor.chunk_text_intelligent(text, chunk_size, overlap)
    return [chunk['content'] for chunk in chunks]

def extract_key_terms(text: str, max_terms: int = 20) -> List[str]:
    """Enhanced key terms extraction (backward compatibility)"""
    processor = IntelligentTextProcessor()
    return processor.extract_key_terms_simple(text, max_terms)

def format_response_with_citations(response: str, sources: List[str]) -> str:
    """Enhanced response formatting (backward compatibility)"""
    formatter = AdvancedResponseFormatter()
    # Convert simple sources to enhanced format
    enhanced_sources = [{'content': source, 'doc_type': 'content'} for source in sources]
    return formatter.format_response_with_enhanced_citations(response, enhanced_sources)

def validate_knowledge_base(kb: Dict[Any, Any]) -> Tuple[bool, List[str]]:
    """Enhanced knowledge base validation (backward compatibility)"""
    validator = KnowledgeBaseValidator()
    result = validator.validate_knowledge_base_comprehensive(kb)
    return result['valid'], result['errors']

# New enhanced utility functions

def create_learning_progress_tracker(total_steps: int, step_names: List[str] = None):
    """Enhanced progress tracking with educational features"""
    class EnhancedProgressTracker:
        def __init__(self, total, names):
            self.total = total
            self.current = 0
            self.names = names or [f"Step {i+1}" for i in range(total)]
            self.start_time = time.time()
            
            # Streamlit UI elements
            self.progress_container = st.container()
            with self.progress_container:
                self.progress_bar = st.progress(0)
                self.status_text = st.empty()
                self.time_text = st.empty()
        
        def update(self, step_name: str = None, details: str = None):
            self.current += 1
            progress = self.current / self.total
            
            # Update progress bar with color coding
            self.progress_bar.progress(progress)
            
            # Update status
            if step_name:
                status = step_name
            elif self.current <= len(self.names):
                status = self.names[self.current - 1]
            else:
                status = f"Step {self.current}"
            
            if details:
                status += f" - {details}"
            
            self.status_text.text(f"📚 {status}")
            
            # Update time estimate
            elapsed = time.time() - self.start_time
            if self.current > 0:
                avg_time_per_step = elapsed / self.current
                remaining_time = avg_time_per_step * (self.total - self.current)
                
                if remaining_time > 60:
                    time_str = f"{remaining_time/60:.1f} min remaining"
                else:
                    time_str = f"{remaining_time:.0f} sec remaining"
                
                self.time_text.text(f"⏱️ {time_str}")
        
        def complete(self, message: str = "🎉 Learning content processed successfully!"):
            self.progress_bar.progress(1.0)
            self.status_text.success(message)
            
            total_time = time.time() - self.start_time
            if total_time > 60:
                time_str = f"Completed in {total_time/60:.1f} minutes"
            else:
                time_str = f"Completed in {total_time:.0f} seconds"
            
            self.time_text.text(f"✅ {time_str}")
    
    return EnhancedProgressTracker(total_steps, step_names)

def display_knowledge_base_statistics(kb: Dict[Any, Any]):
    """Display comprehensive knowledge base statistics"""
    
    st.subheader("📊 Knowledge Base Statistics")
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        topic_count = len(kb.get('topics', {}))
        st.metric("Topics", topic_count)
    
    with col2:
        concept_count = len(kb.get('concepts', []))
        st.metric("Concepts", concept_count)
    
    with col3:
        example_count = len(kb.get('examples', []))
        st.metric("Examples", example_count)
    
    with col4:
        chapter_count = len(kb.get('chapters', {}))
        st.metric("Chapters", chapter_count)
    
    # Advanced statistics
    if ENHANCED_FEATURES_AVAILABLE and topic_count > 0:
        # Difficulty distribution
        difficulty_dist = {}
        total_content_length = 0
        
        for topic_data in kb.get('topics', {}).values():
            difficulty = topic_data.get('difficulty', 'Unknown')
            difficulty_dist[difficulty] = difficulty_dist.get(difficulty, 0) + 1
            
            content = topic_data.get('content', '')
            total_content_length += len(content)
        
        # Display difficulty distribution
        if difficulty_dist:
            st.subheader("🎯 Difficulty Distribution")
            
            fig = px.pie(
                values=list(difficulty_dist.values()),
                names=list(difficulty_dist.keys()),
                title="Topics by Difficulty Level"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Content analysis
        st.subheader("📝 Content Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            avg_content_length = total_content_length / topic_count if topic_count > 0 else 0
            st.metric("Average Content Length", f"{avg_content_length:.0f} chars")
            
            total_reading_time = estimate_reading_time_enhanced(total_content_length)
            st.metric("Total Reading Time", f"{total_reading_time} min")
        
        with col2:
            # Most common terms across all content
            all_content = ""
            for topic_data in kb.get('topics', {}).values():
                all_content += " " + topic_data.get('content', '')
            
            if all_content.strip():
                processor = IntelligentTextProcessor()
                key_terms = processor.extract_key_terms_simple(all_content, 10)
                
                st.write("**Top Terms:**")
                for i, term in enumerate(key_terms[:5], 1):
                    st.write(f"{i}. {term}")

def estimate_reading_time_enhanced(content_or_length: Union[str, int], 
                                 wpm: int = 200, 
                                 difficulty_adjustment: bool = True) -> int:
    """Enhanced reading time estimation with difficulty factors"""
    
    if isinstance(content_or_length, str):
        content = content_or_length
        word_count = len(content.split())
    else:
        word_count = content_or_length // 5  # Approximate words from character count
        content = None
    
    base_time = max(1, round(word_count / wpm))
    
    # Apply difficulty adjustment if content is available
    if difficulty_adjustment and content and ENHANCED_FEATURES_AVAILABLE:
        try:
            # Use readability score to adjust time
            reading_ease = textstat.flesch_reading_ease(content)
            
            if reading_ease < 30:  # Very difficult
                multiplier = 1.5
            elif reading_ease < 50:  # Difficult
                multiplier = 1.3
            elif reading_ease < 70:  # Standard
                multiplier = 1.0
            else:  # Easy
                multiplier = 0.9
            
            return max(1, round(base_time * multiplier))
        
        except Exception:
            pass
    
    return base_time

def create_download_button_enhanced(content: Union[str, bytes], 
                                  filename: str, 
                                  mime_type: str = "text/plain",
                                  button_text: str = None) -> bool:
    """Enhanced download functionality with better UX"""
    
    if button_text is None:
        button_text = f"📥 Download {filename}"
    
    if isinstance(content, str):
        content_bytes = content.encode('utf-8')
    else:
        content_bytes = content
    
    return st.download_button(
        label=button_text,
        data=content_bytes,
        file_name=filename,
        mime=mime_type,
        help=f"Download {filename} ({len(content_bytes)} bytes)"
    )

def display_processing_summary(processing_results: Dict[str, Any]):
    """Display comprehensive processing summary"""
    
    st.subheader("🎯 Processing Summary")
    
    if 'success' in processing_results:
        if processing_results['success']:
            st.success("✅ Processing completed successfully!")
        else:
            st.error("❌ Processing encountered errors")
    
    # Display metrics
    if 'metrics' in processing_results:
        metrics = processing_results['metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'processing_time' in metrics:
                st.metric("Processing Time", f"{metrics['processing_time']:.1f}s")
        
        with col2:
            if 'items_processed' in metrics:
                st.metric("Items Processed", metrics['items_processed'])
        
        with col3:
            if 'success_rate' in metrics:
                st.metric("Success Rate", f"{metrics['success_rate']:.1f}%")
        
        with col4:
            if 'total_size' in metrics:
                size_mb = metrics['total_size'] / (1024 * 1024)
                st.metric("Total Size", f"{size_mb:.1f} MB")
    
    # Display warnings and errors
    if 'warnings' in processing_results and processing_results['warnings']:
        st.warning("⚠️ Warnings:")
        for warning in processing_results['warnings']:
            st.write(f"• {warning}")
    
    if 'errors' in processing_results and processing_results['errors']:
        st.error("❌ Errors:")
        for error in processing_results['errors']:
            st.write(f"• {error}")

# Advanced session state management

def ensure_enhanced_session_state():
    """Enhanced session state initialization"""
    default_values = {
        'textbook_processed': False,
        'knowledge_base': None,
        'session_id': hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8],
        'user_preferences': {
            'difficulty_level': 'intermediate',
            'learning_style': 'visual',
            'preferred_content_length': 'medium'
        },
        'conversation_count': 0,
        'learning_progress': {
            'topics_studied': [],
            'quizzes_taken': 0,
            'videos_generated': 0,
            'total_study_time': 0
        },
        'cache_manager': AdvancedCacheManager(),
        'processing_stats': {
            'files_processed': 0,
            'total_processing_time': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    }
    
    for key, default_value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Initialize when module is imported
try:
    setup_directories()
    logger.info("✅ IntelliLearn AI helpers initialized successfully")
except Exception as e:
    logger.error(f"❌ Helper initialization failed: {e}")

# Export enhanced classes for external use
__all__ = [
    'EnhancedDirectoryManager',
    'AdvancedFileValidator', 
    'IntelligentTextProcessor',
    'AdvancedCacheManager',
    'AdvancedResponseFormatter',
    'KnowledgeBaseValidator',
    'TextAnalyzer',
    # Legacy functions
    'setup_directories',
    'validate_file_type',
    'calculate_file_hash',
    'clean_text',
    'chunk_text',
    'extract_key_terms',
    'format_response_with_citations',
    'validate_knowledge_base',
    # Enhanced functions
    'create_learning_progress_tracker',
    'display_knowledge_base_statistics',
    'estimate_reading_time_enhanced',
    'create_download_button_enhanced',
    'display_processing_summary',
    'ensure_enhanced_session_state'
]
