import re
import os
import fitz  # PyMuPDF
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import streamlit as st
from config import Config
import requests
import time
import json
import random
from collections import Counter
import torch 

# Enhanced imports for open-source models
try:
    from .enhanced_chapter_detection_v2 import SuperiorChapterDetector
    from .multimodal_processor import OpenSourceMultimodalProcessor
    from .enhanced_nlp_processor import EnhancedNLPProcessor
    ENHANCED_MODULES_AVAILABLE = True
except ImportError:
    try:
        # Fallback to original enhanced chapter detector
        from .enhanced_chapter_detection import EnhancedChapterDetector
        ENHANCED_MODULES_AVAILABLE = False
    except ImportError:
        # If no enhanced modules, create a basic fallback
        class EnhancedChapterDetector:
            def detect_chapters_enhanced(self, doc):
                return self._basic_chapter_detection(doc)
            
            def _basic_chapter_detection(self, doc):
                chapters = {}
                for page in doc:
                    text = page.get_text()
                    if len(text) > 100:
                        chapters[f"Chapter {len(chapters) + 1}"] = {
                            'content': text,
                            'type': 'content',
                            'word_count': len(text.split()),
                            'estimated_reading_time': len(text.split()) // 200
                        }
                return chapters
        
        ENHANCED_MODULES_AVAILABLE = False

class AdvancedTextProcessor:
    def __init__(self):
        """Initialize enhanced text processor with all available models"""
        self.config = Config()
        
        # Initialize embedding model with fallback
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            st.success("✅ Sentence Transformers loaded successfully")
        except Exception as e:
            st.warning(f"⚠️ Sentence Transformers not available: {e}")
            self.embedding_model = None
        
        # Initialize enhanced chapter detector
        if ENHANCED_MODULES_AVAILABLE:
            try:
                self.chapter_detector = SuperiorChapterDetector()
                st.success("✅ Superior Chapter Detector initialized")
            except Exception as e:
                st.warning(f"Superior detector failed: {e}, using fallback")
                self.chapter_detector = EnhancedChapterDetector()
        else:
            self.chapter_detector = EnhancedChapterDetector()
        
        # Initialize enhanced NLP processor
        self.enhanced_nlp = None
        if ENHANCED_MODULES_AVAILABLE:
            try:
                self.enhanced_nlp = EnhancedNLPProcessor()
                st.success("✅ Enhanced NLP Processor initialized")
            except Exception as e:
                st.warning(f"Enhanced NLP not available: {e}")
        
        # Initialize multimodal processor
        self.multimodal_processor = None
        if ENHANCED_MODULES_AVAILABLE:
            try:
                self.multimodal_processor = OpenSourceMultimodalProcessor()
                st.success("✅ Multimodal Processor initialized")
            except Exception as e:
                st.warning(f"Multimodal processor not available: {e}")
        
        # Load spaCy model with enhanced fallback
        self.nlp = None
        try:
            self.nlp = spacy.load("en_core_web_sm")
            st.success("✅ spaCy NLP model loaded")
        except OSError:
            st.warning("📥 Installing spaCy model...")
            try:
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                self.nlp = spacy.load("en_core_web_sm")
                st.success("✅ spaCy model installed and loaded")
            except Exception as e:
                st.warning(f"⚠️ spaCy not available: {e}. Using basic text processing")
                self.nlp = None
    
    def process_textbook_with_enhanced_detection(self, file_path, enhanced_detector=None, difficulty_level='intermediate'):
        """NEW: Enhanced textbook processing with superior chapter detection and multimodal analysis"""
        try:
            st.info(f"🚀 Starting enhanced textbook processing for {difficulty_level} level...")
            
            # Open document
            doc = fitz.open(file_path)
            
            # Use enhanced chapter detection
            detector = enhanced_detector or self.chapter_detector
            
            if ENHANCED_MODULES_AVAILABLE and hasattr(detector, 'detect_chapters_enhanced'):
                st.info("🧠 Using superior chapter detection with 15+ patterns...")
                chapters = detector.detect_chapters_enhanced(doc)
            else:
                st.info("📖 Using standard enhanced chapter detection...")
                chapters = detector.detect_chapters_enhanced(doc)
            
            # Enhanced processing with difficulty adaptation
            structured_content = {
                'metadata': {
                    'total_pages': doc.page_count,
                    'processing_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'chapters_found': len(chapters),
                    'processing_method': 'Enhanced AI Analysis with Superior Chapter Detection',
                    'difficulty_level': difficulty_level,
                    'enhanced_features': {
                        'superior_detection': ENHANCED_MODULES_AVAILABLE,
                        'multimodal_analysis': bool(self.multimodal_processor),
                        'advanced_nlp': bool(self.enhanced_nlp),
                        'embedding_model': bool(self.embedding_model)
                    }
                },
                'chapters': {}
            }
            
            # Display detection statistics
            if hasattr(detector, 'get_detection_statistics'):
                detection_stats = detector.get_detection_statistics(chapters)
                st.success(f"📊 Detected {detection_stats.get('total_chapters', len(chapters))} chapters with {detection_stats.get('total_words', 0):,} total words")
            else:
                total_words = sum(ch.get('word_count', 0) for ch in chapters.values())
                st.success(f"📊 Detected {len(chapters)} chapters with {total_words:,} total words")
            
            # Process chapters with enhanced features and rate limiting
            return self._process_chapters_enhanced(chapters, structured_content, difficulty_level, doc)
            
        except Exception as e:
            st.error(f"❌ Enhanced processing failed: {str(e)}")
            # Fallback to original method
            return self.process_textbook_with_full_hierarchy(file_path)
    
    def _process_chapters_enhanced(self, chapters, structured_content, difficulty_level, doc):
        """Process chapters with enhanced features and difficulty adaptation"""
        
        chapter_items = list(chapters.items())
        batch_size = 3  # Smaller batches for better rate limiting
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(0, len(chapter_items), batch_size):
            batch = chapter_items[i:i+batch_size]
            
            batch_num = i // batch_size + 1
            total_batches = (len(chapter_items) + batch_size - 1) // batch_size
            
            status_text.text(f"🔄 Processing batch {batch_num} of {total_batches}")
            progress_bar.progress(i / len(chapter_items))
            
            for j, (chapter_name, chapter_data) in enumerate(batch):
                st.write(f"📚 Processing: {chapter_name}")
                
                # Access content from enhanced chapter data
                chapter_content = chapter_data.get('content', '')
                
                if not chapter_content:
                    st.warning(f"⚠️ No content found for {chapter_name}")
                    continue
                
                # Enhanced topic extraction with difficulty adaptation
                topics = self._extract_topics_with_enhanced_ai(
                    chapter_content, chapter_name, difficulty_level
                )
                
                # Enhanced concept extraction
                concepts = self._extract_concepts_with_enhanced_nlp(
                    chapter_content, difficulty_level
                )
                
                # Enhanced example extraction
                examples = self._find_enhanced_real_world_examples(
                    chapter_content, difficulty_level
                )
                
                # Advanced difficulty assessment
                difficulty_assessment = self._assess_content_difficulty_advanced(
                    chapter_content, difficulty_level
                )
                
                # Enhanced key points extraction
                key_points = self._extract_key_points_with_enhanced_ai(
                    chapter_content, difficulty_level
                )
                
                # Generate difficulty-adapted quizzes
                quizzes = self._generate_enhanced_chapter_quizzes(
                    topics, chapter_name, difficulty_level
                )
                
                # Extract learning objectives with difficulty context
                learning_objectives = self._extract_enhanced_learning_objectives(
                    chapter_content, difficulty_level
                )
                
                # Identify prerequisites with difficulty awareness
                prerequisites = self._identify_enhanced_prerequisites(
                    chapter_name, topics, difficulty_level
                )
                
                # Merge all enhanced data
                structured_content['chapters'][chapter_name] = {
                    **chapter_data,  # Include all original enhanced detection data
                    'topics': topics,
                    'concepts': concepts,
                    'examples': examples,
                    'difficulty_assessment': difficulty_assessment,
                    'key_points': key_points,
                    'quizzes': quizzes,
                    'learning_objectives': learning_objectives,
                    'prerequisites': prerequisites,
                    'processing_metadata': {
                        'difficulty_level': difficulty_level,
                        'enhanced_features_used': {
                            'advanced_nlp': bool(self.enhanced_nlp),
                            'embedding_model': bool(self.embedding_model),
                            'spacy_processing': bool(self.nlp)
                        },
                        'processing_time': time.strftime("%H:%M:%S")
                    }
                }
                
                # Small delay between chapters
                if j < len(batch) - 1:
                    time.sleep(1)
            
            # Update progress
            progress_bar.progress((i + len(batch)) / len(chapter_items))
            
            # Longer delay between batches for rate limiting
            if i + batch_size < len(chapter_items):
                wait_time = 45  # Reduced wait time with smaller batches
                status_text.text(f"⏳ Waiting {wait_time}s before next batch...")
                time.sleep(wait_time)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Enhanced processing complete!")
        
        doc.close()
        return structured_content
    
    def process_textbook_with_full_hierarchy(self, file_path):
        """Original enhanced textbook processing method (kept for backward compatibility)"""
        try:
            # Extract structured content
            doc = fitz.open(file_path)
            
            # Use comprehensive enhanced chapter detection
            chapters = self.chapter_detector.detect_chapters_enhanced(doc)
            
            structured_content = {
                'metadata': {
                    'total_pages': doc.page_count,
                    'processing_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'chapters_found': len(chapters),
                    'processing_method': 'Enhanced AI Analysis with Advanced Chapter Detection'
                },
                'chapters': {}
            }
            
            # Display detection statistics
            if hasattr(self.chapter_detector, 'get_detection_statistics'):
                detection_stats = self.chapter_detector.get_detection_statistics(chapters)
                st.info(f"📊 Detected {detection_stats.get('total_chapters', 0)} chapters with {detection_stats.get('total_words', 0)} total words")
            
            # Process chapters with rate limiting
            chapter_items = list(chapters.items())
            batch_size = 5  # Process 5 chapters at a time to respect API limits
            
            for i in range(0, len(chapter_items), batch_size):
                batch = chapter_items[i:i+batch_size]
                
                if len(chapter_items) > batch_size:
                    st.info(f"📦 Processing batch {i//batch_size + 1} of {(len(chapter_items) + batch_size - 1)//batch_size}")
                
                for chapter_name, chapter_data in batch:
                    st.write(f"🔍 Processing chapter: {chapter_name}")
                    
                    # Access content from enhanced chapter data
                    chapter_content = chapter_data.get('content', '')
                    
                    # Add small delay between requests to respect rate limits
                    if i > 0:
                        time.sleep(2)
                    
                    # Extract enhanced topics with AI
                    topics = self._extract_topics_with_ai_enhanced(chapter_content, chapter_name)
                    
                    # Generate intelligent quizzes
                    quizzes = self._generate_chapter_quizzes(topics, chapter_name)
                    
                    # Extract concepts with enhanced NLP
                    concepts = self._extract_chapter_concepts_enhanced(chapter_content)
                    
                    # Find real-world examples
                    examples = self._find_real_world_examples_enhanced(chapter_content)
                    
                    # Assess content difficulty with advanced metrics
                    difficulty = self._assess_content_difficulty_enhanced(chapter_content)
                    
                    # Extract key points with AI
                    key_points = self._extract_key_points_ai_enhanced(chapter_content)
                    
                    # Merge enhanced chapter data with AI analysis
                    structured_content['chapters'][chapter_name] = {
                        **chapter_data,  # Include all enhanced detection data
                        'topics': topics,
                        'concepts': concepts,
                        'examples': examples,
                        'difficulty': difficulty,
                        'key_points': key_points,
                        'quizzes': quizzes,
                        'learning_objectives': self._extract_learning_objectives(chapter_content),
                        'prerequisites': self._identify_prerequisites(chapter_name, topics)
                    }
                
                # Longer delay between batches to respect rate limits
                if i + batch_size < len(chapter_items):
                    st.info("⏳ Waiting 60 seconds before next batch to respect rate limits...")
                    time.sleep(60)
            
            doc.close()
            return structured_content
            
        except Exception as e:
            st.error(f"Error processing textbook: {str(e)}")
            return None

    def _extract_topics_with_enhanced_ai(self, chapter_content, chapter_name, difficulty_level='intermediate'):
        """NEW: Enhanced AI topic extraction with difficulty adaptation"""
        try:
            # Use enhanced NLP processor if available
            if self.enhanced_nlp:
                topics_data = self.enhanced_nlp.extract_educational_topics_advanced(
                    [chapter_content], difficulty_level
                )
                
                if topics_data:
                    # Convert to expected format
                    formatted_topics = {}
                    for topic in topics_data[:6]:  # Limit to 6 topics
                        topic_name = topic.get('name', f"Topic {len(formatted_topics) + 1}")
                        formatted_topics[topic_name] = {
                            'key_points': topic.get('keywords', [])[:5],
                            'difficulty': topic.get('difficulty_level', difficulty_level).title(),
                            'summary': f"Educational topic covering {', '.join(topic.get('keywords', [])[:3])}",
                            'learning_objectives': topic.get('learning_objectives', [])[:3],
                            'estimated_time': f"{topic.get('related_concepts', 3) * 5} minutes",
                            'content': chapter_content[:500],
                            'educational_category': topic.get('educational_category', 'general'),
                            'confidence_score': topic.get('confidence_score', 0.7)
                        }
                    
                    if formatted_topics:
                        st.success(f"✅ Enhanced NLP extracted {len(formatted_topics)} topics")
                        return formatted_topics
            
            # Fallback to AI-enhanced extraction
            return self._extract_topics_with_ai_enhanced(chapter_content, chapter_name, difficulty_level)
            
        except Exception as e:
            st.warning(f"Enhanced topic extraction failed: {e}")
            return self._extract_topics_with_ai_enhanced(chapter_content, chapter_name, difficulty_level)

    def _extract_topics_with_ai_enhanced(self, chapter_content, chapter_name, difficulty_level='intermediate'):
        """Enhanced AI topic extraction with difficulty adaptation"""
        try:
            # Difficulty-adapted system prompts
            difficulty_prompts = {
                'beginner': f"""You are an expert educational content analyzer for BEGINNER level learners. 

Analyze this chapter and extract 4-6 main topics suitable for beginners.

BEGINNER REQUIREMENTS:
- Use simple, clear topic names
- Focus on fundamental concepts
- Provide basic, easy-to-understand explanations
- Include encouraging learning objectives
- Estimate longer learning times for comprehension

FORMAT AS JSON:
{{
  "topic_name": {{
    "key_points": ["simple point 1", "basic point 2", "fundamental point 3"],
    "difficulty": "Beginner",
    "summary": "Simple explanation using basic vocabulary",
    "learning_objectives": ["Understand basic...", "Identify simple..."],
    "estimated_time": "20 minutes"
  }}
}}

CHAPTER: {chapter_name}""",

                'intermediate': f"""You are an expert educational content analyzer for INTERMEDIATE level learners.

Analyze this chapter and extract 5-7 main topics suitable for intermediate learners.

INTERMEDIATE REQUIREMENTS:
- Use clear but moderately technical topic names
- Balance detail with accessibility
- Include practical applications
- Provide standard learning objectives
- Estimate moderate learning times

FORMAT AS JSON:
{{
  "topic_name": {{
    "key_points": ["detailed point 1", "analytical point 2", "practical point 3"],
    "difficulty": "Intermediate", 
    "summary": "Clear explanation with moderate technical detail",
    "learning_objectives": ["Analyze...", "Apply...", "Evaluate..."],
    "estimated_time": "15 minutes"
  }}
}}

CHAPTER: {chapter_name}""",

                'advanced': f"""You are an expert educational content analyzer for ADVANCED level learners.

Analyze this chapter and extract 6-8 main topics suitable for advanced learners.

ADVANCED REQUIREMENTS:
- Use precise technical terminology
- Focus on complex relationships and implications
- Include advanced analytical perspectives
- Provide challenging learning objectives
- Estimate efficient learning times for experts

FORMAT AS JSON:
{{
  "topic_name": {{
    "key_points": ["complex analysis 1", "technical implication 2", "advanced application 3"],
    "difficulty": "Advanced",
    "summary": "Comprehensive technical explanation with advanced concepts",
    "learning_objectives": ["Synthesize...", "Evaluate critically...", "Create solutions..."],
    "estimated_time": "12 minutes"
  }}
}}

CHAPTER: {chapter_name}"""
            }

            system_prompt = difficulty_prompts.get(difficulty_level, difficulty_prompts['intermediate'])

            # Optimize content for API call with difficulty consideration
            if difficulty_level == 'beginner':
                content_sample = self._optimize_content_for_api(chapter_content, 1000)  # Less content for beginners
            elif difficulty_level == 'advanced':
                content_sample = self._optimize_content_for_api(chapter_content, 1500)  # More content for advanced
            else:
                content_sample = self._optimize_content_for_api(chapter_content, 1200)
            
            response = self._call_groq_api_with_retry(content_sample, system_prompt)
            
            try:
                topics = json.loads(response)
                
                # Enhance topics with additional difficulty-adapted content
                for topic_name, topic_data in topics.items():
                    topic_data['content'] = self._extract_topic_content_enhanced(chapter_content, topic_name)
                    topic_data['keywords'] = self._extract_topic_keywords(topic_name, topic_data)
                    topic_data['difficulty_adapted'] = True
                    topic_data['target_level'] = difficulty_level
                
                return topics
            except json.JSONDecodeError:
                return self._basic_topic_extraction_enhanced(chapter_content, difficulty_level)
                
        except Exception as e:
            st.warning(f"AI topic extraction failed: {e}. Using fallback method.")
            return self._basic_topic_extraction_enhanced(chapter_content, difficulty_level)

    def _extract_concepts_with_enhanced_nlp(self, chapter_content, difficulty_level='intermediate'):
        """NEW: Extract concepts using enhanced NLP with difficulty adaptation"""
        
        if self.enhanced_nlp:
            try:
                # Use enhanced NLP processor
                concepts = self.enhanced_nlp.extract_educational_concepts_advanced(
                    chapter_content, subject="general"
                )
                
                # Filter concepts based on difficulty level
                if difficulty_level == 'beginner':
                    # Prefer shorter, simpler concepts
                    filtered_concepts = [c for c in concepts if len(c.split()) <= 2 and len(c) <= 15]
                elif difficulty_level == 'advanced':
                    # Include more complex concepts
                    filtered_concepts = concepts[:20]
                else:  # intermediate
                    # Balanced selection
                    filtered_concepts = [c for c in concepts if len(c.split()) <= 3][:15]
                
                if filtered_concepts:
                    st.success(f"✅ Enhanced NLP extracted {len(filtered_concepts)} concepts for {difficulty_level} level")
                    return filtered_concepts
                    
            except Exception as e:
                st.warning(f"Enhanced concept extraction failed: {e}")
        
        # Fallback to original method
        return self._extract_chapter_concepts_enhanced(chapter_content)

    def _find_enhanced_real_world_examples(self, chapter_content, difficulty_level='intermediate'):
        """NEW: Enhanced real-world example extraction with difficulty adaptation"""
        
        # Difficulty-adapted example patterns
        if difficulty_level == 'beginner':
            example_patterns = [
                r'(?i)simple example[,:]?\s*([^.!?]*[.!?])',
                r'(?i)for instance[,:]?\s*([^.!?]*[.!?])',
                r'(?i)like\s+([^.!?]*[.!?])',
                r'(?i)such as\s+([^.!?]*[.!?])'
            ]
        elif difficulty_level == 'advanced':
            example_patterns = [
                r'(?i)case study[,:]?\s*([^.!?]*[.!?])',
                r'(?i)research shows[,:]?\s*([^.!?]*[.!?])',
                r'(?i)in practice[,:]?\s*([^.!?]*[.!?])',
                r'(?i)real[\\-\\s]world application[,:]?\s*([^.!?]*[.!?])',
                r'(?i)empirical evidence[,:]?\s*([^.!?]*[.!?])'
            ]
        else:  # intermediate
            example_patterns = [
                r'(?i)for example[,:]?\s*([^.!?]*[.!?])',
                r'(?i)example\s*\d*[:\\-]?\s*([^.!?]*[.!?])',
                r'(?i)consider\s+([^.!?]*[.!?])',
                r'(?i)such as\s+([^.!?]*[.!?])',
                r'(?i)instance[,:]?\s*([^.!?]*[.!?])',
                r'(?i)in practice[,:]?\s*([^.!?]*[.!?])'
            ]
        
        examples = []
        for pattern in example_patterns:
            matches = re.findall(pattern, chapter_content)
            for match in matches:
                cleaned_match = match.strip()
                
                # Difficulty-adapted filtering
                if difficulty_level == 'beginner':
                    # Prefer shorter, simpler examples
                    if 20 <= len(cleaned_match) <= 150 and len(cleaned_match.split()) <= 25:
                        examples.append(cleaned_match)
                elif difficulty_level == 'advanced':
                    # Allow longer, more complex examples
                    if 50 <= len(cleaned_match) <= 400:
                        examples.append(cleaned_match)
                else:  # intermediate
                    # Balanced length examples
                    if 30 <= len(cleaned_match) <= 250:
                        examples.append(cleaned_match)
        
        # Remove duplicates and limit based on difficulty
        seen = set()
        unique_examples = []
        max_examples = {'beginner': 5, 'intermediate': 8, 'advanced': 12}
        
        for example in examples:
            if example.lower() not in seen and len(unique_examples) < max_examples.get(difficulty_level, 8):
                seen.add(example.lower())
                unique_examples.append(example)
        
        return unique_examples

    def _assess_content_difficulty_advanced(self, content, target_difficulty='intermediate'):
        """NEW: Advanced difficulty assessment with target level consideration"""
        
        if self.enhanced_nlp:
            try:
                # Use enhanced NLP processor for advanced assessment
                assessment = self.enhanced_nlp.assess_text_difficulty_enhanced(content)
                
                # Compare with target difficulty and provide recommendations
                detected_level = assessment.get('level', 'intermediate')
                
                return {
                    'detected_level': detected_level,
                    'target_level': target_difficulty,
                    'match': detected_level == target_difficulty,
                    'scores': assessment.get('scores', {}),
                    'metrics': assessment.get('metrics', {}),
                    'recommendations': assessment.get('recommendations', []),
                    'content_adaptation_needed': detected_level != target_difficulty
                }
                
            except Exception as e:
                st.warning(f"Advanced difficulty assessment failed: {e}")
        
        # Fallback to original method
        basic_difficulty = self._assess_content_difficulty_enhanced(content)
        
        return {
            'detected_level': basic_difficulty,
            'target_level': target_difficulty,
            'match': basic_difficulty.lower() == target_difficulty.lower(),
            'scores': {},
            'metrics': {},
            'recommendations': [],
            'content_adaptation_needed': basic_difficulty.lower() != target_difficulty.lower()
        }

    def _extract_key_points_with_enhanced_ai(self, content, difficulty_level='intermediate'):
        """NEW: Enhanced key points extraction with difficulty adaptation"""
        
        # Difficulty-adapted prompts for key points
        difficulty_prompts = {
            'beginner': """Extract 4-6 key learning points from this educational content for BEGINNER learners.

BEGINNER REQUIREMENTS:
- Use simple, clear language
- Focus on fundamental concepts
- Make each point easy to understand
- Include encouraging language
- Keep points concise but complete

FORMAT: Return as JSON array of strings
Example: ["Basic concept: Students will understand...", "Simple principle: ...", "Important idea: ..."]

Content:""",

            'intermediate': """Extract 5-7 key learning points from this educational content for INTERMEDIATE learners.

INTERMEDIATE REQUIREMENTS:
- Use clear, moderately technical language
- Balance detail with accessibility
- Focus on important concepts and applications
- Include analytical elements
- Ensure points build on each other

FORMAT: Return as JSON array of strings
Example: ["Key principle: Students will analyze...", "Important application: ...", "Critical concept: ..."]

Content:""",

            'advanced': """Extract 6-8 key learning points from this educational content for ADVANCED learners.

ADVANCED REQUIREMENTS:
- Use precise technical terminology
- Focus on complex relationships and implications
- Include analytical and synthesis elements
- Assume strong foundational knowledge
- Emphasize critical thinking aspects

FORMAT: Return as JSON array of strings
Example: ["Complex analysis: Advanced students will synthesize...", "Critical evaluation: ...", "Technical mastery: ..."]

Content:"""
        }

        try:
            system_prompt = difficulty_prompts.get(difficulty_level, difficulty_prompts['intermediate'])
            
            # Adjust content length based on difficulty
            content_limits = {'beginner': 600, 'intermediate': 800, 'advanced': 1000}
            content_sample = self._optimize_content_for_api(content, content_limits.get(difficulty_level, 800))
            
            response = self._call_groq_api_with_retry(content_sample, system_prompt)
            
            try:
                points = json.loads(response)
                if isinstance(points, list) and len(points) > 0:
                    # Filter points based on difficulty level
                    filtered_points = []
                    max_points = {'beginner': 6, 'intermediate': 7, 'advanced': 8}
                    
                    for point in points[:max_points.get(difficulty_level, 7)]:
                        if isinstance(point, str) and len(point.strip()) > 10:
                            filtered_points.append(point.strip())
                    
                    return filtered_points
            except json.JSONDecodeError:
                pass
            
            # Fallback to regex parsing with difficulty adaptation
            points = []
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith(('-', '•', '*')) or re.match(r'^\d+\.', line):
                    point = re.sub(r'^[-•*\d\.\s]+', '', line).strip()
                    if len(point) > 15:
                        points.append(point)
            
            max_points = {'beginner': 6, 'intermediate': 7, 'advanced': 8}
            return points[:max_points.get(difficulty_level, 7)] if points else self._extract_key_points_fallback(content, difficulty_level)
            
        except Exception as e:
            return self._extract_key_points_fallback(content, difficulty_level)

    def _generate_enhanced_chapter_quizzes(self, topics, chapter_name, difficulty_level='intermediate'):
        """NEW: Generate enhanced quizzes with difficulty adaptation"""
        all_quizzes = {}
        
        for topic_name, topic_data in topics.items():
            if len(topic_data.get('key_points', [])) >= 2:
                # Generate difficulty-adapted quiz
                quiz = self.generate_intelligent_quizzes_enhanced(
                    topic_data, 
                    difficulty_level, 
                    num_questions=self._get_questions_per_difficulty(difficulty_level)
                )
                
                if quiz and quiz.get('questions'):
                    all_quizzes[topic_name] = quiz
        
        return all_quizzes

    def _get_questions_per_difficulty(self, difficulty_level):
        """Get number of questions based on difficulty level"""
        question_counts = {
            'beginner': 3,      # Fewer questions, more manageable
            'intermediate': 5,   # Standard amount
            'advanced': 7       # More questions for comprehensive assessment
        }
        return question_counts.get(difficulty_level, 5)

    def generate_intelligent_quizzes_enhanced(self, topic_data, difficulty_level="intermediate", num_questions=5):
        """NEW: Generate enhanced quizzes with difficulty-specific requirements"""
        
        # Difficulty-adapted quiz generation prompts
        difficulty_requirements = {
            'beginner': {
                'question_types': '70% Multiple Choice (3 options each), 20% True/False, 10% Short Answer',
                'complexity': 'Simple, clear questions focusing on basic understanding',
                'language': 'Use simple vocabulary and short sentences',
                'time': 'Allow extra time for comprehension'
            },
            'intermediate': {
                'question_types': '60% Multiple Choice (4 options each), 20% True/False, 20% Short Answer',
                'complexity': 'Balanced questions testing understanding and application',
                'language': 'Use clear, moderately technical vocabulary',
                'time': 'Standard time allocation'
            },
            'advanced': {
                'question_types': '50% Multiple Choice (5 options each), 15% True/False, 35% Short Answer',
                'complexity': 'Complex questions requiring analysis and synthesis',
                'language': 'Use technical terminology and complex concepts',
                'time': 'Efficient time allocation for expert learners'
            }
        }

        requirements = difficulty_requirements.get(difficulty_level, difficulty_requirements['intermediate'])

        system_prompt = f"""You are an expert educational assessment creator for {difficulty_level.upper()} level learners.

Generate {num_questions} quiz questions from the provided content.

DIFFICULTY LEVEL: {difficulty_level.upper()}
REQUIREMENTS:
- Question Distribution: {requirements['question_types']}
- Complexity: {requirements['complexity']}
- Language: {requirements['language']}
- Time Consideration: {requirements['time']}

ADDITIONAL REQUIREMENTS:
- Include correct answers and detailed explanations adapted for {difficulty_level} learners
- Make distractors (wrong options) plausible but clearly incorrect
- Questions should test {self._get_bloom_focus(difficulty_level)}

FORMAT AS JSON:
{{
  "quiz_metadata": {{
    "topic": "{topic_data.get('summary', 'Educational Topic')}",
    "difficulty": "{difficulty_level}",
    "total_questions": {num_questions},
    "estimated_time": "{self._get_quiz_time(difficulty_level, num_questions)}",
    "target_level": "{difficulty_level}",
    "bloom_taxonomy_focus": "{self._get_bloom_focus(difficulty_level)}"
  }},
  "questions": [
    {{
      "id": 1,
      "type": "multiple_choice",
      "question": "Question text adapted for {difficulty_level} level?",
      "options": ["Option A", "Option B", "Option C"{', "Option D"' if difficulty_level != 'beginner' else ''}{', "Option E"' if difficulty_level == 'advanced' else ''}],
      "correct_answer": 0,
      "explanation": "Detailed explanation adapted for {difficulty_level} learners",
      "difficulty": "{difficulty_level}",
      "bloom_level": "{self._get_bloom_focus(difficulty_level)}",
      "topic_area": "specific topic"
    }}
  ]
}}

CONTENT TO ANALYZE:
Topic: {topic_data.get('summary', 'Educational Topic')}
Key Points: {', '.join(topic_data.get('key_points', [])[:5])}
Content: {topic_data.get('content', '')[:800]}"""

        try:
            response = self._call_groq_api_with_retry(topic_data.get('content', ''), system_prompt)
            
            try:
                quiz_data = json.loads(response)
                return self._validate_enhanced_quiz_data(quiz_data, difficulty_level)
            except json.JSONDecodeError:
                return self._generate_fallback_quiz_enhanced(topic_data, difficulty_level, num_questions)
                
        except Exception as e:
            st.warning(f"Enhanced quiz generation failed: {e}. Using fallback method.")
            return self._generate_fallback_quiz_enhanced(topic_data, difficulty_level, num_questions)

    def _get_bloom_focus(self, difficulty_level):
        """Get Bloom's taxonomy focus for difficulty level"""
        bloom_focus = {
            'beginner': 'Remember and Understand',
            'intermediate': 'Understand and Apply', 
            'advanced': 'Analyze, Evaluate, and Create'
        }
        return bloom_focus.get(difficulty_level, 'Understand and Apply')

    def _get_quiz_time(self, difficulty_level, num_questions):
        """Get estimated quiz time based on difficulty and question count"""
        time_per_question = {
            'beginner': 3,      # 3 minutes per question
            'intermediate': 2,  # 2 minutes per question
            'advanced': 2.5     # 2.5 minutes per question (complex questions)
        }
        
        minutes = num_questions * time_per_question.get(difficulty_level, 2)
        return f"{int(minutes)} minutes"

    def _extract_enhanced_learning_objectives(self, content, difficulty_level='intermediate'):
        """NEW: Extract learning objectives adapted to difficulty level"""
        
        difficulty_objective_prompts = {
            'beginner': """Extract 3-4 clear learning objectives for BEGINNER learners from this content.

BEGINNER REQUIREMENTS:
- Start with basic action verbs (understand, identify, describe, list)
- Use simple, encouraging language
- Focus on fundamental concepts
- Make objectives achievable and confidence-building

FORMAT: Return as JSON array
Example: ["Understand the basic concept of...", "Identify simple examples of...", "Describe the main features of..."]

Content:""",

            'intermediate': """Extract 4-5 clear learning objectives for INTERMEDIATE learners from this content.

INTERMEDIATE REQUIREMENTS:
- Use moderate action verbs (analyze, apply, compare, explain)
- Balance specificity with achievability
- Include both theoretical and practical elements
- Build on existing knowledge

FORMAT: Return as JSON array
Example: ["Analyze the relationship between...", "Apply the principles of...", "Compare different approaches to..."]

Content:""",

            'advanced': """Extract 5-6 challenging learning objectives for ADVANCED learners from this content.

ADVANCED REQUIREMENTS:
- Use high-level action verbs (synthesize, evaluate, create, critique)
- Focus on complex thinking and analysis
- Include synthesis and evaluation elements
- Assume strong foundational knowledge

FORMAT: Return as JSON array
Example: ["Synthesize multiple perspectives on...", "Evaluate the effectiveness of...", "Create innovative solutions for..."]

Content:"""
        }

        try:
            system_prompt = difficulty_objective_prompts.get(difficulty_level, difficulty_objective_prompts['intermediate'])
            content_sample = content[:800]
            response = self._call_groq_api_with_retry(content_sample, system_prompt)
            
            try:
                objectives = json.loads(response)
                if isinstance(objectives, list):
                    return objectives
            except json.JSONDecodeError:
                pass
                
            return self._extract_objectives_fallback(content, difficulty_level)
                
        except Exception as e:
            return self._extract_objectives_fallback(content, difficulty_level)

    def _identify_enhanced_prerequisites(self, chapter_name, topics, difficulty_level='intermediate'):
        """NEW: Identify prerequisites with difficulty level consideration"""
        
        prerequisites = []
        chapter_lower = chapter_name.lower()
        
        # Difficulty-adapted prerequisite identification
        if difficulty_level == 'beginner':
            # Basic prerequisites for beginners
            if any(word in chapter_lower for word in ['introduction', 'basic', 'fundamental']):
                prerequisites.append('Basic reading comprehension')
            if any(word in chapter_lower for word in ['math', 'calculation', 'number']):
                prerequisites.append('Basic arithmetic')
        
        elif difficulty_level == 'intermediate':
            # Standard prerequisites
            if any(word in chapter_lower for word in ['advanced', 'complex', 'analysis']):
                prerequisites.append('Understanding of basic concepts')
            if any(word in chapter_lower for word in ['chemistry', 'physics', 'biology']):
                prerequisites.append('Basic scientific knowledge')
        
        else:  # advanced
            # Advanced prerequisites
            if any(word in chapter_lower for word in ['theory', 'analysis', 'research']):
                prerequisites.append('Strong foundation in previous chapters')
                prerequisites.append('Analytical thinking skills')
            if any(word in chapter_lower for word in ['methodology', 'framework', 'model']):
                prerequisites.append('Advanced conceptual understanding')
        
        # Topic-based prerequisites
        topic_keywords = []
        for topic_data in topics.values():
            topic_keywords.extend(topic_data.get('key_points', [])[:2])
        
        if any('statistical' in keyword.lower() for keyword in topic_keywords):
            prerequisites.append('Basic statistics knowledge')
        if any('technical' in keyword.lower() for keyword in topic_keywords):
            prerequisites.append('Technical vocabulary familiarity')
        
        return prerequisites[:4]  # Limit to 4 prerequisites

    def _optimize_content_for_api(self, content, max_chars):
        """Enhanced content optimization for API calls"""
        if len(content) <= max_chars:
            return content
        
        # Split into sentences and select most informative ones
        sentences = content.split('.')
        
        # Score sentences based on educational value
        scored_sentences = []
        for sentence in sentences:
            if len(sentence.strip()) < 10:
                continue
            
            score = 0
            sentence_lower = sentence.lower()
            
            # Higher score for sentences with educational keywords
            educational_keywords = ['important', 'key', 'main', 'concept', 'principle', 'theory', 'method', 'process']
            for keyword in educational_keywords:
                if keyword in sentence_lower:
                    score += 2
            
            # Higher score for sentences with numbers (often key facts)
            if re.search(r'\d+', sentence):
                score += 1
            
            # Higher score for longer, informative sentences
            score += len(sentence.split()) / 20
            
            scored_sentences.append((sentence, score))
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        selected_content = ""
        for sentence, _ in scored_sentences:
            if len(selected_content + sentence) <= max_chars:
                selected_content += sentence + ". "
            else:
                break
        
        return selected_content.strip()

    def _generate_chapter_quizzes(self, topics, chapter_name):
        """Generate intelligent quizzes for the chapter (original method)"""
        all_quizzes = {}
        
        for topic_name, topic_data in topics.items():
            if len(topic_data.get('key_points', [])) >= 2:  # Only create quizzes for substantial topics
                quiz = self.generate_intelligent_quizzes(topic_data, topic_data.get('difficulty', 'Intermediate'))
                if quiz and quiz.get('questions'):
                    all_quizzes[topic_name] = quiz
        
        return all_quizzes

    def generate_intelligent_quizzes(self, topic_data, difficulty_level="Intermediate", num_questions=5):
        """Generate contextual quizzes from topic content with enhanced validation (original method)"""
        
        system_prompt = f"""You are an expert educational assessment creator. Generate {num_questions} quiz questions from the provided content.

REQUIREMENTS:
- {difficulty_level} difficulty level
- Mix of question types: 
  * 60% Multiple Choice (4 options each)
  * 20% True/False 
  * 20% Short Answer
- Include correct answers and detailed explanations
- Questions should test understanding, not just memorization
- Make distractors (wrong options) plausible but clearly incorrect

FORMAT AS JSON:
{{
  "quiz_metadata": {{
    "topic": "{topic_data.get('summary', 'Educational Topic')}",
    "difficulty": "{difficulty_level}",
    "total_questions": {num_questions},
    "estimated_time": "X minutes"
  }},
  "questions": [
    {{
      "id": 1,
      "type": "multiple_choice",
      "question": "Question text here?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": 0,
      "explanation": "Detailed explanation of why this is correct",
      "difficulty": "{difficulty_level}",
      "topic_area": "specific topic"
    }},
    {{
      "id": 2,
      "type": "true_false",
      "question": "Statement to evaluate",
      "correct_answer": true,
      "explanation": "Why this statement is true/false"
    }},
    {{
      "id": 3,
      "type": "short_answer",
      "question": "Open-ended question?",
      "sample_answer": "Example of a good answer",
      "key_points": ["Point 1", "Point 2", "Point 3"]
    }}
  ]
}}

CONTENT TO ANALYZE:
Topic: {topic_data.get('summary', 'Educational Topic')}
Key Points: {', '.join(topic_data.get('key_points', [])[:5])}
Content: {topic_data.get('content', '')[:800]}"""

        try:
            response = self._call_groq_api_with_retry(topic_data.get('content', ''), system_prompt)
            
            try:
                quiz_data = json.loads(response)
                return self._validate_quiz_data_enhanced(quiz_data)
            except json.JSONDecodeError:
                return self._generate_fallback_quiz_enhanced(topic_data, difficulty_level, num_questions)
                
        except Exception as e:
            st.warning(f"AI quiz generation failed: {e}. Using fallback method.")
            return self._generate_fallback_quiz_enhanced(topic_data, difficulty_level, num_questions)

    # Enhanced validation methods
    def _validate_enhanced_quiz_data(self, quiz_data, difficulty_level):
        """NEW: Enhanced validation with difficulty-specific requirements"""
        
        # Ensure required fields exist
        if 'questions' not in quiz_data:
            quiz_data['questions'] = []
        
        if 'quiz_metadata' not in quiz_data:
            quiz_data['quiz_metadata'] = {
                "topic": "Generated Quiz",
                "difficulty": difficulty_level,
                "total_questions": len(quiz_data['questions']),
                "estimated_time": self._get_quiz_time(difficulty_level, len(quiz_data['questions'])),
                "target_level": difficulty_level
            }
        
        # Validate each question with difficulty considerations
        valid_questions = []
        for i, question in enumerate(quiz_data['questions']):
            if self._is_valid_enhanced_question(question, difficulty_level):
                question['id'] = i + 1
                question = self._enhance_question_quality_advanced(question, difficulty_level)
                valid_questions.append(question)
        
        quiz_data['questions'] = valid_questions
        quiz_data['quiz_metadata']['total_questions'] = len(valid_questions)
        quiz_data['quiz_metadata']['question_types'] = self._analyze_question_types(valid_questions)
        quiz_data['quiz_metadata']['difficulty_validation'] = {
            'target_level': difficulty_level,
            'validated_questions': len(valid_questions),
            'difficulty_adapted': True
        }
        
        return quiz_data

    def _is_valid_enhanced_question(self, question, difficulty_level):
        """NEW: Enhanced question validation with difficulty considerations"""
        
        required_fields = ['type', 'question']
        
        # Check required fields
        for field in required_fields:
            if field not in question or not question[field]:
                return False
        
        # Difficulty-specific length requirements
        min_lengths = {'beginner': 8, 'intermediate': 10, 'advanced': 12}
        if len(question['question']) < min_lengths.get(difficulty_level, 10):
            return False
        
        # Type-specific validation with difficulty considerations
        if question['type'] == 'multiple_choice':
            options = question.get('options', [])
            correct_answer = question.get('correct_answer')
            
            # Difficulty-specific option count requirements
            min_options = {'beginner': 3, 'intermediate': 4, 'advanced': 4}
            expected_options = {'beginner': 3, 'intermediate': 4, 'advanced': 5}
            
            return (len(options) >= min_options.get(difficulty_level, 3) and 
                    isinstance(correct_answer, int) and 
                    0 <= correct_answer < len(options))
        
        elif question['type'] == 'true_false':
            correct_answer = question.get('correct_answer')
            return isinstance(correct_answer, bool)
        
        elif question['type'] == 'short_answer':
            # Advanced levels should have more detailed requirements
            if difficulty_level == 'advanced':
                return 'sample_answer' in question and 'key_points' in question
            return True
        
        return False

    def _enhance_question_quality_advanced(self, question, difficulty_level):
        """NEW: Advanced question enhancement with difficulty adaptation"""
        
        # Add bloom's taxonomy level based on difficulty
        difficulty_bloom_mapping = {
            'beginner': ['Remember', 'Understand'],
            'intermediate': ['Understand', 'Apply', 'Analyze'],
            'advanced': ['Analyze', 'Evaluate', 'Create']
        }
        
        question['blooms_level'] = self._classify_blooms_taxonomy_advanced(
            question['question'], 
            difficulty_bloom_mapping.get(difficulty_level, ['Understand', 'Apply'])
        )
        
        # Add difficulty-specific estimated time
        time_multipliers = {'beginner': 1.5, 'intermediate': 1.0, 'advanced': 1.2}
        base_times = {'multiple_choice': 90, 'true_false': 60, 'short_answer': 180}
        
        base_time = base_times.get(question['type'], 90)
        adjusted_time = int(base_time * time_multipliers.get(difficulty_level, 1.0))
        question['estimated_time'] = f'{adjusted_time} seconds'
        
        # Add difficulty-specific hints for beginners
        if difficulty_level == 'beginner' and question['type'] == 'multiple_choice':
            question['hint_available'] = True
            question['hint'] = f"Think about the key concepts related to {question.get('topic_area', 'this topic')}."
        
        # Add advanced context for complex questions
        if difficulty_level == 'advanced':
            question['requires_synthesis'] = question['type'] == 'short_answer'
            question['complexity_level'] = 'high'
        
        return question

    def _classify_blooms_taxonomy_advanced(self, question_text, allowed_levels):
        """NEW: Advanced Bloom's taxonomy classification with level restrictions"""
        
        question_lower = question_text.lower()
        
        # Comprehensive taxonomy mapping
        taxonomy_indicators = {
            'Remember': ['remember', 'recall', 'list', 'define', 'identify', 'name', 'state'],
            'Understand': ['understand', 'explain', 'describe', 'summarize', 'interpret', 'discuss'],
            'Apply': ['apply', 'use', 'demonstrate', 'solve', 'implement', 'execute'],
            'Analyze': ['analyze', 'compare', 'contrast', 'examine', 'differentiate', 'investigate'],
            'Evaluate': ['evaluate', 'judge', 'critique', 'assess', 'justify', 'defend'],
            'Create': ['create', 'design', 'construct', 'develop', 'formulate', 'synthesize']
        }
        
        # Find the highest level that matches and is allowed
        for level in ['Create', 'Evaluate', 'Analyze', 'Apply', 'Understand', 'Remember']:
            if level in allowed_levels:
                indicators = taxonomy_indicators[level]
                if any(indicator in question_lower for indicator in indicators):
                    return level
        
        # Default to the highest allowed level
        return allowed_levels[0] if allowed_levels else 'Understand'

    # Keep all original methods for backward compatibility
    def _validate_quiz_data_enhanced(self, quiz_data):
        """Enhanced validation and improvement of generated quiz data (original method)"""
        # Ensure required fields exist
        if 'questions' not in quiz_data:
            quiz_data['questions'] = []
        
        if 'quiz_metadata' not in quiz_data:
            quiz_data['quiz_metadata'] = {
                "topic": "Generated Quiz",
                "difficulty": "Intermediate",
                "total_questions": len(quiz_data['questions']),
                "estimated_time": f"{len(quiz_data['questions']) * 2} minutes"
            }
        
        # Validate each question
        valid_questions = []
        for i, question in enumerate(quiz_data['questions']):
            if self._is_valid_question_enhanced(question):
                question['id'] = i + 1
                question = self._enhance_question_quality(question)
                valid_questions.append(question)
        
        quiz_data['questions'] = valid_questions
        quiz_data['quiz_metadata']['total_questions'] = len(valid_questions)
        quiz_data['quiz_metadata']['question_types'] = self._analyze_question_types(valid_questions)
        
        return quiz_data

    def _enhance_question_quality(self, question):
        """Enhance individual question quality (original method)"""
        # Add bloom's taxonomy level
        question['blooms_level'] = self._classify_blooms_taxonomy(question['question'])
        
        # Add estimated time
        if question['type'] == 'multiple_choice':
            question['estimated_time'] = '90 seconds'
        elif question['type'] == 'true_false':
            question['estimated_time'] = '60 seconds'
        else:
            question['estimated_time'] = '3 minutes'
        
        return question

    def _classify_blooms_taxonomy(self, question_text):
        """Classify question according to Bloom's Taxonomy (original method)"""
        question_lower = question_text.lower()
        
        if any(word in question_lower for word in ['remember', 'recall', 'list', 'define', 'identify']):
            return 'Remember'
        elif any(word in question_lower for word in ['understand', 'explain', 'describe', 'summarize']):
            return 'Understand'
        elif any(word in question_lower for word in ['apply', 'use', 'demonstrate', 'solve']):
            return 'Apply'
        elif any(word in question_lower for word in ['analyze', 'compare', 'contrast', 'examine']):
            return 'Analyze'
        elif any(word in question_lower for word in ['evaluate', 'judge', 'critique', 'assess']):
            return 'Evaluate'
        elif any(word in question_lower for word in ['create', 'design', 'construct', 'develop']):
            return 'Create'
        else:
            return 'Understand'  # Default

    def _extract_learning_objectives(self, content):
        """Extract learning objectives from content (original method)"""
        try:
            system_prompt = """Extract 3-5 clear learning objectives from this educational content.

FORMAT: Return as JSON array of objectives
REQUIREMENTS:
- Start each objective with an action verb (understand, analyze, apply, etc.)
- Be specific and measurable
- Focus on what students will be able to do after learning

Example format:
["Understand the basic principles of...", "Apply the concept of... to solve problems", "Analyze the relationship between..."]

Content:"""
            
            content_sample = content[:800]
            response = self._call_groq_api_with_retry(content_sample, system_prompt)
            
            try:
                objectives = json.loads(response)
                return objectives if isinstance(objectives, list) else []
            except:
                return self._extract_objectives_fallback(content)
                
        except:
            return self._extract_objectives_fallback(content)

    def _extract_objectives_fallback(self, content, difficulty_level='intermediate'):
        """Enhanced fallback method for extracting learning objectives"""
        # Look for explicit objectives in text
        objective_patterns = [
            r'(?i)objective[s]?[:\-]\s*(.+?)(?=\n|\.|;)',
            r'(?i)learning outcome[s]?[:\-]\s*(.+?)(?=\n|\.|;)',
            r'(?i)students? will[:\-]\s*(.+?)(?=\n|\.|;)',
            r'(?i)after this lesson[:\-]\s*(.+?)(?=\n|\.|;)'
        ]
        
        objectives = []
        for pattern in objective_patterns:
            matches = re.findall(pattern, content)
            objectives.extend(matches)
        
        if not objectives:
            # Generate basic objectives from content based on difficulty
            sentences = content.split('.')[:5]
            
            if difficulty_level == 'beginner':
                objectives = [f"Understand the basic idea of {sentence.strip()}" for sentence in sentences if len(sentence.strip()) > 20]
            elif difficulty_level == 'advanced':
                objectives = [f"Analyze and evaluate {sentence.strip()}" for sentence in sentences if len(sentence.strip()) > 20]
            else:  # intermediate
                objectives = [f"Understand and explain {sentence.strip()}" for sentence in sentences if len(sentence.strip()) > 20]
        
        return objectives[:5]

    def _identify_prerequisites(self, chapter_name, topics):
        """Identify prerequisites for the chapter (original method)"""
        prerequisites = []
        
        # Basic prerequisite identification based on chapter name and topics
        chapter_lower = chapter_name.lower()
        
        # Mathematics prerequisites
        if any(word in chapter_lower for word in ['advanced', 'calculus', 'differential', 'integral']):
            prerequisites.append('Basic Algebra')
            prerequisites.append('Functions and Graphing')
        
        # Science prerequisites  
        if any(word in chapter_lower for word in ['chemistry', 'physics', 'biology']):
            prerequisites.append('Basic Mathematics')
            prerequisites.append('Scientific Method')
        
        # Advanced topics
        if any(word in chapter_lower for word in ['analysis', 'theory', 'advanced']):
            prerequisites.append('Fundamental Concepts from Previous Chapters')
        
        return prerequisites[:3]  # Limit to 3 prerequisites

    def _call_groq_api_with_retry(self, content, system_prompt, max_retries=3):
        """Enhanced API call with intelligent retry logic and rate limiting"""
        base_delay = 60  # 1 minute base delay
        
        for attempt in range(max_retries):
            try:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.config.GROQ_API_KEY}"
                }
                
                # Debug API key
                if not self.config.GROQ_API_KEY:
                    raise Exception("GROQ_API_KEY is empty or not loaded")
                
                if not self.config.GROQ_API_KEY.startswith('gsk_'):
                    raise Exception(f"Invalid API key format. Got: {self.config.GROQ_API_KEY[:10]}...")
                
                payload = {
                    "model": self.config.GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content}
                    ],
                    "max_tokens": 800,  # Reduced to save tokens
                    "temperature": 0.3
                }
                
                response = requests.post(
                    f"{self.config.GROQ_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json()['choices'][0]['message']['content']
                elif response.status_code == 429:  # Rate limit exceeded
                    retry_after = response.headers.get('retry-after', base_delay)
                    wait_time = int(retry_after) + random.randint(5, 15)  # Add jitter
                    
                    st.warning(f"⏳ Rate limit reached. Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    
                    if attempt < max_retries - 1:
                        continue
                    else:
                        raise Exception("Rate limit exceeded after multiple retries")
                elif response.status_code == 401:
                    raise Exception(f"API Authentication failed. Check your Groq API key.")
                else:
                    raise Exception(f"API call failed with status {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (attempt + 1))
                    continue
                raise Exception(f"Network error after {max_retries} attempts: {str(e)}")
        
        raise Exception(f"API call failed after {max_retries} attempts")

    # Enhanced content analysis methods (keep all original methods)
    def _extract_chapter_concepts_enhanced(self, chapter_content):
        """Enhanced concept extraction with better NLP (original method)"""
        concepts = set()
        
        if self.nlp:
            # Process in chunks to handle long content
            content_chunks = [chapter_content[i:i+2000] for i in range(0, len(chapter_content), 2000)]
            
            for chunk in content_chunks:
                doc = self.nlp(chunk)
                
                # Extract named entities
                for ent in doc.ents:
                    if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT', 'PRODUCT', 'LAW', 'LANGUAGE']:
                        if len(ent.text) > 2:
                            concepts.add(ent.text)
                
                # Extract noun phrases with filtering
                for chunk in doc.noun_chunks:
                    chunk_text = chunk.text.strip()
                    if (2 <= len(chunk_text.split()) <= 4 and 
                        len(chunk_text) > 5 and
                        not chunk_text.lower().startswith(('the ', 'a ', 'an '))):
                        concepts.add(chunk_text)
        
        # Enhanced pattern-based extraction
        concept_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns
            r'\b\w+(?:tion|sion|ment|ness|ity|ism|ogy|graphy)\b',  # Academic suffixes
            r'\b(?:principle|theory|law|rule|concept|method|process|system|model|framework)\s+of\s+\w+',
        ]
        
        for pattern in concept_patterns:
            matches = re.findall(pattern, chapter_content)
            concepts.update(match for match in matches if len(match) > 3)
        
        # Filter and rank concepts
        filtered_concepts = []
        for concept in concepts:
            if (len(concept.split()) <= 4 and 
                len(concept) > 3 and
                not re.match(r'^\d+', concept) and
                concept.lower() not in ['chapter', 'section', 'page', 'figure', 'table']):
                filtered_concepts.append(concept)
        
        return filtered_concepts[:20]  # Return top 20 concepts

    def _find_real_world_examples_enhanced(self, chapter_content):
        """Enhanced real-world example extraction (original method)"""
        example_patterns = [
            r'(?i)for example[,:]?\s*([^.!?]*[.!?])',
            r'(?i)example\s*\d*[:\-]?\s*([^.!?]*[.!?])',
            r'(?i)consider\s+([^.!?]*[.!?])',
            r'(?i)such as\s+([^.!?]*[.!?])',
            r'(?i)instance[,:]?\s*([^.!?]*[.!?])',
            r'(?i)in practice[,:]?\s*([^.!?]*[.!?])',
            r'(?i)real[\-\s]world\s+([^.!?]*[.!?])',
            r'(?i)case study[,:]?\s*([^.!?]*[.!?])'
        ]
        
        examples = []
        for pattern in example_patterns:
            matches = re.findall(pattern, chapter_content)
            for match in matches:
                cleaned_match = match.strip()
                if (len(cleaned_match) > 30 and  # Longer examples are more valuable
                    len(cleaned_match) < 300 and  # But not too long
                    not cleaned_match.lower().startswith(('the following', 'as follows'))):
                    examples.append(cleaned_match)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_examples = []
        for example in examples:
            if example.lower() not in seen:
                seen.add(example.lower())
                unique_examples.append(example)
        
        return unique_examples[:10]  # Return top 10 examples

    def _assess_content_difficulty_enhanced(self, content):
        """Enhanced difficulty assessment using multiple metrics (original method)"""
        if not content or len(content.strip()) == 0:
            return "Intermediate"
        
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        words = content.split()
        
        if not sentences or not words:
            return "Intermediate"
        
        # Calculate various metrics
        avg_sentence_length = np.mean([len(s.split()) for s in sentences])
        
        # Syllable approximation (more accurate than just length)
        complex_words = []
        for word in words:
            word_lower = word.lower().strip('.,!?";:')
            if len(word_lower) > 6:  # Likely multi-syllabic
                complex_words.append(word_lower)
        
        complexity_ratio = len(complex_words) / len(words) if words else 0
        
        # Academic vocabulary detection
        academic_words = 0
        academic_indicators = [
            'analyze', 'synthesize', 'evaluate', 'methodology', 'hypothesis',
            'correlation', 'significant', 'furthermore', 'consequently', 'theoretical',
            'empirical', 'quantitative', 'qualitative', 'paradigm', 'framework'
        ]
        
        content_lower = content.lower()
        for indicator in academic_indicators:
            if indicator in content_lower:
                academic_words += content_lower.count(indicator)
        
        academic_ratio = academic_words / len(words) if words else 0
        
        # Calculate composite difficulty score
        sentence_score = min(avg_sentence_length / 25, 1.0)  # Normalize to 0-1
        complexity_score = min(complexity_ratio / 0.3, 1.0)  # Normalize to 0-1
        academic_score = min(academic_ratio / 0.05, 1.0)  # Normalize to 0-1
        
        # Weighted composite score
        composite_score = (sentence_score * 0.4 + complexity_score * 0.4 + academic_score * 0.2)
        
        # Classify difficulty
        if composite_score < 0.3:
            return "Beginner"
        elif composite_score < 0.7:
            return "Intermediate"
        else:
            return "Advanced"

    def _extract_key_points_ai_enhanced(self, content):
        """Enhanced key points extraction with fallback strategies (original method)"""
        try:
            system_prompt = """Extract 5-7 key learning points from this educational content. 

FORMAT: Return as a JSON array of strings
REQUIREMENTS:
- Each point should be clear and educational
- Focus on main concepts and important information  
- Keep each point concise (1-2 sentences)
- Ensure points are directly from the content
- Prioritize actionable learning outcomes

Example: ["Students will understand the concept of...", "Key principle: ...", "Important application: ..."]

Content:"""
            
            content_sample = self._optimize_content_for_api(content, 800)
            response = self._call_groq_api_with_retry(content_sample, system_prompt)
            
            try:
                points = json.loads(response)
                if isinstance(points, list) and len(points) > 0:
                    return points[:7]
            except:
                pass
            
            # Fallback to regex parsing
            points = []
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith(('-', '•', '*')) or re.match(r'^\d+\.', line):
                    point = re.sub(r'^[-•*\d\.\s]+', '', line).strip()
                    if len(point) > 15:  # Filter out very short points
                        points.append(point)
            
            return points[:7] if points else self._extract_key_points_fallback(content)
            
        except Exception as e:
            return self._extract_key_points_fallback(content)

    def _extract_key_points_fallback(self, content, difficulty_level='intermediate'):
        """Enhanced fallback method for key points extraction"""
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        
        # Score sentences based on educational value
        scored_sentences = []
        for sentence in sentences:
            if len(sentence) < 20:
                continue
                
            score = 0
            sentence_lower = sentence.lower()
            
            # Higher score for sentences with key educational terms
            key_terms = ['important', 'key', 'main', 'primary', 'essential', 'fundamental', 'crucial']
            for term in key_terms:
                if term in sentence_lower:
                    score += 2
            
            # Higher score for sentences with numbers (often key facts)
            if re.search(r'\d+', sentence):
                score += 1
            
            # Higher score for longer, informative sentences
            score += len(sentence.split()) / 30
            
            # Difficulty-specific scoring adjustments
            if difficulty_level == 'beginner':
                # Prefer simpler sentences for beginners
                if len(sentence.split()) > 20:
                    score -= 1
            elif difficulty_level == 'advanced':
                # Prefer complex sentences for advanced learners
                if len(sentence.split()) > 15:
                    score += 1
            
            scored_sentences.append((sentence, score))
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        max_points = {'beginner': 5, 'intermediate': 7, 'advanced': 8}
        return [sentence for sentence, _ in scored_sentences[:max_points.get(difficulty_level, 7)]]

    # Keep all other original methods for compatibility
    def _basic_topic_extraction_enhanced(self, chapter_content, difficulty_level='intermediate'):
        """Enhanced fallback topic extraction with difficulty adaptation"""
        sentences = [s.strip() for s in chapter_content.split('.') if s.strip()][:25]
        
        if len(sentences) < 3:
            return {
                "Main Topic": {
                    "key_points": sentences[:3] if sentences else ["Content analysis in progress"],
                    "difficulty": difficulty_level.title(),
                    "summary": chapter_content[:200] + "..." if len(chapter_content) > 200 else chapter_content,
                    "content": chapter_content[:500],
                    "learning_objectives": [f"Understand the main concepts for {difficulty_level} level"],
                    "estimated_time": self._get_estimated_time_by_difficulty(difficulty_level, len(sentences))
                }
            }
        
        # Use simple clustering to group related sentences
        topics = {}
        sentences_per_topic = max(3, len(sentences) // 4)
        
        for i in range(0, len(sentences), sentences_per_topic):
            topic_sentences = sentences[i:i+sentences_per_topic]
            topic_name = f"Key Concept {len(topics) + 1}"
            
            # Try to extract a meaningful topic name from first sentence
            first_sentence = topic_sentences[0] if topic_sentences else ""
            potential_name = self._extract_topic_name_from_sentence(first_sentence)
            if potential_name:
                topic_name = potential_name
            
            topics[topic_name] = {
                "key_points": topic_sentences[:4],
                "difficulty": difficulty_level.title(),
                "summary": '. '.join(topic_sentences[:2]) + '.',
                "content": '. '.join(topic_sentences) + '.',
                "learning_objectives": [f"Understand {topic_name.lower()} at {difficulty_level} level"],
                "estimated_time": self._get_estimated_time_by_difficulty(difficulty_level, len(topic_sentences))
            }
            
            if len(topics) >= 5:  # Limit to 5 topics
                break
        
        return topics

    def _get_estimated_time_by_difficulty(self, difficulty_level, content_units):
        """Get estimated time based on difficulty level and content amount"""
        time_multipliers = {'beginner': 3, 'intermediate': 2, 'advanced': 1.5}
        base_time = content_units * time_multipliers.get(difficulty_level, 2)
        return f"{int(base_time)} minutes"

    # Keep all remaining original methods for backward compatibility...
    def _extract_topic_name_from_sentence(self, sentence):
        """Extract a meaningful topic name from a sentence"""
        if not sentence or len(sentence) < 10:
            return None
        
        # Look for patterns that might indicate topic names
        patterns = [
            r'(?:the concept of|the principle of|the theory of)\s+([^,\.]+)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is|are|refers?)',
            r'^([A-Z][a-z]+(?:\s+[a-z]+){0,2})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, sentence)
            if match:
                topic_name = match.group(1).strip()
                if len(topic_name) > 3 and len(topic_name.split()) <= 4:
                    return topic_name
        
        return None

    def _extract_topic_content_enhanced(self, chapter_content, topic_name):
        """Enhanced topic content extraction using semantic similarity"""
        sentences = [s.strip() for s in chapter_content.split('.') if s.strip()]
        topic_keywords = topic_name.lower().split()
        
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Direct keyword matching
            keyword_matches = sum(1 for keyword in topic_keywords if keyword in sentence_lower)
            
            # Bonus for semantic relevance (simple approach)
            relevance_score = keyword_matches
            
            # Look for related terms
            if any(word in sentence_lower for word in ['principle', 'concept', 'theory', 'method', 'process']):
                relevance_score += 0.5
            
            if relevance_score > 0:
                relevant_sentences.append((sentence, relevance_score))
        
        # Sort by relevance and take top sentences
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        
        if relevant_sentences:
            selected_sentences = [s for s, _ in relevant_sentences[:7]]
            return '. '.join(selected_sentences) + '.'
        else:
            return chapter_content[:400] + "..." if len(chapter_content) > 400 else chapter_content

    def _extract_topic_keywords(self, topic_name, topic_data):
        """Extract keywords for the topic"""
        keywords = set()
        
        # Add topic name words
        keywords.update(word.lower() for word in topic_name.split() if len(word) > 3)
        
        # Extract from key points
        key_points = topic_data.get('key_points', [])
        for point in key_points:
            words = re.findall(r'\b[a-zA-Z]{4,}\b', point.lower())
            keywords.update(words)
        
        # Remove common stop words
        stop_words = {'this', 'that', 'with', 'from', 'they', 'have', 'were', 'been', 'their', 'said', 'each', 'which', 'them', 'will'}
        keywords = keywords - stop_words
        
        return list(keywords)[:10]

    def _generate_fallback_quiz_enhanced(self, topic_data, difficulty_level, num_questions):
        """Enhanced fallback quiz generation with difficulty adaptation"""
        key_points = topic_data.get('key_points', [])
        
        questions = []
        
        # Difficulty-adapted question types
        if difficulty_level == 'beginner':
            question_types = ['multiple_choice', 'true_false', 'multiple_choice']  # More MC for beginners
        elif difficulty_level == 'advanced':
            question_types = ['short_answer', 'multiple_choice', 'short_answer']  # More SA for advanced
        else:
            question_types = ['multiple_choice', 'true_false', 'short_answer']
        
        for i in range(min(num_questions, len(key_points))):
            point = key_points[i]
            question_type = question_types[i % len(question_types)]
            
            if question_type == 'multiple_choice':
                # Difficulty-adapted options
                if difficulty_level == 'beginner':
                    options = [
                        point[:50] + "...",
                        "Incorrect option A",
                        "Incorrect option B"
                    ]
                else:
                    options = [
                        point[:50] + "...",
                        "Incorrect option A",
                        "Incorrect option B", 
                        "Incorrect option C"
                    ]
                    
                    if difficulty_level == 'advanced':
                        options.append("Incorrect option D")
                
                questions.append({
                    "id": i + 1,
                    "type": "multiple_choice",
                    "question": f"Which of the following best describes: {point[:80]}?",
                    "options": options,
                    "correct_answer": 0,
                    "explanation": f"This relates to {point}",
                    "difficulty": difficulty_level,
                    "topic_area": topic_data.get('summary', 'General')
                })
                
            elif question_type == 'true_false':
                questions.append({
                    "id": i + 1,
                    "type": "true_false",
                    "question": f"True or False: {point}",
                    "correct_answer": True,
                    "explanation": f"This statement is true based on the content: {point}",
                    "difficulty": difficulty_level
                })
                
            else:  # short_answer
                if difficulty_level == 'beginner':
                    question_text = f"In simple terms, explain: {point[:40]}"
                elif difficulty_level == 'advanced':
                    question_text = f"Analyze and evaluate the significance of: {point[:60]}"
                else:
                    question_text = f"Explain the concept: {point[:60]}"
                
                questions.append({
                    "id": i + 1,
                    "type": "short_answer",
                    "question": question_text,
                    "sample_answer": f"This relates to {point} as discussed in the chapter.",
                    "key_points": [point],
                    "difficulty": difficulty_level,
                    "topic_area": topic_data.get('summary', 'General')
                })
        
        return {
            "quiz_metadata": {
                "topic": topic_data.get('summary', 'Educational Topic'),
                "difficulty": difficulty_level,
                "total_questions": len(questions),
                "estimated_time": self._get_quiz_time(difficulty_level, len(questions)),
                "question_types": self._analyze_question_types(questions),
                "difficulty_adapted": True
            },
            "questions": questions
        }

    def _is_valid_question_enhanced(self, question):
        """Enhanced question validation (original method)"""
        required_fields = ['type', 'question']
        
        # Check required fields
        for field in required_fields:
            if field not in question or not question[field]:
                return False
        
        # Check question is not too short
        if len(question['question']) < 10:
            return False
        
        # Type-specific validation
        if question['type'] == 'multiple_choice':
            options = question.get('options', [])
            correct_answer = question.get('correct_answer')
            
            return (len(options) >= 2 and 
                    isinstance(correct_answer, int) and 
                    0 <= correct_answer < len(options))
        
        elif question['type'] == 'true_false':
            correct_answer = question.get('correct_answer')
            return isinstance(correct_answer, bool)
        
        elif question['type'] == 'short_answer':
            return True  # Short answer questions just need question text
        
        return False

    def _analyze_question_types(self, questions):
        """Analyze distribution of question types"""
        type_counts = {}
        for question in questions:
            q_type = question.get('type', 'unknown')
            type_counts[q_type] = type_counts.get(q_type, 0) + 1
        
        return type_counts

    # Keep your original _call_groq_api method as fallback
    def _call_groq_api(self, content, system_prompt):
        """Original API call method (kept for backward compatibility)"""
        return self._call_groq_api_with_retry(content, system_prompt, max_retries=1)
