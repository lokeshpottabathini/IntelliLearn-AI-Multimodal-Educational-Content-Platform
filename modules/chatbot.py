import requests
import json
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
from config import Config
import time
import wikipedia
from duckduckgo_search import DDGS
import speech_recognition as sr
from gtts import gTTS
import tempfile
import os
from datetime import datetime
import torch 

# Enhanced imports for difficulty-aware chatbot
try:
    from .voice_chat_enhanced import VoiceChatProcessor
    from .difficulty_adapter import DifficultyAdaptiveResponder
    from .enhanced_nlp_processor import EnhancedNLPProcessor
    ENHANCED_MODULES_AVAILABLE = True
except ImportError:
    ENHANCED_MODULES_AVAILABLE = False

class EnhancedEducationalChatbot:
    def __init__(self, knowledge_base, subject="General", difficulty_level='intermediate'):
        """Initialize enhanced chatbot with difficulty adaptation and voice capabilities"""
        self.knowledge_base = knowledge_base
        self.subject = subject
        self.difficulty_level = difficulty_level
        self.config = Config()
        self.conversation_history = []
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            st.success("✅ Embedding model loaded successfully")
        except Exception as e:
            st.warning(f"⚠️ Embedding model not available: {e}")
            self.embedding_model = None
        
        # Create embeddings for textbook content
        self._create_textbook_embeddings()
        
        # Initialize enhanced modules
        self.voice_processor = None
        self.difficulty_adapter = None
        self.enhanced_nlp = None
        
        if ENHANCED_MODULES_AVAILABLE:
            try:
                self.voice_processor = VoiceChatProcessor()
                self.difficulty_adapter = DifficultyAdaptiveResponder()
                self.enhanced_nlp = EnhancedNLPProcessor()
                st.success("✅ Enhanced chatbot features loaded")
            except Exception as e:
                st.warning(f"Enhanced features not available: {e}")
        
        # Initialize external knowledge sources with enhanced capabilities
        self.external_sources = {
            'wikipedia': True,
            'web_search': True,
            'subject_specific': True,
            'voice_enabled': bool(self.voice_processor),
            'difficulty_adaptive': bool(self.difficulty_adapter)
        }
        
        # Enhanced conversation tracking
        self.session_stats = {
            'total_questions': 0,
            'voice_interactions': 0,
            'difficulty_adaptations': 0,
            'sources_accessed': set(),
            'session_start': datetime.now(),
            'topics_discussed': set()
        }

    def get_comprehensive_response_with_difficulty(self, user_query, use_voice=False, adapt_difficulty=True):
        """NEW: Enhanced response generation with difficulty adaptation and voice support"""
        try:
            st.write(f"🔍 Analyzing your {self.difficulty_level} level question...")
            
            # Voice input processing if enabled
            if use_voice and self.voice_processor:
                processed_query = self.voice_processor.process_voice_input(user_query)
                if processed_query:
                    user_query = processed_query
                    self.session_stats['voice_interactions'] += 1
            
            # Get textbook context (primary source)
            textbook_context = self._get_textbook_context_enhanced(user_query)
            
            # Get difficulty-adaptive external context
            external_context = self._get_external_context_enhanced(user_query)
            
            # Generate difficulty-adapted response
            if self.difficulty_adapter and adapt_difficulty:
                response = self.difficulty_adapter.generate_adaptive_response(
                    user_query, textbook_context, external_context, self.difficulty_level
                )
                self.session_stats['difficulty_adaptations'] += 1
            else:
                response = self._generate_multi_source_response_enhanced(
                    user_query, textbook_context, external_context
                )
            
            # Ensure response matches difficulty level
            final_response = self._adapt_response_to_difficulty(response, user_query)
            
            # Voice output if requested
            voice_output = None
            if use_voice and self.voice_processor:
                voice_output = self.voice_processor.generate_voice_response(final_response)
            
            # Enhanced conversation tracking
            self._track_conversation_enhanced(user_query, final_response, textbook_context, external_context)
            
            return {
                'text_response': final_response,
                'voice_output': voice_output,
                'sources_used': self._count_sources_used(textbook_context, external_context),
                'difficulty_level': self.difficulty_level,
                'processing_enhanced': True
            }
            
        except Exception as e:
            error_msg = f"I apologize for the error: {str(e)}. Let me provide a response from the textbook."
            return {
                'text_response': self._get_fallback_response_enhanced(user_query),
                'voice_output': None,
                'sources_used': 1,
                'difficulty_level': self.difficulty_level,
                'processing_enhanced': False
            }

    def get_comprehensive_response(self, user_query):
        """Enhanced comprehensive response using multiple sources (original method with enhancements)"""
        try:
            st.write("🔍 Analyzing your question...")
            
            # Get textbook context (primary source)
            textbook_context = self._get_textbook_context(user_query)
            
            # Get external context for enhancement
            external_context = self._get_external_context(user_query)
            
            # Generate comprehensive response
            response = self._generate_multi_source_response(
                user_query, textbook_context, external_context
            )
            
            # Ensure response is understandable
            final_response = self._simplify_for_understanding(response, user_query)
            
            # Add to conversation history
            self.conversation_history.append({
                'user': user_query,
                'assistant': final_response,
                'timestamp': time.strftime("%H:%M:%S"),
                'sources_used': self._count_sources_used(textbook_context, external_context)
            })
            
            # Update session stats
            self.session_stats['total_questions'] += 1
            
            return final_response
            
        except Exception as e:
            error_msg = f"I apologize for the error: {str(e)}. Let me provide a basic response from the textbook."
            return self._get_fallback_response(user_query)

    def _create_textbook_embeddings(self):
        """Enhanced textbook embedding creation with difficulty awareness"""
        try:
            self.content_chunks = []
            
            # Process chapters and topics with enhanced metadata
            for chapter_name, chapter_data in self.knowledge_base.get('chapters', {}).items():
                # Add chapter overview with difficulty context
                chapter_difficulty = chapter_data.get('difficulty_assessment', {}).get('detected_level', 'intermediate')
                
                self.content_chunks.append({
                    'content': f"Chapter: {chapter_name}\n{chapter_data.get('content', '')[:500]}",
                    'type': 'chapter',
                    'source': chapter_name,
                    'difficulty': chapter_difficulty,
                    'word_count': len(chapter_data.get('content', '').split()),
                    'concepts': chapter_data.get('concepts', [])[:5]
                })
                
                # Add topics with enhanced metadata
                for topic_name, topic_data in chapter_data.get('topics', {}).items():
                    topic_difficulty = topic_data.get('difficulty', 'intermediate').lower()
                    
                    self.content_chunks.append({
                        'content': f"Topic: {topic_name}\n{topic_data.get('content', '')}",
                        'type': 'topic',
                        'source': f"{chapter_name} - {topic_name}",
                        'difficulty': topic_difficulty,
                        'estimated_time': topic_data.get('estimated_time', 'N/A'),
                        'key_points': topic_data.get('key_points', [])
                    })
                    
                    # Add key points with context
                    for i, point in enumerate(topic_data.get('key_points', [])):
                        self.content_chunks.append({
                            'content': f"Key Point: {point}",
                            'type': 'key_point',
                            'source': f"{chapter_name} - {topic_name}",
                            'difficulty': topic_difficulty,
                            'point_index': i
                        })
            
            # Generate embeddings if model is available
            if self.embedding_model and self.content_chunks:
                texts = [chunk['content'] for chunk in self.content_chunks]
                self.embeddings = self.embedding_model.encode(texts)
                st.success(f"✅ Created embeddings for {len(self.content_chunks)} content chunks")
            else:
                self.embeddings = np.array([])
                st.warning("⚠️ No embedding model available or no content to process")
                
        except Exception as e:
            st.warning(f"Error creating embeddings: {e}")
            self.content_chunks = []
            self.embeddings = np.array([])

    def _get_textbook_context_enhanced(self, query, top_k=5):
        """NEW: Enhanced textbook context retrieval with difficulty filtering"""
        if len(self.embeddings) == 0:
            return [{'content': 'No textbook content available.', 'source': 'textbook', 'type': 'general'}]
        
        try:
            # Encode query
            query_embedding = self.embedding_model.encode([query])
            
            # Calculate similarities
            similarities = np.dot(query_embedding, self.embeddings.T).flatten()
            
            # Get top results with difficulty filtering
            top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get more to filter by difficulty
            
            relevant_chunks = []
            difficulty_priorities = {
                'beginner': ['beginner', 'intermediate', 'advanced'],
                'intermediate': ['intermediate', 'beginner', 'advanced'],
                'advanced': ['advanced', 'intermediate', 'beginner']
            }
            
            priority_order = difficulty_priorities.get(self.difficulty_level, ['intermediate', 'beginner', 'advanced'])
            
            # First, get chunks matching current difficulty level
            for priority_difficulty in priority_order:
                for idx in top_indices:
                    if len(relevant_chunks) >= top_k:
                        break
                    
                    if similarities[idx] > 0.25:  # Lower threshold for more results
                        chunk = self.content_chunks[idx]
                        chunk_difficulty = chunk.get('difficulty', 'intermediate').lower()
                        
                        if chunk_difficulty == priority_difficulty:
                            relevant_chunks.append({
                                'content': chunk['content'],
                                'source': chunk['source'],
                                'type': chunk['type'],
                                'difficulty': chunk_difficulty,
                                'similarity': float(similarities[idx]),
                                'metadata': {
                                    'word_count': chunk.get('word_count', 0),
                                    'concepts': chunk.get('concepts', []),
                                    'estimated_time': chunk.get('estimated_time', 'N/A')
                                }
                            })
                
                if len(relevant_chunks) >= top_k:
                    break
            
            return relevant_chunks if relevant_chunks else [
                {'content': 'No highly relevant textbook content found for your difficulty level.', 
                 'source': 'textbook', 'type': 'general', 'difficulty': self.difficulty_level}
            ]
            
        except Exception as e:
            return [{'content': f'Error retrieving textbook context: {e}', 'source': 'error', 'type': 'error'}]

    def _get_textbook_context(self, query, top_k=4):
        """Get relevant context from textbook (original method)"""
        if len(self.embeddings) == 0:
            return "No textbook content available."
        
        try:
            # Encode query
            query_embedding = self.embedding_model.encode([query])
            
            # Calculate similarities
            similarities = np.dot(query_embedding, self.embeddings.T).flatten()
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            relevant_chunks = []
            for idx in top_indices:
                if similarities[idx] > 0.3:  # Similarity threshold
                    chunk = self.content_chunks[idx]
                    relevant_chunks.append({
                        'content': chunk['content'],
                        'source': chunk['source'],
                        'type': chunk['type'],
                        'similarity': float(similarities[idx])
                    })
            
            return relevant_chunks if relevant_chunks else [{'content': 'No highly relevant textbook content found.', 'source': 'textbook', 'type': 'general'}]
            
        except Exception as e:
            return [{'content': f'Error retrieving textbook context: {e}', 'source': 'error', 'type': 'error'}]

    def _get_external_context_enhanced(self, query):
        """NEW: Enhanced external context with difficulty-appropriate sources"""
        external_info = {}
        
        # Wikipedia search with difficulty-appropriate content length
        if self.external_sources['wikipedia']:
            try:
                wiki_info = self._search_wikipedia_enhanced(query)
                external_info['wikipedia'] = wiki_info
                if wiki_info.get('available'):
                    self.session_stats['sources_accessed'].add('wikipedia')
            except Exception as e:
                external_info['wikipedia'] = {'error': str(e)}
        
        # Enhanced subject-specific information
        if self.external_sources['subject_specific']:
            try:
                subject_info = self._get_subject_specific_info_enhanced(query)
                external_info['subject_specific'] = subject_info
                if subject_info.get('content'):
                    self.session_stats['sources_accessed'].add('subject_database')
            except Exception as e:
                external_info['subject_specific'] = {'error': str(e)}
        
        # Web search for advanced users
        if self.difficulty_level == 'advanced' and self.external_sources['web_search']:
            try:
                web_info = self._search_web_enhanced(query)
                external_info['web_search'] = web_info
                if web_info.get('results'):
                    self.session_stats['sources_accessed'].add('web_search')
            except Exception as e:
                external_info['web_search'] = {'error': str(e)}
        
        return external_info

    def _get_external_context(self, query):
        """Get external context from multiple sources (original method)"""
        external_info = {}
        
        # Wikipedia search
        if self.external_sources['wikipedia']:
            try:
                wiki_info = self._search_wikipedia(query)
                external_info['wikipedia'] = wiki_info
            except Exception as e:
                external_info['wikipedia'] = {'error': str(e)}
        
        # Subject-specific information
        if self.external_sources['subject_specific']:
            try:
                subject_info = self._get_subject_specific_info(query)
                external_info['subject_specific'] = subject_info
            except Exception as e:
                external_info['subject_specific'] = {'error': str(e)}
        
        return external_info

    def _search_wikipedia_enhanced(self, query):
        """NEW: Enhanced Wikipedia search with difficulty-appropriate content"""
        try:
            # Search for relevant articles
            search_results = wikipedia.search(query, results=3)
            
            if search_results:
                # Get summary of most relevant article
                page_title = search_results[0]
                
                # Adjust summary length based on difficulty level
                sentence_counts = {'beginner': 2, 'intermediate': 3, 'advanced': 4}
                sentences = sentence_counts.get(self.difficulty_level, 3)
                
                summary = wikipedia.summary(page_title, sentences=sentences)
                
                # Get page for additional info
                try:
                    page = wikipedia.page(page_title)
                    url = page.url
                except:
                    url = None
                
                return {
                    'title': page_title,
                    'summary': summary,
                    'url': url,
                    'available': True,
                    'difficulty_adapted': True,
                    'sentences_count': sentences
                }
            else:
                return {'available': False, 'message': 'No relevant Wikipedia articles found'}
                
        except wikipedia.exceptions.DisambiguationError as e:
            # Handle disambiguation - choose most relevant option
            if e.options:
                try:
                    sentence_counts = {'beginner': 2, 'intermediate': 3, 'advanced': 4}
                    sentences = sentence_counts.get(self.difficulty_level, 3)
                    
                    summary = wikipedia.summary(e.options[0], sentences=sentences)
                    return {
                        'title': e.options[0],
                        'summary': summary,
                        'available': True,
                        'disambiguation_resolved': True
                    }
                except:
                    return {'available': False, 'error': 'Disambiguation resolution failed'}
            return {'available': False, 'error': 'Disambiguation options unavailable'}
        except Exception as e:
            return {'available': False, 'error': str(e)}

    def _search_wikipedia(self, query):
        """Search Wikipedia for additional context (original method)"""
        try:
            # Search for relevant articles
            search_results = wikipedia.search(query, results=2)
            
            if search_results:
                # Get summary of most relevant article
                page_title = search_results[0]
                summary = wikipedia.summary(page_title, sentences=3)
                
                return {
                    'title': page_title,
                    'summary': summary,
                    'available': True
                }
            else:
                return {'available': False, 'message': 'No relevant Wikipedia articles found'}
                
        except wikipedia.exceptions.DisambiguationError as e:
            # Handle disambiguation
            return {
                'title': e.options[0] if e.options else 'Unknown',
                'summary': wikipedia.summary(e.options[0], sentences=2) if e.options else 'No summary available',
                'available': True
            }
        except Exception as e:
            return {'available': False, 'error': str(e)}

    def _search_web_enhanced(self, query):
        """NEW: Enhanced web search for advanced learners"""
        try:
            # Use DuckDuckGo for web search
            with DDGS() as ddgs:
                # Limit results based on difficulty
                max_results = {'beginner': 2, 'intermediate': 3, 'advanced': 5}
                results = list(ddgs.text(f"{query} {self.subject}", max_results=max_results.get(self.difficulty_level, 3)))
                
                if results:
                    formatted_results = []
                    for result in results[:3]:  # Limit to top 3
                        formatted_results.append({
                            'title': result.get('title', 'No title'),
                            'snippet': result.get('body', 'No content')[:200] + "...",
                            'url': result.get('href', 'No URL')
                        })
                    
                    return {
                        'results': formatted_results,
                        'count': len(formatted_results),
                        'available': True
                    }
                else:
                    return {'available': False, 'message': 'No web results found'}
        except Exception as e:
            return {'available': False, 'error': str(e)}

    def _get_subject_specific_info_enhanced(self, query):
        """NEW: Enhanced subject-specific information with difficulty adaptation"""
        
        # Difficulty-adapted subject prompts
        difficulty_prompts = {
            'beginner': {
                'Biology': f"Provide simple, beginner-friendly biological information about: {query}. Use basic vocabulary and clear explanations.",
                'Chemistry': f"Explain basic chemistry concepts related to: {query}. Keep it simple and easy to understand.",
                'Physics': f"Provide elementary physics information about: {query}. Use everyday examples and simple terms.",
                'Mathematics': f"Explain basic mathematical concepts for: {query}. Use simple examples and clear steps.",
                'History': f"Provide basic historical context for: {query}. Use simple language and focus on main events.",
                'Literature': f"Explain basic literary concepts for: {query}. Use clear, simple explanations."
            },
            'intermediate': {
                'Biology': f"Provide detailed biological context for: {query}. Include mechanisms and processes.",
                'Chemistry': f"Provide chemical context with reactions and principles for: {query}.",
                'Physics': f"Provide physics context with formulas and applications for: {query}.",
                'Mathematics': f"Provide mathematical context with detailed examples for: {query}.",
                'History': f"Provide historical context with causes and effects for: {query}.",
                'Literature': f"Provide literary analysis and context for: {query}."
            },
            'advanced': {
                'Biology': f"Provide advanced biological analysis including molecular mechanisms and research findings for: {query}.",
                'Chemistry': f"Provide advanced chemical analysis with complex reactions and theoretical frameworks for: {query}.",
                'Physics': f"Provide advanced physics analysis with complex mathematics and theoretical implications for: {query}.",
                'Mathematics': f"Provide advanced mathematical analysis with proofs and complex applications for: {query}.",
                'History': f"Provide advanced historical analysis with multiple perspectives and scholarly interpretations for: {query}.",
                'Literature': f"Provide advanced literary criticism and theoretical analysis for: {query}."
            }
        }
        
        level_prompts = difficulty_prompts.get(self.difficulty_level, difficulty_prompts['intermediate'])
        
        if self.subject in level_prompts:
            try:
                # Use Groq to generate difficulty-adapted subject-specific context
                context = self._call_groq_api_enhanced(
                    level_prompts[self.subject],
                    f"You are a {self.subject} expert specializing in {self.difficulty_level} level education. Provide accurate, {self.difficulty_level}-appropriate educational content."
                )
                return {
                    'content': context, 
                    'subject': self.subject,
                    'difficulty_level': self.difficulty_level,
                    'enhanced': True
                }
            except Exception as e:
                return {'error': str(e)}
        
        return {'message': f'No specific database available for {self.subject} at {self.difficulty_level} level'}

    def _get_subject_specific_info(self, query):
        """Get subject-specific information (original method)"""
        subject_prompts = {
            'Biology': f"Provide additional biological context for: {query}",
            'Chemistry': f"Provide additional chemical context for: {query}",
            'Physics': f"Provide additional physics context for: {query}",
            'Mathematics': f"Provide additional mathematical context for: {query}",
            'History': f"Provide additional historical context for: {query}",
            'Literature': f"Provide additional literary context for: {query}"
        }
        
        if self.subject in subject_prompts:
            try:
                # Use Groq to generate subject-specific context
                context = self._call_groq_api(
                    subject_prompts[self.subject],
                    f"You are a {self.subject} expert. Provide accurate, educational context."
                )
                return {'content': context, 'subject': self.subject}
            except Exception as e:
                return {'error': str(e)}
        
        return {'message': f'No specific database available for {self.subject}'}

    def _generate_multi_source_response_enhanced(self, query, textbook_context, external_context):
        """NEW: Enhanced multi-source response generation with difficulty adaptation"""
        
        # Prepare textbook content with difficulty awareness
        textbook_content = ""
        for chunk in textbook_context:
            difficulty_match = chunk.get('difficulty', 'intermediate') == self.difficulty_level
            relevance_indicator = "🎯 HIGHLY RELEVANT" if difficulty_match else "📚 ADDITIONAL"
            textbook_content += f"{relevance_indicator} - From {chunk.get('source', 'textbook')}: {chunk.get('content', '')}\n\n"
        
        # Prepare external content with source attribution
        external_content = ""
        
        # Add Wikipedia content
        wiki_info = external_context.get('wikipedia', {})
        if wiki_info.get('available'):
            external_content += f"📖 Wikipedia Context ({wiki_info.get('title', 'Article')}): {wiki_info.get('summary', '')}\n\n"
        
        # Add subject-specific content
        subject_info = external_context.get('subject_specific', {})
        if 'content' in subject_info:
            external_content += f"🔬 {self.subject} Expert Knowledge: {subject_info['content']}\n\n"
        
        # Add web search results for advanced users
        web_info = external_context.get('web_search', {})
        if web_info.get('available') and self.difficulty_level == 'advanced':
            external_content += "🌐 Additional Web Resources:\n"
            for result in web_info.get('results', [])[:2]:
                external_content += f"• {result.get('title', 'Resource')}: {result.get('snippet', 'No description')}\n"
            external_content += "\n"

        # Difficulty-specific system prompts
        difficulty_system_prompts = {
            'beginner': f"""You are a patient, encouraging AI tutor specializing in {self.subject} for BEGINNER learners.

TEACHING APPROACH FOR BEGINNERS:
- Use simple, clear language and short sentences
- Define all technical terms immediately
- Use lots of examples and analogies from everyday life
- Break complex ideas into small, easy steps
- Include encouragement and positive reinforcement
- Repeat key concepts in different ways
- Use bullet points and clear structure

RESPONSE STYLE:
- Start with encouragement
- Explain concepts step-by-step
- Use "Let's explore..." or "Think of it like..." phrases
- Include emojis for engagement
- End with encouragement and next steps""",

            'intermediate': f"""You are a knowledgeable AI tutor specializing in {self.subject} for INTERMEDIATE learners.

TEACHING APPROACH FOR INTERMEDIATE:
- Use clear, moderately technical language
- Build on existing knowledge
- Include practical applications and examples
- Balance depth with accessibility
- Connect concepts to broader understanding
- Use structured explanations

RESPONSE STYLE:
- Professional but approachable
- Well-organized with clear sections
- Include real-world connections
- Encourage deeper thinking
- Provide pathways for further learning""",

            'advanced': f"""You are an expert AI tutor specializing in {self.subject} for ADVANCED learners.

TEACHING APPROACH FOR ADVANCED:
- Use precise technical terminology
- Assume strong foundational knowledge
- Focus on complex relationships and implications
- Include current research and developments
- Challenge assumptions and encourage critical thinking
- Provide comprehensive analysis

RESPONSE STYLE:
- Scholarly and sophisticated
- Detailed analysis with multiple perspectives
- Include research references when relevant
- Encourage independent investigation
- Focus on synthesis and evaluation"""
        }

        system_prompt = difficulty_system_prompts.get(self.difficulty_level, difficulty_system_prompts['intermediate'])

        system_prompt += f"""

STUDENT QUESTION: {query}

TEXTBOOK CONTENT (Primary Source - {self.difficulty_level} level):
{textbook_content}

EXTERNAL SOURCES (For Enhancement):
{external_content}

TASK: Create a comprehensive educational response that:
- Prioritizes textbook content but enhances with external sources
- Matches {self.difficulty_level} level perfectly
- Answers the student's question thoroughly
- Uses appropriate vocabulary and complexity for {self.difficulty_level} learners
- Includes examples appropriate for the difficulty level
- Provides clear structure and organization
- Encourages continued learning
- Clearly indicates information sources"""

        try:
            response = self._call_groq_api_enhanced(query, system_prompt)
            return response
        except Exception as e:
            return self._generate_fallback_response_enhanced(query, textbook_context)

    def _generate_multi_source_response(self, query, textbook_context, external_context):
        """Generate response combining multiple sources (original method)"""
        
        # Prepare textbook content
        textbook_content = ""
        for chunk in textbook_context:
            textbook_content += f"From {chunk.get('source', 'textbook')}: {chunk.get('content', '')}\n\n"
        
        # Prepare external content
        external_content = ""
        
        # Add Wikipedia content
        wiki_info = external_context.get('wikipedia', {})
        if wiki_info.get('available'):
            external_content += f"Wikipedia Context: {wiki_info.get('summary', '')}\n\n"
        
        # Add subject-specific content
        subject_info = external_context.get('subject_specific', {})
        if 'content' in subject_info:
            external_content += f"Additional {self.subject} Context: {subject_info['content']}\n\n"
        
        system_prompt = f"""You are an expert AI tutor specializing in {self.subject}. Your goal is to provide comprehensive, accurate, and understandable educational responses.

RESPONSE GUIDELINES:
1. **Primary Source**: Start with and prioritize textbook content
2. **External Enhancement**: Use external sources to provide additional context and examples
3. **Source Attribution**: Clearly indicate information sources
4. **Accuracy**: Ensure all information is factually correct
5. **Clarity**: Explain concepts clearly and appropriately for the student level
6. **Engagement**: Make learning interesting and memorable

STUDENT QUESTION: {query}

TEXTBOOK CONTENT (Primary Source):
{textbook_content}

EXTERNAL SOURCES (For Enhancement):
{external_content}

TASK: Create a comprehensive educational response that:
- Answers the student's question thoroughly
- Uses primarily textbook content with external enhancement
- Includes real-world examples where appropriate
- Explains complex concepts in understandable terms
- Encourages further learning
- Clearly indicates sources of information"""

        try:
            response = self._call_groq_api(query, system_prompt)
            return response
        except Exception as e:
            return self._generate_fallback_response(query, textbook_context)

    def _adapt_response_to_difficulty(self, response, original_query):
        """NEW: Adapt response complexity to match difficulty level"""
        
        if not self.difficulty_adapter:
            return self._simplify_for_understanding(response, original_query)
        
        try:
            adapted_response = self.difficulty_adapter.adapt_response_complexity(
                response, self.difficulty_level, original_query
            )
            return adapted_response
        except Exception as e:
            st.warning(f"Difficulty adaptation failed: {e}")
            return self._simplify_for_understanding(response, original_query)

    def _simplify_for_understanding(self, response, original_query):
        """Ensure response is clear and understandable (enhanced)"""
        
        # Difficulty-specific simplification prompts
        simplification_prompts = {
            'beginner': f"""Make this educational response perfect for BEGINNER learners:

ORIGINAL RESPONSE: {response}

STUDENT QUESTION: {original_query}

BEGINNER IMPROVEMENTS:
1. Use very simple language (avoid complex words)
2. Break into small, easy sections with clear headings
3. Add lots of examples from everyday life
4. Explain ALL technical terms immediately
5. Add encouraging language throughout
6. Use bullet points and short paragraphs
7. Include emojis for engagement
8. End with "Great question!" and encouragement

Keep all important information but make it very easy to understand.""",

            'intermediate': f"""Improve this educational response for INTERMEDIATE learners:

ORIGINAL RESPONSE: {response}

STUDENT QUESTION: {original_query}

INTERMEDIATE IMPROVEMENTS:
1. Use clear, moderately technical language
2. Add clear structure with headings
3. Include practical examples and applications
4. Explain technical terms when first used
5. Add connections to broader concepts
6. Make it engaging and well-organized
7. Include pathways for further learning

Return a well-structured, clear version.""",

            'advanced': f"""Refine this educational response for ADVANCED learners:

ORIGINAL RESPONSE: {response}

STUDENT QUESTION: {original_query}

ADVANCED IMPROVEMENTS:
1. Use precise technical terminology appropriately
2. Add sophisticated analysis and connections
3. Include implications and broader context
4. Reference current research or developments
5. Encourage critical thinking and deeper investigation
6. Provide comprehensive, scholarly perspective
7. Challenge assumptions where appropriate

Return a sophisticated, comprehensive version."""
        }

        prompt = simplification_prompts.get(self.difficulty_level, simplification_prompts['intermediate'])
        
        system_prompt = f"You are an expert at adapting educational content for {self.difficulty_level} level learners. Focus on making content perfectly suited for this learning level."
        
        try:
            simplified = self._call_groq_api_enhanced(prompt, system_prompt)
            return simplified
        except:
            return response  # Return original if simplification fails

    def _track_conversation_enhanced(self, user_query, response, textbook_context, external_context):
        """NEW: Enhanced conversation tracking with detailed metadata"""
        
        # Extract topics from query
        query_topics = self._extract_topics_from_query(user_query)
        self.session_stats['topics_discussed'].update(query_topics)
        
        conversation_entry = {
            'user': user_query,
            'assistant': response,
            'timestamp': datetime.now().isoformat(),
            'difficulty_level': self.difficulty_level,
            'sources_used': self._count_sources_used(textbook_context, external_context),
            'textbook_sources': [ctx.get('source', 'unknown') for ctx in textbook_context],
            'external_sources': list(external_context.keys()),
            'topics_discussed': list(query_topics),
            'response_length': len(response),
            'processing_enhanced': True
        }
        
        self.conversation_history.append(conversation_entry)
        self.session_stats['total_questions'] += 1
        
        # Keep conversation history manageable
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-40:]

    def _extract_topics_from_query(self, query):
        """Extract main topics from user query"""
        # Simple topic extraction based on keywords
        topics = set()
        
        # Extract subject-specific terms
        subject_keywords = {
            'Biology': ['cell', 'organism', 'evolution', 'genetics', 'ecology', 'metabolism'],
            'Chemistry': ['atom', 'molecule', 'reaction', 'bond', 'element', 'compound'],
            'Physics': ['force', 'energy', 'motion', 'wave', 'particle', 'field'],
            'Mathematics': ['equation', 'function', 'theorem', 'proof', 'calculation', 'formula'],
            'History': ['event', 'period', 'civilization', 'war', 'revolution', 'empire'],
            'Literature': ['character', 'theme', 'plot', 'author', 'style', 'genre']
        }
        
        query_lower = query.lower()
        keywords = subject_keywords.get(self.subject, [])
        
        for keyword in keywords:
            if keyword in query_lower:
                topics.add(keyword)
        
        # Add general academic terms
        general_terms = ['concept', 'theory', 'principle', 'method', 'process', 'system']
        for term in general_terms:
            if term in query_lower:
                topics.add(term)
        
        return topics

    def set_difficulty_level(self, new_difficulty):
        """NEW: Set new difficulty level and adapt accordingly"""
        old_difficulty = self.difficulty_level
        self.difficulty_level = new_difficulty
        
        # Update difficulty adapter if available
        if self.difficulty_adapter:
            self.difficulty_adapter.set_difficulty_level(new_difficulty)
        
        # Log the change
        self.session_stats['difficulty_adaptations'] += 1
        
        return f"✅ Difficulty level changed from {old_difficulty} to {new_difficulty}! I'll now provide {new_difficulty}-level responses tailored to your learning needs."

    def enable_voice_chat(self):
        """NEW: Enable voice chat functionality"""
        if self.voice_processor:
            self.external_sources['voice_enabled'] = True
            return "🎤 Voice chat enabled! You can now ask questions using voice input and receive voice responses."
        else:
            return "❌ Voice chat not available. Please ensure voice processing modules are installed."

    def disable_voice_chat(self):
        """NEW: Disable voice chat functionality"""
        self.external_sources['voice_enabled'] = False
        return "🔇 Voice chat disabled. Continuing with text-based interaction."

    def get_personalized_study_suggestions(self):
        """NEW: Get personalized study suggestions based on conversation history"""
        
        if not self.conversation_history:
            return "Start asking questions to receive personalized study suggestions! 📚"
        
        # Analyze conversation patterns
        topics_discussed = self.session_stats['topics_discussed']
        total_questions = self.session_stats['total_questions']
        difficulty = self.difficulty_level
        
        suggestions = []
        
        # Difficulty-specific suggestions
        if difficulty == 'beginner':
            suggestions.append("📚 Focus on building strong foundational concepts")
            suggestions.append("🎯 Practice with simple examples and basic terminology")
            suggestions.append("⏰ Take breaks between topics to consolidate learning")
        elif difficulty == 'intermediate':
            suggestions.append("🔗 Work on connecting different concepts together")
            suggestions.append("🎯 Practice applying concepts to real-world problems")
            suggestions.append("📊 Try creating concept maps to visualize relationships")
        else:  # advanced
            suggestions.append("🔬 Explore current research in areas of interest")
            suggestions.append("🎯 Focus on critical analysis and synthesis")
            suggestions.append("💡 Consider interdisciplinary connections")
        
        # Topic-specific suggestions
        if len(topics_discussed) > 0:
            suggestions.append(f"📖 Continue exploring: {', '.join(list(topics_discussed)[:3])}")
        
        # Engagement suggestions
        if total_questions >= 5:
            suggestions.append("🎬 Try generating educational videos on your favorite topics")
            suggestions.append("🧠 Test your knowledge with difficulty-appropriate quizzes")
        
        return "\n".join([f"• {suggestion}" for suggestion in suggestions])

    def get_session_analytics(self):
        """NEW: Get detailed session analytics"""
        
        duration = datetime.now() - self.session_stats['session_start']
        
        analytics = {
            'session_duration': str(duration).split('.')[0],  # Remove microseconds
            'total_questions': self.session_stats['total_questions'],
            'voice_interactions': self.session_stats['voice_interactions'],
            'difficulty_level': self.difficulty_level,
            'topics_explored': len(self.session_stats['topics_discussed']),
            'sources_accessed': len(self.session_stats['sources_accessed']),
            'avg_response_length': 0,
            'most_discussed_topics': list(self.session_stats['topics_discussed'])[:5]
        }
        
        # Calculate average response length
        if self.conversation_history:
            total_length = sum(len(entry.get('assistant', '')) for entry in self.conversation_history)
            analytics['avg_response_length'] = total_length // len(self.conversation_history)
        
        return analytics

    def switch_subject_enhanced(self, new_subject):
        """NEW: Enhanced subject switching with difficulty awareness"""
        old_subject = self.subject
        self.subject = new_subject
        
        # Reset subject-specific stats
        self.session_stats['topics_discussed'] = set()
        
        # Update external sources
        self.external_sources['subject_specific'] = True
        
        return f"✅ Subject changed from {old_subject} to {new_subject}! I'm now optimized for {self.difficulty_level}-level {new_subject} questions using textbook content and specialized {new_subject} resources."

    def switch_subject(self, new_subject):
        """Switch to a different subject context (original method)"""
        self.subject = new_subject
        return f"✅ Switched to {new_subject}! I'm now optimized to help with {new_subject} questions using both textbook content and additional {new_subject} resources."

    def get_conversation_summary_enhanced(self):
        """NEW: Enhanced conversation summary with analytics"""
        if not self.conversation_history:
            return "No conversation history yet. Start asking questions to build your learning session! 🚀"
        
        analytics = self.get_session_analytics()
        
        summary = f"""📊 **Enhanced Learning Session Summary**

🎯 **Session Overview**
• Duration: {analytics['session_duration']}
• Questions Asked: {analytics['total_questions']}
• Difficulty Level: {self.difficulty_level.title()}
• Subject: {self.subject}

🔊 **Interaction Statistics**
• Voice Interactions: {analytics['voice_interactions']}
• Sources Accessed: {', '.join(self.session_stats['sources_accessed']) if self.session_stats['sources_accessed'] else 'Textbook only'}
• Topics Explored: {analytics['topics_explored']}

📚 **Learning Focus Areas**
{', '.join(analytics['most_discussed_topics']) if analytics['most_discussed_topics'] else 'No specific topics identified yet'}

📈 **Recent Conversation History**"""
        
        # Add recent exchanges
        for i, exchange in enumerate(self.conversation_history[-3:], 1):
            summary += f"\n**{i}.** *{exchange['user'][:80]}{'...' if len(exchange['user']) > 80 else ''}*"
            summary += f"\n   📚 Sources: {exchange.get('sources_used', 'Unknown')} | ⏰ {exchange.get('timestamp', 'N/A')[:19]}"
        
        return summary

    def get_conversation_summary(self):
        """Get summary of current conversation (original method)"""
        if not self.conversation_history:
            return "No conversation history yet."
        
        summary = f"📊 **Conversation Summary** ({len(self.conversation_history)} exchanges)\n\n"
        
        for i, exchange in enumerate(self.conversation_history[-5:], 1):
            summary += f"**{i}.** Student asked: *{exchange['user'][:60]}...*\n"
            summary += f"   Sources used: {exchange.get('sources_used', 'Unknown')}\n"
            summary += f"   Time: {exchange.get('timestamp', 'N/A')}\n\n"
        
        return summary

    def _count_sources_used(self, textbook_context, external_context):
        """Count number of sources used in response"""
        count = 0
        
        # Count textbook sources
        if textbook_context and len(textbook_context) > 0:
            count += len([c for c in textbook_context if c.get('content') and len(c['content']) > 50])
        
        # Count external sources
        for source_type, source_data in external_context.items():
            if isinstance(source_data, dict) and (source_data.get('available') or source_data.get('content')):
                count += 1
        
        return count

    def _generate_fallback_response_enhanced(self, query, textbook_context):
        """NEW: Enhanced fallback response generation"""
        
        if not textbook_context or textbook_context[0].get('content') == 'No textbook content available.':
            return f"""I don't have information about that topic in the current textbook for {self.difficulty_level} level learners. 

Here are some suggestions:
• Try asking about a different topic from the textbook
• Upload a textbook that covers this subject
• Adjust your difficulty level if needed
• Rephrase your question using different terms

I'm here to help with your {self.difficulty_level}-level {self.subject} learning! 📚"""
        
        # Use first textbook chunk with difficulty awareness
        content = textbook_context[0].get('content', '')
        source = textbook_context[0].get('source', 'textbook')
        difficulty = textbook_context[0].get('difficulty', self.difficulty_level)
        
        # Adapt response based on difficulty level
        if self.difficulty_level == 'beginner':
            response = f"""🌟 **Great question!** Let me help you understand this step by step.

📚 From your textbook ({source}):

{content[:300]}...

**Let me break this down for you:**
• This is a {difficulty}-level topic, perfect for building your foundation
• The key idea is about "{query}"
• Don't worry if it seems complex at first - we'll work through it together!

💡 **What would you like me to explain more clearly?** I'm here to help you succeed! 🎯"""

        elif self.difficulty_level == 'advanced':
            response = f"""📚 **Textbook Analysis** (Source: {source})

{content[:500]}...

**Advanced Perspective:**
This {difficulty}-level content addresses your question about "{query}" with comprehensive detail. The information provides a solid foundation for deeper analysis.

**For Further Investigation:**
• Consider the broader implications of this concept
• Explore connections to related theoretical frameworks
• Examine current research developments in this area

**Ready for deeper discussion?** What specific aspects would you like to analyze further? 🔬"""

        else:  # intermediate
            response = f"""📚 **From your textbook** ({source}):

{content[:400]}...

This information should help answer your question about "{query}". The content is at {difficulty} level, which aligns well with building strong conceptual understanding.

**Key takeaways:**
• This concept is fundamental to understanding {self.subject}
• The explanation builds on previous knowledge
• There are practical applications to explore

💡 **Would you like me to explain any specific part in more detail?** Feel free to ask follow-up questions! 🎯"""
        
        return response

    def _generate_fallback_response(self, query, textbook_context):
        """Generate basic fallback response (original method)"""
        if not textbook_context or textbook_context[0].get('content') == 'No textbook content available.':
            return "I don't have enough information about that topic in the current textbook. Could you try asking about a different topic, or upload a textbook that covers this subject?"
        
        # Use first textbook chunk
        content = textbook_context[0].get('content', '')
        source = textbook_context[0].get('source', 'textbook')
        
        return f"""Based on the textbook content from {source}:

{content[:400]}...

This should help answer your question about "{query}". Would you like me to explain any specific part in more detail?

💡 *Feel free to ask follow-up questions to dive deeper into this topic!*"""

    def _get_fallback_response_enhanced(self, query):
        """NEW: Enhanced simple fallback response"""
        
        difficulty_messages = {
            'beginner': f"""🌟 I'm having a little trouble with your question "{query}" right now, but don't worry!

**Let's try these simple steps:**
• Ask about a basic topic from your textbook
• Use simple, clear words in your question
• Make sure your textbook has been uploaded and processed

Remember, every expert was once a beginner! I'm here to help you learn step by step. 📚✨""",

            'intermediate': f"""I'm having trouble processing your question "{query}" at the moment.

**Here are some helpful suggestions:**
• Rephrase your question with different terms
• Ask about a specific concept from your textbook
• Check that your textbook content has been properly loaded
• Try breaking complex questions into smaller parts

I'm ready to help with your {self.subject} learning when you're ready! 🎯""",

            'advanced': f"""I'm currently unable to process your sophisticated question about "{query}".

**Troubleshooting options:**
• Verify textbook preprocessing completed successfully
• Reformulate query with alternative terminology
• Consider whether this topic requires additional specialized resources
• Break complex multi-part questions into focused components

I'm equipped to handle advanced {self.subject} discussions once we resolve this technical issue. 🔬"""
        }
        
        return difficulty_messages.get(self.difficulty_level, 
                                     f"I'm having trouble processing your question '{query}' right now. Please try rephrasing or ask about a topic from your textbook. I'm here to help! 📚")

    def _get_fallback_response(self, query):
        """Simple fallback response (original method)"""
        return f"""I'm having trouble processing your question "{query}" right now. 

Here are some things you can try:
• Rephrase your question
• Ask about a specific topic from the textbook
• Check if the textbook has been properly processed

I'm here to help with your learning! 📚"""

    def _call_groq_api_enhanced(self, content, system_prompt):
        """NEW: Enhanced Groq API call with difficulty-specific parameters"""
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.GROQ_API_KEY}"
        }
        
        # Difficulty-specific parameters
        difficulty_params = {
            'beginner': {'max_tokens': 800, 'temperature': 0.5},
            'intermediate': {'max_tokens': 1000, 'temperature': 0.6},
            'advanced': {'max_tokens': 1200, 'temperature': 0.7}
        }
        
        params = difficulty_params.get(self.difficulty_level, difficulty_params['intermediate'])
        
        payload = {
            "model": self.config.GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            "max_tokens": params['max_tokens'],
            "temperature": params['temperature']
        }
        
        response = requests.post(
            f"{self.config.GROQ_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            raise Exception(f"Enhanced API call failed: {response.status_code}")

    def _call_groq_api(self, content, system_prompt):
        """Call Groq API for response generation (original method)"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.GROQ_API_KEY}"
        }
        
        payload = {
            "model": self.config.GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            "max_tokens": self.config.MAX_TOKENS,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{self.config.GROQ_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            raise Exception(f"API call failed: {response.status_code}")

    # Additional utility methods for enhanced functionality
    def export_conversation_history(self):
        """NEW: Export conversation history for analysis"""
        
        export_data = {
            'session_info': {
                'subject': self.subject,
                'difficulty_level': self.difficulty_level,
                'session_start': self.session_stats['session_start'].isoformat(),
                'export_time': datetime.now().isoformat()
            },
            'conversation_history': self.conversation_history,
            'session_analytics': self.get_session_analytics(),
            'platform_version': 'IntelliLearn AI Enhanced v2.0'
        }
        
        return json.dumps(export_data, indent=2)

    def reset_conversation(self):
        """NEW: Reset conversation while preserving settings"""
        self.conversation_history = []
        self.session_stats = {
            'total_questions': 0,
            'voice_interactions': 0,
            'difficulty_adaptations': 0,
            'sources_accessed': set(),
            'session_start': datetime.now(),
            'topics_discussed': set()
        }
        
        return f"🔄 Conversation reset! Ready for a fresh {self.difficulty_level}-level {self.subject} learning session."

    def get_available_features(self):
        """NEW: Get list of available enhanced features"""
        
        features = {
            'core_features': [
                '📚 Multi-source content integration',
                '🎯 Difficulty-adaptive responses',
                '🔍 Smart textbook search',
                '📊 Conversation analytics',
                '💾 Session export'
            ],
            'enhanced_features': []
        }
        
        if ENHANCED_MODULES_AVAILABLE:
            if self.voice_processor:
                features['enhanced_features'].append('🎤 Voice chat capabilities')
            if self.difficulty_adapter:
                features['enhanced_features'].append('🎯 Advanced difficulty adaptation')
            if self.enhanced_nlp:
                features['enhanced_features'].append('🧠 Enhanced NLP processing')
        
        if self.external_sources['wikipedia']:
            features['enhanced_features'].append('📖 Wikipedia integration')
        if self.external_sources['web_search']:
            features['enhanced_features'].append('🌐 Web search (advanced level)')
        
        return features
