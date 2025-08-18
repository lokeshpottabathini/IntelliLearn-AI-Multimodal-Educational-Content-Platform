import streamlit as st
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import re
from typing import List, Dict, Tuple, Optional
import json

# Try to import advanced NLP libraries
try:
    from transformers import (
        AutoModel, AutoTokenizer, AutoModelForTokenClassification,
        pipeline, BertModel, RobertaModel, AutoModelForSequenceClassification
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from textstat import flesch_reading_ease, flesch_kincaid_grade
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False

class EnhancedNLPProcessor:
    def __init__(self):
        """Initialize enhanced NLP processor with available models"""
        
        self.models = {}
        self.processors = {}
        
        # Initialize Sentence Transformers
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                st.success("✅ Sentence Transformers loaded")
            except Exception as e:
                st.warning(f"Sentence Transformers failed: {e}")
                self.sentence_model = None
        else:
            self.sentence_model = None
        
        # Initialize BERTopic for advanced topic modeling
        if BERTOPIC_AVAILABLE and self.sentence_model:
            try:
                # Custom vectorizer for educational content
                vectorizer_model = CountVectorizer(
                    ngram_range=(1, 3),
                    stop_words="english",
                    min_df=2,
                    max_features=1000
                )
                
                self.topic_model = BERTopic(
                    embedding_model=self.sentence_model,
                    vectorizer_model=vectorizer_model,
                    verbose=True,
                    calculate_probabilities=True
                )
                st.success("✅ BERTopic model initialized")
            except Exception as e:
                st.warning(f"BERTopic initialization failed: {e}")
                self.topic_model = None
        else:
            self.topic_model = None
        
        # Initialize spaCy if available
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                import en_core_web_sm
                self.nlp = en_core_web_sm.load()
                st.success("✅ spaCy model loaded")
            except:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    st.success("✅ spaCy model loaded")
                except:
                    st.warning("⚠️ spaCy model not available")
                    self.nlp = None
        
        # Initialize transformers models
        self._initialize_transformer_models()
        
        # Educational domain vocabulary
        self.educational_terms = self._load_educational_vocabulary()
        
        # Difficulty indicators
        self.difficulty_indicators = self._load_difficulty_indicators()
    
    def _initialize_transformer_models(self):
        """Initialize transformer models for various NLP tasks"""
        
        if not TRANSFORMERS_AVAILABLE:
            return
        
        try:
            # Text classification for educational content
            self.education_classifier = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium",
                return_all_scores=True
            )
            
            # Named Entity Recognition
            self.ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple"
            )
            
            # Question answering for educational content
            self.qa_pipeline = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad"
            )
            
            st.success("✅ Transformer models loaded successfully")
            
        except Exception as e:
            st.warning(f"Some transformer models failed to load: {e}")
    
    def _load_educational_vocabulary(self):
        """Load educational domain vocabulary"""
        
        return {
            'biology': [
                'photosynthesis', 'mitosis', 'meiosis', 'chromosome', 'gene', 'DNA', 'RNA',
                'protein', 'enzyme', 'cell', 'nucleus', 'mitochondria', 'chloroplast',
                'evolution', 'natural selection', 'ecosystem', 'biodiversity', 'organism',
                'population', 'community', 'habitat', 'adaptation', 'mutation'
            ],
            'chemistry': [
                'atom', 'molecule', 'element', 'compound', 'reaction', 'bond', 'electron',
                'proton', 'neutron', 'periodic table', 'oxidation', 'reduction', 'acid',
                'base', 'pH', 'catalyst', 'equilibrium', 'thermodynamics', 'kinetics',
                'organic', 'inorganic', 'polymer', 'crystallization'
            ],
            'physics': [
                'force', 'energy', 'momentum', 'velocity', 'acceleration', 'wave',
                'frequency', 'amplitude', 'gravity', 'magnetism', 'electricity', 'quantum',
                'relativity', 'thermodynamics', 'optics', 'mechanics', 'particle',
                'field', 'radiation', 'nuclear', 'atomic', 'electromagnetic'
            ],
            'mathematics': [
                'equation', 'function', 'derivative', 'integral', 'limit', 'matrix',
                'vector', 'polynomial', 'logarithm', 'exponential', 'trigonometry',
                'geometry', 'algebra', 'calculus', 'statistics', 'probability',
                'theorem', 'proof', 'axiom', 'hypothesis', 'variable', 'constant'
            ]
        }
    
    def _load_difficulty_indicators(self):
        """Load difficulty assessment indicators"""
        
        return {
            'beginner': {
                'vocabulary': ['basic', 'simple', 'easy', 'introduction', 'fundamental', 'overview'],
                'sentence_length': (5, 15),
                'syllables_per_word': (1, 2),
                'technical_terms': (0, 5)
            },
            'intermediate': {
                'vocabulary': ['moderate', 'standard', 'analysis', 'comparison', 'detailed'],
                'sentence_length': (10, 25),
                'syllables_per_word': (2, 3),
                'technical_terms': (5, 15)
            },
            'advanced': {
                'vocabulary': ['complex', 'sophisticated', 'comprehensive', 'synthesis', 'evaluation'],
                'sentence_length': (15, 40),
                'syllables_per_word': (3, 5),
                'technical_terms': (15, 50)
            }
        }
    
    def extract_educational_topics_advanced(self, texts, difficulty_level="intermediate"):
        """Advanced topic extraction using BERTopic and educational context"""
        
        if not texts or not self.topic_model:
            return self._fallback_topic_extraction(texts, difficulty_level)
        
        try:
            st.info("🧠 Performing advanced topic modeling...")
            
            # Clean and prepare texts
            cleaned_texts = [self._clean_educational_text(text) for text in texts]
            cleaned_texts = [text for text in cleaned_texts if len(text.split()) > 10]
            
            if len(cleaned_texts) < 3:
                return self._fallback_topic_extraction(texts, difficulty_level)
            
            # Fit BERTopic model
            topics, probabilities = self.topic_model.fit_transform(cleaned_texts)
            
            # Get topic information
            topic_info = self.topic_model.get_topic_info()
            
            # Enhanced topic processing
            educational_topics = []
            
            for idx, row in topic_info.iterrows():
                if row['Topic'] != -1:  # Exclude outlier topic
                    topic_words = [word for word, _ in self.topic_model.get_topic(row['Topic'])]
                    
                    topic_data = {
                        'topic_id': row['Topic'],
                        'keywords': topic_words[:10],
                        'name': self._generate_educational_topic_name(topic_words),
                        'count': row['Count'],
                        'educational_category': self._categorize_educational_topic(topic_words),
                        'difficulty_level': self._assess_topic_difficulty(topic_words, difficulty_level),
                        'related_concepts': self._find_related_concepts(topic_words),
                        'learning_objectives': self._generate_learning_objectives(topic_words),
                        'representative_docs': self._get_representative_documents(
                            row['Topic'], cleaned_texts, topics
                        ),
                        'confidence_score': self._calculate_topic_confidence(row, probabilities)
                    }
                    
                    educational_topics.append(topic_data)
            
            # Sort by educational relevance and confidence
            educational_topics.sort(
                key=lambda x: (x['confidence_score'], x['count']), 
                reverse=True
            )
            
            st.success(f"✅ Extracted {len(educational_topics)} educational topics")
            return educational_topics[:15]  # Return top 15 topics
            
        except Exception as e:
            st.warning(f"Advanced topic extraction failed: {e}")
            return self._fallback_topic_extraction(texts, difficulty_level)
    
    def _clean_educational_text(self, text):
        """Clean text specifically for educational content analysis"""
        
        if not text:
            return ""
        
        # Remove non-educational noise
        text = re.sub(r'\b(page|figure|table|chapter|section)\s+\d+\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\d{3,}', '', text)  # Remove long numbers
        text = re.sub(r'[^\w\s\.]', ' ', text)  # Keep only words and periods
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        # Remove very short and very long sentences
        sentences = text.split('.')
        filtered_sentences = [s.strip() for s in sentences 
                            if 5 <= len(s.split()) <= 50]
        
        return '. '.join(filtered_sentences)
    
    def _generate_educational_topic_name(self, topic_words):
        """Generate meaningful educational topic names"""
        
        if not topic_words:
            return "General Topic"
        
        # Check for subject-specific keywords
        for subject, keywords in self.educational_terms.items():
            if any(word.lower() in [kw.lower() for kw in keywords] for word in topic_words):
                primary_concept = next((word for word in topic_words 
                                      if word.lower() in [kw.lower() for kw in keywords]), 
                                     topic_words[0])
                return f"{subject.title()}: {primary_concept.title()}"
        
        # Generate descriptive name from top words
        top_words = [word.title() for word in topic_words[:3] if len(word) > 3]
        
        if len(top_words) >= 2:
            return f"{top_words[0]} and {top_words[1]}"
        elif top_words:
            return f"{top_words[0]} Concepts"
        else:
            return "Educational Topic"
    
    def _categorize_educational_topic(self, topic_words):
        """Categorize topic into educational domain"""
        
        word_set = set(word.lower() for word in topic_words)
        
        category_scores = {}
        
        for subject, keywords in self.educational_terms.items():
            keyword_set = set(kw.lower() for kw in keywords)
            overlap = len(word_set.intersection(keyword_set))
            category_scores[subject] = overlap / len(keyword_set) if keyword_set else 0
        
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])
            if best_category[1] > 0.1:  # Minimum confidence threshold
                return best_category[0]
        
        return 'general'
    
    def _assess_topic_difficulty(self, topic_words, target_difficulty):
        """Assess topic difficulty level"""
        
        word_set = set(word.lower() for word in topic_words)
        difficulty_scores = {}
        
        for level, indicators in self.difficulty_indicators.items():
            vocabulary_overlap = len(word_set.intersection(set(indicators['vocabulary'])))
            
            # Count technical terms
            technical_count = sum(1 for word in topic_words 
                                if len(word) > 8 or any(suffix in word.lower() 
                                for suffix in ['tion', 'sion', 'ism', 'ogy', 'graphy']))
            
            # Calculate complexity score
            avg_word_length = np.mean([len(word) for word in topic_words])
            
            difficulty_scores[level] = (
                vocabulary_overlap * 0.4 +
                (1 if indicators['technical_terms'][0] <= technical_count <= indicators['technical_terms'][1] else 0) * 0.3 +
                (1 if avg_word_length >= 5 else 0) * 0.3
            )
        
        # Return the level with highest score, or target difficulty if close
        best_level = max(difficulty_scores.items(), key=lambda x: x[1])[0]
        
        # If target difficulty has reasonable score, prefer it
        if difficulty_scores.get(target_difficulty, 0) >= 0.3:
            return target_difficulty
        
        return best_level
    
    def _find_related_concepts(self, topic_words):
        """Find related educational concepts"""
        
        if not self.sentence_model:
            return []
        
        try:
            # Create embeddings for topic words
            topic_text = ' '.join(topic_words)
            topic_embedding = self.sentence_model.encode([topic_text])
            
            # Find similar concepts from educational vocabulary
            all_concepts = []
            for subject_concepts in self.educational_terms.values():
                all_concepts.extend(subject_concepts)
            
            # Calculate similarities
            concept_embeddings = self.sentence_model.encode(all_concepts)
            similarities = self.sentence_model.similarity(topic_embedding, concept_embeddings)
            
            # Get top related concepts
            top_indices = similarities[0].argsort()[-5:][::-1]
            related_concepts = [all_concepts[i] for i in top_indices 
                              if similarities[0][i] > 0.5]
            
            return related_concepts
            
        except Exception:
            return []
    
    def _generate_learning_objectives(self, topic_words):
        """Generate learning objectives for topics"""
        
        # Bloom's taxonomy action verbs by level
        bloom_verbs = {
            'remember': ['identify', 'list', 'describe', 'define', 'recognize'],
            'understand': ['explain', 'interpret', 'summarize', 'compare', 'discuss'],
            'apply': ['demonstrate', 'solve', 'use', 'implement', 'calculate'],
            'analyze': ['analyze', 'examine', 'investigate', 'categorize', 'differentiate'],
            'evaluate': ['evaluate', 'assess', 'critique', 'judge', 'justify'],
            'create': ['design', 'construct', 'develop', 'create', 'formulate']
        }
        
        objectives = []
        primary_concept = topic_words[0] if topic_words else "concept"
        
        # Generate objectives for different Bloom's levels
        for level, verbs in list(bloom_verbs.items())[:3]:  # Use first 3 levels
            verb = np.random.choice(verbs)
            objective = f"Students will be able to {verb} {primary_concept}"
            objectives.append({
                'objective': objective,
                'bloom_level': level,
                'concepts': topic_words[:3]
            })
        
        return objectives
    
    def _get_representative_documents(self, topic_id, documents, topic_assignments):
        """Get representative documents for a topic"""
        
        if not documents or not topic_assignments.any():
            return []
        
        # Find documents assigned to this topic
        topic_docs = [doc for i, doc in enumerate(documents) 
                     if i < len(topic_assignments) and topic_assignments[i] == topic_id]
        
        # Return first few representative documents
        return topic_docs[:3] if topic_docs else []
    
    def _calculate_topic_confidence(self, topic_row, probabilities):
        """Calculate confidence score for topic"""
        
        try:
            if probabilities is not None and len(probabilities) > 0:
                # Average probability for this topic
                topic_id = topic_row['Topic']
                if topic_id >= 0 and topic_id < probabilities.shape[1]:
                    return np.mean(probabilities[:, topic_id])
            
            # Fallback: use count-based confidence
            return min(topic_row['Count'] / 10, 1.0)
            
        except Exception:
            return 0.5  # Default medium confidence
    
    def _fallback_topic_extraction(self, texts, difficulty_level):
        """Fallback topic extraction using basic NLP"""
        
        st.info("🔄 Using fallback topic extraction...")
        
        if not texts:
            return []
        
        # Combine all texts
        combined_text = ' '.join(texts)
        
        # Use spaCy for basic topic extraction if available
        if self.nlp:
            topics = self._extract_topics_with_spacy(combined_text, difficulty_level)
        else:
            topics = self._extract_topics_basic_nlp(combined_text, difficulty_level)
        
        return topics
    
    def _extract_topics_with_spacy(self, text, difficulty_level):
        """Extract topics using spaCy"""
        
        try:
            doc = self.nlp(text)
            
            # Extract noun phrases as potential topics
            noun_phrases = [chunk.text for chunk in doc.noun_chunks 
                          if len(chunk.text.split()) <= 3 and len(chunk.text) > 3]
            
            # Extract named entities
            entities = [ent.text for ent in doc.ents 
                       if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']]
            
            # Combine and filter
            all_topics = noun_phrases + entities
            topic_counts = Counter(all_topics)
            
            # Create topic objects
            topics = []
            for i, (topic, count) in enumerate(topic_counts.most_common(10)):
                topics.append({
                    'topic_id': i,
                    'keywords': topic.split(),
                    'name': topic.title(),
                    'count': count,
                    'educational_category': 'general',
                    'difficulty_level': difficulty_level,
                    'related_concepts': [],
                    'learning_objectives': [],
                    'representative_docs': [],
                    'confidence_score': min(count / 10, 1.0)
                })
            
            return topics
            
        except Exception as e:
            st.warning(f"spaCy topic extraction failed: {e}")
            return self._extract_topics_basic_nlp(text, difficulty_level)
    
    def _extract_topics_basic_nlp(self, text, difficulty_level):
        """Basic topic extraction using keyword analysis"""
        
        try:
            # Basic preprocessing
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            
            # Remove common stop words
            stop_words = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'more', 'very', 'what', 'know', 'just', 'than', 'only', 'other', 'take', 'come', 'could', 'them', 'some', 'make', 'like', 'into', 'even', 'also', 'back', 'after', 'first', 'well', 'work', 'through', 'way'}
            
            filtered_words = [word for word in words if word not in stop_words]
            
            # Count word frequencies
            word_counts = Counter(filtered_words)
            
            # Create basic topics from most frequent words
            topics = []
            for i, (word, count) in enumerate(word_counts.most_common(8)):
                topics.append({
                    'topic_id': i,
                    'keywords': [word],
                    'name': word.title(),
                    'count': count,
                    'educational_category': 'general',
                    'difficulty_level': difficulty_level,
                    'related_concepts': [],
                    'learning_objectives': [],
                    'representative_docs': [],
                    'confidence_score': min(count / 20, 1.0)
                })
            
            return topics
            
        except Exception as e:
            st.error(f"Basic topic extraction failed: {e}")
            return []
    
    def extract_educational_concepts_advanced(self, text, subject="general"):
        """Advanced concept extraction using multiple methods"""
        
        if not text:
            return []
        
        concepts = set()
        
        # Method 1: Named Entity Recognition
        if hasattr(self, 'ner_pipeline') and self.ner_pipeline:
            try:
                entities = self.ner_pipeline(text[:1000])  # Limit text length
                for entity in entities:
                    if entity['score'] > 0.8:  # High confidence only
                        concepts.add(entity['word'])
            except Exception:
                pass
        
        # Method 2: spaCy NER and noun chunks
        if self.nlp:
            try:
                doc = self.nlp(text[:1000])
                
                # Extract entities
                for ent in doc.ents:
                    if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'LAW']:
                        concepts.add(ent.text)
                
                # Extract noun chunks
                for chunk in doc.noun_chunks:
                    if 2 <= len(chunk.text.split()) <= 3:
                        concepts.add(chunk.text)
                        
            except Exception:
                pass
        
        # Method 3: Subject-specific concept extraction
        subject_terms = self.educational_terms.get(subject.lower(), [])
        text_lower = text.lower()
        
        for term in subject_terms:
            if term.lower() in text_lower:
                concepts.add(term)
        
        # Method 4: Pattern-based concept extraction
        concept_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns
            r'\b\w+(?:tion|sion|ment|ness|ity|ism|ogy|graphy)\b',  # Academic suffixes
            r'\b(?:principle|theory|law|rule|concept|method|process|system|model)\s+of\s+\w+',
            r'\b\w+(?:\s+\w+)?\s+(?:process|method|technique|approach|strategy)\b'
        ]
        
        for pattern in concept_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if 3 <= len(match) <= 30:  # Reasonable length
                    concepts.add(match)
        
        # Filter and clean concepts
        filtered_concepts = []
        for concept in concepts:
            # Remove very common words
            if concept.lower() not in {'this', 'that', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}:
                # Clean the concept
                cleaned = re.sub(r'[^\w\s]', '', concept).strip()
                if cleaned and len(cleaned.split()) <= 4:
                    filtered_concepts.append(cleaned)
        
        # Return top concepts
        concept_counts = Counter(filtered_concepts)
        return [concept for concept, count in concept_counts.most_common(15)]
    
    def assess_text_difficulty_enhanced(self, text):
        """Enhanced text difficulty assessment"""
        
        if not text:
            return {'level': 'unknown', 'scores': {}}
        
        assessment = {
            'level': 'intermediate',
            'scores': {},
            'metrics': {},
            'recommendations': []
        }
        
        try:
            # Basic metrics
            words = text.split()
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            assessment['metrics'] = {
                'word_count': len(words),
                'sentence_count': len(sentences),
                'avg_words_per_sentence': len(words) / len(sentences) if sentences else 0,
                'avg_word_length': np.mean([len(word) for word in words]) if words else 0
            }
            
            # Readability scores
            if TEXTSTAT_AVAILABLE:
                assessment['scores']['flesch_reading_ease'] = flesch_reading_ease(text)
                assessment['scores']['flesch_kincaid_grade'] = flesch_kincaid_grade(text)
            
            # Technical vocabulary analysis
            technical_words = sum(1 for word in words if len(word) > 8 or 
                                any(suffix in word.lower() for suffix in ['tion', 'sion', 'ism', 'ogy']))
            
            assessment['metrics']['technical_vocabulary_ratio'] = technical_words / len(words) if words else 0
            
            # Complexity indicators
            complex_sentences = sum(1 for s in sentences if len(s.split()) > 20)
            assessment['metrics']['complex_sentence_ratio'] = complex_sentences / len(sentences) if sentences else 0
            
            # Determine difficulty level
            assessment['level'] = self._determine_difficulty_level(assessment['metrics'], assessment['scores'])
            
            # Generate recommendations
            assessment['recommendations'] = self._generate_difficulty_recommendations(assessment)
            
        except Exception as e:
            st.warning(f"Difficulty assessment failed: {e}")
        
        return assessment
    
    def _determine_difficulty_level(self, metrics, scores):
        """Determine overall difficulty level from metrics"""
        
        score = 0
        
        # Sentence length scoring
        avg_sentence_length = metrics.get('avg_words_per_sentence', 0)
        if avg_sentence_length > 25:
            score += 2
        elif avg_sentence_length > 15:
            score += 1
        
        # Word length scoring
        avg_word_length = metrics.get('avg_word_length', 0)
        if avg_word_length > 6:
            score += 2
        elif avg_word_length > 4.5:
            score += 1
        
        # Technical vocabulary scoring
        tech_ratio = metrics.get('technical_vocabulary_ratio', 0)
        if tech_ratio > 0.2:
            score += 2
        elif tech_ratio > 0.1:
            score += 1
        
        # Complex sentence scoring
        complex_ratio = metrics.get('complex_sentence_ratio', 0)
        if complex_ratio > 0.3:
            score += 2
        elif complex_ratio > 0.15:
            score += 1
        
        # Readability scoring
        if scores.get('flesch_reading_ease'):
            flesch_score = scores['flesch_reading_ease']
            if flesch_score < 30:
                score += 3
            elif flesch_score < 50:
                score += 2
            elif flesch_score < 70:
                score += 1
        
        # Determine level
        if score >= 6:
            return 'advanced'
        elif score >= 3:
            return 'intermediate'
        else:
            return 'beginner'
    
    def _generate_difficulty_recommendations(self, assessment):
        """Generate recommendations based on difficulty assessment"""
        
        recommendations = []
        level = assessment['level']
        metrics = assessment['metrics']
        
        if level == 'advanced':
            recommendations.append("Consider providing glossaries for technical terms")
            recommendations.append("Break down complex concepts into smaller parts")
            
            if metrics.get('avg_words_per_sentence', 0) > 25:
                recommendations.append("Consider shorter sentences for better readability")
        
        elif level == 'beginner':
            recommendations.append("Content is well-suited for beginners")
            recommendations.append("Consider adding more examples and illustrations")
        
        else:  # intermediate
            recommendations.append("Good balance of complexity and clarity")
            recommendations.append("Consider adding more advanced examples for challenge")
        
        return recommendations
    
    def generate_quiz_questions_advanced(self, text, difficulty_level="intermediate", num_questions=5):
        """Generate advanced quiz questions using NLP analysis"""
        
        if not text:
            return []
        
        questions = []
        
        # Extract key concepts for question generation
        concepts = self.extract_educational_concepts_advanced(text)
        
        # Use spaCy for sentence analysis if available
        if self.nlp:
            questions.extend(self._generate_questions_with_spacy(text, concepts, difficulty_level))
        
        # Generate questions using concept analysis
        questions.extend(self._generate_concept_based_questions(concepts, difficulty_level))
        
        # Generate questions using text patterns
        questions.extend(self._generate_pattern_based_questions(text, difficulty_level))
        
        # Filter and rank questions
        ranked_questions = self._rank_and_filter_questions(questions, num_questions)
        
        return ranked_questions[:num_questions]
    
    def _generate_questions_with_spacy(self, text, concepts, difficulty_level):
        """Generate questions using spaCy analysis"""
        
        questions = []
        
        try:
            doc = self.nlp(text[:1000])  # Limit text length
            
            # Extract important sentences
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.split()) > 5]
            
            # Generate questions from sentences containing concepts
            for sentence in sentences[:5]:  # Limit to first 5 sentences
                for concept in concepts[:3]:  # Top 3 concepts
                    if concept.lower() in sentence.lower():
                        question = self._create_question_from_sentence(sentence, concept, difficulty_level)
                        if question:
                            questions.append(question)
        
        except Exception as e:
            st.warning(f"spaCy question generation failed: {e}")
        
        return questions
    
    def _generate_concept_based_questions(self, concepts, difficulty_level):
        """Generate questions based on extracted concepts"""
        
        questions = []
        
        for concept in concepts[:8]:  # Use top 8 concepts
            if difficulty_level == 'beginner':
                templates = [
                    f"What is {concept}?",
                    f"Define {concept}.",
                    f"List the main characteristics of {concept}."
                ]
            elif difficulty_level == 'intermediate':
                templates = [
                    f"Explain how {concept} works.",
                    f"What are the main functions of {concept}?",
                    f"Compare {concept} with related concepts.",
                    f"What role does {concept} play in this system?"
                ]
            else:  # advanced
                templates = [
                    f"Analyze the significance of {concept} in this context.",
                    f"Evaluate the implications of {concept}.",
                    f"How would you modify or improve {concept}?",
                    f"Critically assess the role of {concept}."
                ]
            
            for template in templates[:2]:  # Limit templates per concept
                questions.append({
                    'question': template,
                    'type': 'short_answer',
                    'concept': concept,
                    'difficulty': difficulty_level,
                    'bloom_level': self._determine_bloom_level(template),
                    'score': 0.7  # Base score for concept questions
                })
        
        return questions
    
    def _generate_pattern_based_questions(self, text, difficulty_level):
        """Generate questions using text patterns"""
        
        questions = []
        
        # Pattern for definitions
        definition_patterns = [
            r'(\w+(?:\s+\w+)*)\s+is\s+(.*?)(?:\.|$)',
            r'(\w+(?:\s+\w+)*)\s+refers to\s+(.*?)(?:\.|$)',
            r'(\w+(?:\s+\w+)*)\s+means\s+(.*?)(?:\.|$)'
        ]
        
        for pattern in definition_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                term = match.group(1).strip()
                definition = match.group(2).strip()
                
                if len(term.split()) <= 3 and len(definition.split()) >= 3:
                    if difficulty_level == 'beginner':
                        question_text = f"What is {term}?"
                    else:
                        question_text = f"Define {term} and explain its significance."
                    
                    questions.append({
                        'question': question_text,
                        'type': 'short_answer',
                        'concept': term,
                        'difficulty': difficulty_level,
                        'bloom_level': 'understand',
                        'expected_answer': definition,
                        'score': 0.8  # Higher score for pattern-based
                    })
        
        return questions
    
    def _create_question_from_sentence(self, sentence, concept, difficulty_level):
        """Create a question from a sentence containing a concept"""
        
        try:
            if difficulty_level == 'beginner':
                return {
                    'question': f"According to the text, what is mentioned about {concept}?",
                    'type': 'short_answer',
                    'concept': concept,
                    'difficulty': difficulty_level,
                    'bloom_level': 'remember',
                    'context': sentence,
                    'score': 0.6
                }
            
            elif difficulty_level == 'intermediate':
                return {
                    'question': f"Based on the information provided, explain the role of {concept}.",
                    'type': 'short_answer', 
                    'concept': concept,
                    'difficulty': difficulty_level,
                    'bloom_level': 'understand',
                    'context': sentence,
                    'score': 0.7
                }
            
            else:  # advanced
                return {
                    'question': f"Analyze the relationship between {concept} and other elements mentioned in this context.",
                    'type': 'short_answer',
                    'concept': concept,
                    'difficulty': difficulty_level,
                    'bloom_level': 'analyze',
                    'context': sentence,
                    'score': 0.8
                }
        
        except Exception:
            return None
    
    def _determine_bloom_level(self, question_text):
        """Determine Bloom's taxonomy level from question text"""
        
        question_lower = question_text.lower()
        
        if any(verb in question_lower for verb in ['create', 'design', 'develop', 'formulate']):
            return 'create'
        elif any(verb in question_lower for verb in ['evaluate', 'assess', 'critique', 'judge']):
            return 'evaluate'  
        elif any(verb in question_lower for verb in ['analyze', 'examine', 'compare', 'contrast']):
            return 'analyze'
        elif any(verb in question_lower for verb in ['apply', 'demonstrate', 'solve', 'use']):
            return 'apply'
        elif any(verb in question_lower for verb in ['explain', 'describe', 'discuss', 'interpret']):
            return 'understand'
        else:
            return 'remember'
    
    def _rank_and_filter_questions(self, questions, target_count):
        """Rank and filter questions for quality"""
        
        # Remove duplicates
        seen_questions = set()
        unique_questions = []
        
        for q in questions:
            q_text = q['question'].lower().strip()
            if q_text not in seen_questions:
                seen_questions.add(q_text)
                unique_questions.append(q)
        
        # Sort by score and diversity
        unique_questions.sort(key=lambda x: (x.get('score', 0.5), len(x['question'])), reverse=True)
        
        # Ensure diversity in Bloom's levels
        bloom_distribution = {}
        final_questions = []
        
        for question in unique_questions:
            bloom_level = question.get('bloom_level', 'remember')
            
            if len(final_questions) < target_count:
                if bloom_distribution.get(bloom_level, 0) < target_count // 3:
                    final_questions.append(question)
                    bloom_distribution[bloom_level] = bloom_distribution.get(bloom_level, 0) + 1
        
        # Fill remaining slots if needed
        for question in unique_questions:
            if len(final_questions) < target_count and question not in final_questions:
                final_questions.append(question)
        
        return final_questions
    
    def create_nlp_analysis_report(self, text, difficulty_level="intermediate"):
        """Create comprehensive NLP analysis report"""
        
        if not text:
            return "No text provided for analysis."
        
        report = []
        report.append("# 🧠 Advanced NLP Analysis Report\n")
        
        # Text statistics
        words = len(text.split())
        sentences = len(re.split(r'[.!?]+', text))
        
        report.append(f"**Text Length:** {words} words, {sentences} sentences")
        report.append("")
        
        # Difficulty assessment
        difficulty = self.assess_text_difficulty_enhanced(text)
        report.append(f"**Difficulty Level:** {difficulty['level'].title()}")
        
        if difficulty['scores']:
            report.append("**Readability Scores:**")
            for metric, score in difficulty['scores'].items():
                report.append(f"- {metric}: {score:.1f}")
        
        report.append("")
        
        # Topic extraction
        topics = self.extract_educational_topics_advanced([text], difficulty_level)
        if topics:
            report.append("**Extracted Topics:**")
            for topic in topics[:5]:
                report.append(f"- **{topic['name']}** (Category: {topic['educational_category']})")
                report.append(f"  Keywords: {', '.join(topic['keywords'][:5])}")
        
        report.append("")
        
        # Concept extraction
        concepts = self.extract_educational_concepts_advanced(text)
        if concepts:
            report.append("**Key Concepts:**")
            for concept in concepts[:10]:
                report.append(f"- {concept}")
        
        report.append("")
        
        # Generated questions
        questions = self.generate_quiz_questions_advanced(text, difficulty_level)
        if questions:
            report.append("**Generated Questions:**")
            for i, q in enumerate(questions, 1):
                report.append(f"{i}. {q['question']} *(Level: {q.get('bloom_level', 'N/A')})*")
        
        return "\n".join(report)

# Global NLP processor instance
nlp_processor = EnhancedNLPProcessor()
