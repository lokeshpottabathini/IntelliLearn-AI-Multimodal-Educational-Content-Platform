import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json

# Enhanced imports for advanced adaptive learning
try:
    from .advanced_learning_analytics import AdvancedLearningAnalytics
    from .cognitive_load_assessor import CognitiveLoadAssessor
    from .learning_pattern_analyzer import LearningPatternAnalyzer
    from .adaptive_content_generator import AdaptiveContentGenerator
    ENHANCED_MODULES_AVAILABLE = True
except ImportError:
    ENHANCED_MODULES_AVAILABLE = False

class AdaptiveLearningEngine:
    def __init__(self):
        """Initialize enhanced adaptive learning engine with comprehensive learning style analysis"""
        
        # Enhanced learning styles with detailed preferences
        self.learning_styles = {
            'Visual': {
                'videos': 0.8, 'images': 0.9, 'text': 0.4, 'audio': 0.3,
                'slides': 0.9, 'diagrams': 0.95, 'charts': 0.85, 'animations': 0.8
            },
            'Auditory': {
                'videos': 0.6, 'images': 0.3, 'text': 0.5, 'audio': 0.9,
                'voice_chat': 0.95, 'discussions': 0.8, 'music': 0.7, 'narration': 0.9
            },
            'Kinesthetic': {
                'videos': 0.7, 'images': 0.6, 'text': 0.6, 'audio': 0.4,
                'interactive_quizzes': 0.9, 'simulations': 0.95, 'exercises': 0.8, 'games': 0.85
            },
            'Reading/Writing': {
                'videos': 0.4, 'images': 0.3, 'text': 0.9, 'audio': 0.5,
                'notes': 0.95, 'summaries': 0.9, 'articles': 0.8, 'essays': 0.85
            }
        }
        
        # Enhanced difficulty levels with detailed attributes
        self.difficulty_levels = {
            'Beginner': {
                'complexity_score': 1,
                'prerequisite_ratio': 0.2,
                'cognitive_load': 'Low',
                'time_multiplier': 1.5,
                'practice_frequency': 'High'
            },
            'Intermediate': {
                'complexity_score': 2,
                'prerequisite_ratio': 0.5,
                'cognitive_load': 'Medium',
                'time_multiplier': 1.0,
                'practice_frequency': 'Medium'
            },
            'Advanced': {
                'complexity_score': 3,
                'prerequisite_ratio': 0.8,
                'cognitive_load': 'High',
                'time_multiplier': 0.8,
                'practice_frequency': 'Low'
            }
        }
        
        # Initialize enhanced modules
        self.advanced_analytics = None
        self.cognitive_assessor = None
        self.pattern_analyzer = None
        self.content_generator = None
        
        if ENHANCED_MODULES_AVAILABLE:
            try:
                self.advanced_analytics = AdvancedLearningAnalytics()
                self.cognitive_assessor = CognitiveLoadAssessor()
                self.pattern_analyzer = LearningPatternAnalyzer()
                self.content_generator = AdaptiveContentGenerator()
                st.success("✅ Enhanced adaptive learning features loaded")
            except Exception as e:
                st.warning(f"Enhanced features not available: {e}")
        
        # Initialize learning pattern tracking
        self.learning_patterns = {
            'time_preferences': {},
            'session_durations': [],
            'break_patterns': [],
            'performance_cycles': [],
            'content_preferences': {},
            'difficulty_progression': []
        }

    def detect_learning_style_enhanced(self, user_interactions, user_progress=None, difficulty_level='intermediate'):
        """NEW: Enhanced learning style detection with comprehensive analysis"""
        
        if not user_interactions:
            return self._get_default_learning_style(difficulty_level)
        
        # Use advanced analytics if available
        if self.advanced_analytics:
            try:
                style_analysis = self.advanced_analytics.analyze_comprehensive_learning_style(
                    user_interactions, user_progress, difficulty_level
                )
                return style_analysis
            except Exception as e:
                st.warning(f"Advanced style detection failed: {e}")
        
        # Enhanced style analysis
        style_scores = {
            'Visual': self._calculate_visual_preference_enhanced(user_interactions),
            'Auditory': self._calculate_auditory_preference_enhanced(user_interactions),
            'Kinesthetic': self._calculate_kinesthetic_preference_enhanced(user_interactions),
            'Reading/Writing': self._calculate_text_preference_enhanced(user_interactions)
        }
        
        # Add confidence scoring and mixed styles
        primary_style = max(style_scores, key=style_scores.get)
        confidence = style_scores[primary_style]
        
        # Detect mixed learning styles (if scores are close)
        sorted_styles = sorted(style_scores.items(), key=lambda x: x[1], reverse=True)
        
        result = {
            'primary_style': primary_style,
            'confidence': confidence,
            'style_scores': style_scores,
            'mixed_styles': [],
            'recommendations': []
        }
        
        # Check for mixed learning styles
        if len(sorted_styles) > 1 and sorted_styles[1][1] > sorted_styles[0][1] * 0.8:
            result['mixed_styles'].append(sorted_styles[1][0])
            result['recommendations'].append(f"Consider combining {primary_style} and {sorted_styles[1][0]} approaches")
        
        # Add difficulty-specific recommendations
        if difficulty_level == 'beginner':
            result['recommendations'].append("Focus on visual aids and step-by-step explanations")
        elif difficulty_level == 'advanced':
            result['recommendations'].append("Incorporate multiple learning modalities for complex concepts")
        
        return result

    def detect_learning_style(self, user_interactions):
        """Detect user's learning style based on interaction patterns (original method enhanced)"""
        if not user_interactions:
            return 'Visual'  # Default
        
        # Analyze interaction patterns
        style_scores = {
            'Visual': self._calculate_visual_preference(user_interactions),
            'Auditory': self._calculate_auditory_preference(user_interactions),
            'Kinesthetic': self._calculate_kinesthetic_preference(user_interactions),
            'Reading/Writing': self._calculate_text_preference(user_interactions)
        }
        
        return max(style_scores, key=style_scores.get)

    def _calculate_visual_preference_enhanced(self, interactions):
        """NEW: Enhanced visual preference calculation with weighted scoring"""
        visual_score = 0
        total_weight = 0
        
        for interaction in interactions:
            action_type = interaction.get('action_type', '')
            duration = interaction.get('duration', 1)
            
            # Weight by engagement time and completion
            weight = duration * (1 + interaction.get('completion_rate', 0))
            total_weight += weight
            
            if action_type in ['video_watched', 'image_viewed', 'slides_generated']:
                visual_score += weight * 1.0
            elif action_type == 'video_completed':
                visual_score += weight * 2.0  # Higher weight for completion
            elif action_type in ['diagram_studied', 'chart_analyzed']:
                visual_score += weight * 1.5
            elif action_type == 'visual_quiz_taken':
                visual_score += weight * 1.2
        
        return visual_score / max(total_weight, 1)

    def _calculate_auditory_preference_enhanced(self, interactions):
        """NEW: Enhanced auditory preference calculation"""
        auditory_score = 0
        total_weight = 0
        
        for interaction in interactions:
            action_type = interaction.get('action_type', '')
            duration = interaction.get('duration', 1)
            weight = duration * (1 + interaction.get('completion_rate', 0))
            total_weight += weight
            
            if action_type in ['audio_played', 'voice_input_used', 'tts_enabled']:
                auditory_score += weight * 1.0
            elif action_type == 'voice_question_asked':
                auditory_score += weight * 2.0
            elif action_type in ['voice_chat_session', 'audio_explanation_listened']:
                auditory_score += weight * 1.5
            elif action_type == 'podcast_listened':
                auditory_score += weight * 1.3
        
        return auditory_score / max(total_weight, 1)

    def _calculate_kinesthetic_preference_enhanced(self, interactions):
        """NEW: Enhanced kinesthetic preference calculation"""
        kinesthetic_score = 0
        total_weight = 0
        
        for interaction in interactions:
            action_type = interaction.get('action_type', '')
            duration = interaction.get('duration', 1)
            weight = duration * (1 + interaction.get('completion_rate', 0))
            total_weight += weight
            
            if action_type in ['quiz_taken', 'interactive_element_clicked', 'practice_completed']:
                kinesthetic_score += weight * 1.0
            elif action_type in ['simulation_completed', 'game_played']:
                kinesthetic_score += weight * 2.0
            elif action_type in ['drag_drop_exercise', 'interactive_quiz']:
                kinesthetic_score += weight * 1.5
            elif action_type == 'hands_on_exercise':
                kinesthetic_score += weight * 1.8
        
        return kinesthetic_score / max(total_weight, 1)

    def _calculate_text_preference_enhanced(self, interactions):
        """NEW: Enhanced text preference calculation"""
        text_score = 0
        total_weight = 0
        
        for interaction in interactions:
            action_type = interaction.get('action_type', '')
            duration = interaction.get('duration', 1)
            weight = duration * (1 + interaction.get('completion_rate', 0))
            total_weight += weight
            
            if action_type in ['text_read', 'chat_message_sent', 'notes_taken']:
                text_score += weight * 1.0
            elif action_type == 'long_text_read':
                text_score += weight * 2.0
            elif action_type in ['summary_created', 'essay_written']:
                text_score += weight * 1.8
            elif action_type in ['article_read', 'documentation_studied']:
                text_score += weight * 1.3
        
        return text_score / max(total_weight, 1)

    def generate_adaptive_path_enhanced(self, current_performance, subject, learning_style_data, knowledge_base, user_progress=None):
        """NEW: Enhanced adaptive path generation with comprehensive personalization"""
        
        # Use advanced content generator if available
        if self.content_generator:
            try:
                return self.content_generator.generate_comprehensive_learning_path(
                    current_performance, subject, learning_style_data, knowledge_base, user_progress
                )
            except Exception as e:
                st.warning(f"Advanced path generation failed: {e}")
        
        # Enhanced path generation
        chapters = knowledge_base.get('chapters', {})
        
        # Determine user's current difficulty level
        difficulty_level = self._determine_optimal_difficulty(current_performance, user_progress)
        
        # Get learning style preferences
        if isinstance(learning_style_data, dict):
            primary_style = learning_style_data.get('primary_style', 'Visual')
            mixed_styles = learning_style_data.get('mixed_styles', [])
        else:
            primary_style = learning_style_data
            mixed_styles = []
        
        # Sort chapters by difficulty and learning efficiency
        sorted_chapters = self._sort_chapters_by_learning_efficiency(chapters, primary_style, difficulty_level)
        
        # Generate personalized learning path
        learning_path = []
        
        if current_performance < 50:
            # Intensive remedial path
            learning_path = self._create_intensive_remedial_path(sorted_chapters, primary_style, mixed_styles)
        elif current_performance < 70:
            # Standard remedial path
            learning_path = self._create_remedial_path_enhanced(sorted_chapters, primary_style, mixed_styles)
        elif current_performance > 90:
            # Accelerated advanced path
            learning_path = self._create_accelerated_advanced_path(sorted_chapters, primary_style, mixed_styles)
        elif current_performance > 80:
            # Standard advanced path
            learning_path = self._create_advanced_path_enhanced(sorted_chapters, primary_style, mixed_styles)
        else:
            # Balanced standard path
            learning_path = self._create_standard_path_enhanced(sorted_chapters, primary_style, mixed_styles)
        
        # Add personalization based on user progress
        if user_progress:
            learning_path = self._personalize_path_with_progress(learning_path, user_progress)
        
        return {
            'path': learning_path,
            'difficulty_level': difficulty_level,
            'primary_style': primary_style,
            'mixed_styles': mixed_styles,
            'estimated_completion_time': self._calculate_total_path_time(learning_path),
            'adaptive_features': {
                'difficulty_adjustment': True,
                'style_optimization': True,
                'progress_tracking': True,
                'cognitive_load_balancing': True
            }
        }

    def generate_adaptive_path(self, current_performance, subject, learning_style, knowledge_base):
        """Generate personalized learning path (original method enhanced)"""
        chapters = knowledge_base.get('chapters', {})
        
        # Sort chapters by difficulty and prerequisites
        sorted_chapters = self._sort_chapters_by_difficulty(chapters)
        
        learning_path = []
        
        if current_performance < 60:
            # Remedial path - focus on fundamentals
            learning_path = self._create_remedial_path(sorted_chapters, learning_style)
        elif current_performance > 85:
            # Advanced path - skip basics, focus on complex topics
            learning_path = self._create_advanced_path(sorted_chapters, learning_style)
        else:
            # Standard path - balanced progression
            learning_path = self._create_standard_path(sorted_chapters, learning_style)
        
        return learning_path

    def _determine_optimal_difficulty(self, current_performance, user_progress):
        """NEW: Determine optimal difficulty level based on performance and progress"""
        
        if not user_progress:
            # Base on performance only
            if current_performance >= 85:
                return 'Advanced'
            elif current_performance >= 65:
                return 'Intermediate'
            else:
                return 'Beginner'
        
        # Consider multiple factors
        factors = {
            'performance': current_performance,
            'streak': user_progress.get('learning_streak', 0),
            'badges': len(user_progress.get('badges_earned', [])),
            'quiz_average': self._calculate_average_quiz_score(user_progress),
            'consistency': self._calculate_learning_consistency(user_progress)
        }
        
        # Calculate composite difficulty score
        difficulty_score = (
            factors['performance'] * 0.4 +
            min(factors['streak'] * 2, 20) * 0.2 +
            min(factors['badges'] * 3, 30) * 0.15 +
            factors['quiz_average'] * 0.15 +
            factors['consistency'] * 0.1
        )
        
        if difficulty_score >= 80:
            return 'Advanced'
        elif difficulty_score >= 60:
            return 'Intermediate'
        else:
            return 'Beginner'

    def _sort_chapters_by_learning_efficiency(self, chapters, learning_style, difficulty_level):
        """NEW: Sort chapters by learning efficiency for the user"""
        
        chapter_scores = []
        
        for chapter_name, chapter_data in chapters.items():
            # Calculate efficiency score
            difficulty_match = self._calculate_difficulty_match(chapter_data, difficulty_level)
            style_match = self._calculate_style_match(chapter_data, learning_style)
            prerequisite_score = self._calculate_prerequisite_readiness(chapter_data)
            
            efficiency_score = (difficulty_match * 0.4 + style_match * 0.35 + prerequisite_score * 0.25)
            
            chapter_scores.append({
                'name': chapter_name,
                'data': chapter_data,
                'efficiency_score': efficiency_score,
                'difficulty_match': difficulty_match,
                'style_match': style_match
            })
        
        # Sort by efficiency score
        return sorted(chapter_scores, key=lambda x: x['efficiency_score'], reverse=True)

    def _create_intensive_remedial_path(self, chapters, primary_style, mixed_styles):
        """NEW: Create intensive remedial path for very low performers"""
        path = []
        
        # Focus on absolute fundamentals with maximum support
        for chapter in chapters:
            if chapter['data'].get('difficulty', 'Intermediate') == 'Beginner':
                content_types = self._get_multi_modal_content_types(primary_style, mixed_styles)
                
                path.append({
                    'chapter': chapter['name'],
                    'recommended_time': chapter['data'].get('estimated_reading_time', 30) * 2.0,
                    'content_types': content_types,
                    'practice_level': 'Guided',
                    'review_frequency': 'Very High',
                    'cognitive_load': 'Very Low',
                    'support_level': 'Maximum',
                    'break_frequency': 'Every 15 minutes',
                    'reinforcement': {
                        'immediate_feedback': True,
                        'multiple_examples': True,
                        'step_by_step_guidance': True,
                        'peer_support': True
                    }
                })
        
        return path

    def _create_remedial_path_enhanced(self, chapters, primary_style, mixed_styles):
        """NEW: Enhanced remedial path with personalized support"""
        path = []
        
        # Focus on beginner-level content with extra support
        for chapter in chapters:
            if chapter['data'].get('difficulty', 'Intermediate') in ['Beginner', 'Intermediate']:
                path.append({
                    'chapter': chapter['name'],
                    'recommended_time': chapter['data'].get('estimated_reading_time', 30) * 1.5,
                    'content_types': self._get_preferred_content_types(primary_style, mixed_styles),
                    'practice_level': 'Basic',
                    'review_frequency': 'High',
                    'cognitive_load': 'Low',
                    'support_level': 'High',
                    'adaptive_features': {
                        'difficulty_adjustment': True,
                        'extra_examples': True,
                        'frequent_checkpoints': True
                    }
                })
        
        return path

    def _create_accelerated_advanced_path(self, chapters, primary_style, mixed_styles):
        """NEW: Create accelerated path for very high performers"""
        path = []
        
        # Skip basics, focus on advanced topics with enrichment
        for chapter in chapters:
            if chapter['data'].get('difficulty', 'Intermediate') in ['Intermediate', 'Advanced']:
                path.append({
                    'chapter': chapter['name'],
                    'recommended_time': chapter['data'].get('estimated_reading_time', 30) * 0.6,
                    'content_types': self._get_advanced_content_types(primary_style, mixed_styles),
                    'practice_level': 'Expert',
                    'review_frequency': 'Very Low',
                    'cognitive_load': 'Very High',
                    'support_level': 'Minimal',
                    'enrichment': {
                        'research_projects': True,
                        'peer_teaching': True,
                        'creative_applications': True,
                        'interdisciplinary_connections': True
                    }
                })
        
        return path

    def _create_advanced_path_enhanced(self, chapters, primary_style, mixed_styles):
        """NEW: Enhanced advanced path with challenging content"""
        path = []
        
        # Skip basics, focus on advanced content
        for chapter in chapters:
            if chapter['data'].get('difficulty', 'Intermediate') in ['Intermediate', 'Advanced']:
                path.append({
                    'chapter': chapter['name'],
                    'recommended_time': chapter['data'].get('estimated_reading_time', 30) * 0.8,
                    'content_types': self._get_preferred_content_types(primary_style, mixed_styles),
                    'practice_level': 'Advanced',
                    'review_frequency': 'Low',
                    'cognitive_load': 'High',
                    'support_level': 'Low',
                    'challenges': {
                        'complex_problems': True,
                        'synthesis_tasks': True,
                        'critical_analysis': True
                    }
                })
        
        return path

    def _create_standard_path_enhanced(self, chapters, primary_style, mixed_styles):
        """NEW: Enhanced standard path with balanced progression"""
        path = []
        
        # Balanced progression through all difficulty levels
        for chapter in chapters:
            difficulty = chapter['data'].get('difficulty', 'Intermediate')
            time_multiplier = {'Beginner': 1.2, 'Intermediate': 1.0, 'Advanced': 1.1}
            
            path.append({
                'chapter': chapter['name'],
                'recommended_time': chapter['data'].get('estimated_reading_time', 30) * time_multiplier.get(difficulty, 1.0),
                'content_types': self._get_preferred_content_types(primary_style, mixed_styles),
                'practice_level': 'Standard',
                'review_frequency': 'Medium',
                'cognitive_load': 'Medium',
                'support_level': 'Medium',
                'progression': {
                    'prerequisite_check': True,
                    'gradual_difficulty_increase': True,
                    'regular_assessments': True
                }
            })
        
        return path

    def _get_multi_modal_content_types(self, primary_style, mixed_styles):
        """NEW: Get multi-modal content types for intensive support"""
        content_types = self._get_preferred_content_types(primary_style, mixed_styles)
        
        # Add extra modalities for intensive support
        additional_types = {
            'Visual': ['step_by_step_diagrams', 'progress_visualizations'],
            'Auditory': ['guided_audio_tours', 'explanation_podcasts'],
            'Kinesthetic': ['micro_interactions', 'progress_games'],
            'Reading/Writing': ['structured_notes', 'guided_summaries']
        }
        
        if primary_style in additional_types:
            content_types.extend(additional_types[primary_style])
        
        return content_types

    def _get_advanced_content_types(self, primary_style, mixed_styles):
        """NEW: Get advanced content types for high performers"""
        base_types = self._get_preferred_content_types(primary_style, mixed_styles)
        
        # Add challenging content types
        advanced_types = {
            'Visual': ['complex_visualizations', 'data_analysis_charts'],
            'Auditory': ['expert_interviews', 'debate_discussions'],
            'Kinesthetic': ['advanced_simulations', 'research_projects'],
            'Reading/Writing': ['academic_papers', 'critical_essays']
        }
        
        if primary_style in advanced_types:
            base_types.extend(advanced_types[primary_style])
        
        return base_types

    def _personalize_path_with_progress(self, learning_path, user_progress):
        """NEW: Personalize path based on user's learning history"""
        
        # Get user's learning patterns
        preferred_times = self._extract_preferred_learning_times(user_progress)
        average_session_length = self._calculate_average_session_length(user_progress)
        weak_areas = self._identify_weak_areas(user_progress)
        
        # Adjust path based on patterns
        for item in learning_path:
            # Adjust timing based on user preferences
            if preferred_times.get('morning', 0) > 0.6:
                item['recommended_schedule'] = 'Morning sessions'
            elif preferred_times.get('evening', 0) > 0.6:
                item['recommended_schedule'] = 'Evening sessions'
            
            # Adjust session length
            if average_session_length < 20:
                item['session_breakdown'] = 'Short 15-minute segments'
            elif average_session_length > 45:
                item['session_breakdown'] = 'Extended 60-minute sessions'
            
            # Add extra support for weak areas
            chapter_topics = item['chapter'].lower()
            for weak_area in weak_areas:
                if weak_area.lower() in chapter_topics:
                    item['extra_support'] = True
                    item['recommended_time'] *= 1.3
                    item['review_frequency'] = 'High'
        
        return learning_path

    def adapt_content_difficulty_realtime(self, current_session_data, user_performance_history):
        """NEW: Real-time difficulty adaptation during learning sessions"""
        
        # Analyze current session performance
        current_accuracy = current_session_data.get('accuracy', 0)
        response_time = current_session_data.get('avg_response_time', 0)
        engagement_level = current_session_data.get('engagement_score', 0.5)
        
        # Calculate optimal difficulty adjustment
        if current_accuracy > 90 and response_time < 10 and engagement_level > 0.8:
            adjustment = 'increase_difficulty'
            recommendation = 'User is performing excellently - increase challenge'
        elif current_accuracy < 60 or engagement_level < 0.3:
            adjustment = 'decrease_difficulty'
            recommendation = 'User struggling - provide more support'
        elif current_accuracy < 70 and response_time > 30:
            adjustment = 'provide_hints'
            recommendation = 'User needs guidance - add hints and examples'
        else:
            adjustment = 'maintain'
            recommendation = 'Current difficulty level is appropriate'
        
        return {
            'adjustment': adjustment,
            'confidence': self._calculate_adjustment_confidence(current_session_data, user_performance_history),
            'recommendation': recommendation,
            'suggested_actions': self._get_adjustment_actions(adjustment),
            'next_content_type': self._suggest_next_content_type(current_session_data)
        }

    def predict_learning_outcomes(self, user_progress, learning_path, time_horizon_days=30):
        """NEW: Predict learning outcomes based on current trajectory"""
        
        if self.pattern_analyzer:
            try:
                return self.pattern_analyzer.predict_comprehensive_outcomes(
                    user_progress, learning_path, time_horizon_days
                )
            except Exception as e:
                st.warning(f"Advanced prediction failed: {e}")
        
        # Basic prediction algorithm
        current_performance = self._calculate_current_performance_trend(user_progress)
        learning_velocity = self._calculate_learning_velocity(user_progress)
        consistency_factor = self._calculate_learning_consistency(user_progress)
        
        # Predict completion timeline
        estimated_completion = self._predict_completion_timeline(learning_path, learning_velocity)
        
        # Predict performance improvement
        performance_projection = current_performance + (learning_velocity * time_horizon_days * consistency_factor)
        performance_projection = min(100, max(0, performance_projection))
        
        # Identify potential challenges
        challenges = self._identify_potential_challenges(user_progress, learning_path)
        
        return {
            'completion_prediction': {
                'estimated_days': estimated_completion,
                'confidence': min(0.95, consistency_factor),
                'completion_probability': min(0.9, consistency_factor * 0.8 + learning_velocity * 0.2)
            },
            'performance_prediction': {
                'projected_score': performance_projection,
                'improvement_rate': learning_velocity,
                'confidence_interval': [
                    max(0, performance_projection - 15),
                    min(100, performance_projection + 15)
                ]
            },
            'risk_factors': challenges,
            'recommendations': self._generate_outcome_recommendations(current_performance, learning_velocity, challenges)
        }

    def generate_personalized_study_schedule(self, learning_path, user_preferences, available_time_per_day=60):
        """NEW: Generate personalized study schedule based on learning path and preferences"""
        
        # Extract user preferences
        preferred_times = user_preferences.get('preferred_study_times', ['morning'])
        session_length_preference = user_preferences.get('preferred_session_length', 30)
        break_preferences = user_preferences.get('break_frequency', 'medium')
        
        # Calculate total content time
        total_time_needed = sum(item.get('recommended_time', 30) for item in learning_path.get('path', []))
        
        # Create schedule
        schedule = {
            'total_duration_days': max(7, total_time_needed // available_time_per_day),
            'daily_sessions': [],
            'weekly_structure': {},
            'milestone_dates': [],
            'adaptive_features': {
                'automatic_rescheduling': True,
                'difficulty_based_timing': True,
                'performance_adjustments': True
            }
        }
        
        # Generate daily sessions
        current_day = 0
        remaining_content = learning_path.get('path', []).copy()
        
        while remaining_content and current_day < 365:  # Limit to 1 year
            daily_content = []
            daily_time_used = 0
            
            while remaining_content and daily_time_used < available_time_per_day:
                next_item = remaining_content[0]
                item_time = next_item.get('recommended_time', 30)
                
                if daily_time_used + item_time <= available_time_per_day:
                    daily_content.append(remaining_content.pop(0))
                    daily_time_used += item_time
                else:
                    # Split the item if it's too long
                    if item_time > available_time_per_day:
                        split_item = next_item.copy()
                        split_item['recommended_time'] = available_time_per_day - daily_time_used
                        daily_content.append(split_item)
                        
                        # Update remaining item
                        remaining_content[0]['recommended_time'] = item_time - (available_time_per_day - daily_time_used)
                        daily_time_used = available_time_per_day
                    else:
                        break
            
            if daily_content:
                schedule['daily_sessions'].append({
                    'day': current_day + 1,
                    'date': (datetime.now() + timedelta(days=current_day)).strftime('%Y-%m-%d'),
                    'content': daily_content,
                    'total_time': daily_time_used,
                    'recommended_times': preferred_times,
                    'session_structure': self._create_session_structure(daily_content, session_length_preference)
                })
            
            current_day += 1
        
        return schedule

    # Original methods (enhanced)
    def _calculate_visual_preference(self, interactions):
        """Calculate preference for visual content (original method)"""
        visual_actions = 0
        total_actions = len(interactions)
        
        for interaction in interactions:
            if interaction.get('action_type') in ['video_watched', 'image_viewed', 'slides_generated']:
                visual_actions += 1
            elif interaction.get('action_type') == 'video_completed':
                visual_actions += 2  # Higher weight for completion
        
        return visual_actions / max(total_actions, 1)

    def _calculate_auditory_preference(self, interactions):
        """Calculate preference for auditory content (original method)"""
        auditory_actions = 0
        total_actions = len(interactions)
        
        for interaction in interactions:
            if interaction.get('action_type') in ['audio_played', 'voice_input_used', 'tts_enabled']:
                auditory_actions += 1
            elif interaction.get('action_type') == 'voice_question_asked':
                auditory_actions += 2
        
        return auditory_actions / max(total_actions, 1)

    def _calculate_text_preference(self, interactions):
        """Calculate preference for text-based content (original method)"""
        text_actions = 0
        total_actions = len(interactions)
        
        for interaction in interactions:
            if interaction.get('action_type') in ['text_read', 'chat_message_sent', 'notes_taken']:
                text_actions += 1
            elif interaction.get('action_type') == 'long_text_read':
                text_actions += 2
        
        return text_actions / max(total_actions, 1)

    def _calculate_kinesthetic_preference(self, interactions):
        """Calculate preference for interactive content (original method)"""
        kinesthetic_actions = 0
        total_actions = len(interactions)
        
        for interaction in interactions:
            if interaction.get('action_type') in ['quiz_taken', 'interactive_element_clicked', 'practice_completed']:
                kinesthetic_actions += 1
        
        return kinesthetic_actions / max(total_actions, 1)

    def _sort_chapters_by_difficulty(self, chapters):
        """Sort chapters by difficulty level"""
        difficulty_order = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
        
        chapter_list = [(name, data) for name, data in chapters.items()]
        
        return sorted(chapter_list, key=lambda x: difficulty_order.get(x[1].get('difficulty', 'Intermediate'), 2))

    def _create_remedial_path(self, chapters, learning_style):
        """Create remedial learning path for struggling students (original method)"""
        path = []
        
        # Focus on beginner-level content first
        for chapter_name, chapter_data in chapters:
            if chapter_data.get('difficulty', 'Intermediate') == 'Beginner':
                path.append({
                    'chapter': chapter_name,
                    'recommended_time': chapter_data.get('estimated_reading_time', 30) * 1.5,  # More time
                    'content_type': self._get_preferred_content_type(learning_style),
                    'practice_level': 'Basic',
                    'review_frequency': 'High'
                })
        
        return path

    def _create_advanced_path(self, chapters, learning_style):
        """Create advanced learning path for high-performing students (original method)"""
        path = []
        
        # Skip basics, focus on advanced content
        for chapter_name, chapter_data in chapters:
            if chapter_data.get('difficulty', 'Intermediate') in ['Intermediate', 'Advanced']:
                path.append({
                    'chapter': chapter_name,
                    'recommended_time': chapter_data.get('estimated_reading_time', 30) * 0.8,  # Less time
                    'content_type': self._get_preferred_content_type(learning_style),
                    'practice_level': 'Advanced',
                    'review_frequency': 'Low'
                })
        
        return path

    def _create_standard_path(self, chapters, learning_style):
        """Create standard learning path"""
        path = []
        
        for chapter_name, chapter_data in chapters:
            path.append({
                'chapter': chapter_name,
                'recommended_time': chapter_data.get('estimated_reading_time', 30),
                'content_type': self._get_preferred_content_type(learning_style),
                'practice_level': 'Standard',
                'review_frequency': 'Medium'
            })
        
        return path

    def _get_preferred_content_type(self, learning_style):
        """Get preferred content type based on learning style (original method)"""
        preferences = {
            'Visual': ['videos', 'slides', 'diagrams'],
            'Auditory': ['audio', 'discussions', 'explanations'],
            'Kinesthetic': ['interactive_quizzes', 'hands_on_exercises'],
            'Reading/Writing': ['text_summaries', 'note_taking', 'written_exercises']
        }
        
        return preferences.get(learning_style, ['videos', 'text_summaries'])

    def _get_preferred_content_types(self, primary_style, mixed_styles):
        """NEW: Get preferred content types considering mixed learning styles"""
        base_preferences = {
            'Visual': ['videos', 'slides', 'diagrams', 'infographics', 'charts'],
            'Auditory': ['audio', 'discussions', 'explanations', 'podcasts', 'voice_narration'],
            'Kinesthetic': ['interactive_quizzes', 'hands_on_exercises', 'simulations', 'games'],
            'Reading/Writing': ['text_summaries', 'note_taking', 'written_exercises', 'articles']
        }
        
        content_types = base_preferences.get(primary_style, ['videos', 'text_summaries'])
        
        # Add content types from mixed styles
        for mixed_style in mixed_styles:
            if mixed_style in base_preferences:
                content_types.extend(base_preferences[mixed_style][:2])  # Add top 2 from mixed style
        
        return list(set(content_types))  # Remove duplicates

    def _get_default_learning_style(self, difficulty_level):
        """NEW: Get default learning style based on difficulty level"""
        defaults = {
            'beginner': 'Visual',
            'intermediate': 'Visual',
            'advanced': 'Reading/Writing'
        }
        
        return defaults.get(difficulty_level, 'Visual')

    # Helper methods for enhanced functionality
    def _calculate_difficulty_match(self, chapter_data, difficulty_level):
        """Calculate how well chapter difficulty matches user level"""
        chapter_difficulty = chapter_data.get('difficulty', 'Intermediate')
        
        difficulty_scores = {
            ('Beginner', 'Beginner'): 1.0,
            ('Beginner', 'Intermediate'): 0.3,
            ('Beginner', 'Advanced'): 0.1,
            ('Intermediate', 'Beginner'): 0.7,
            ('Intermediate', 'Intermediate'): 1.0,
            ('Intermediate', 'Advanced'): 0.5,
            ('Advanced', 'Beginner'): 0.4,
            ('Advanced', 'Intermediate'): 0.8,
            ('Advanced', 'Advanced'): 1.0
        }
        
        return difficulty_scores.get((difficulty_level, chapter_difficulty), 0.5)

    def _calculate_style_match(self, chapter_data, learning_style):
        """Calculate how well chapter content matches learning style"""
        # This would analyze chapter content types and match against learning style preferences
        # For now, return a default value
        return 0.7

    def _calculate_prerequisite_readiness(self, chapter_data):
        """Calculate user's readiness for chapter prerequisites"""
        # This would check if user has completed prerequisite chapters
        # For now, return a default value
        return 0.8

    def _calculate_average_quiz_score(self, user_progress):
        """Calculate average quiz score from user progress"""
        all_scores = []
        
        # Collect scores from different difficulty levels
        for difficulty in ['beginner_scores', 'intermediate_scores', 'advanced_scores']:
            scores = user_progress.get(difficulty, [])
            all_scores.extend(scores)
        
        return np.mean(all_scores) if all_scores else 70

    def _calculate_learning_consistency(self, user_progress):
        """Calculate learning consistency score"""
        streak = user_progress.get('learning_streak', 0)
        total_sessions = user_progress.get('learning_sessions', 1)
        
        # Simple consistency calculation
        consistency = min(1.0, (streak / 7) * 0.5 + (total_sessions / 20) * 0.5)
        return consistency

    def _calculate_current_performance_trend(self, user_progress):
        """Calculate current performance trend"""
        recent_scores = []
        
        # Get recent quiz scores
        for difficulty in ['beginner_scores', 'intermediate_scores', 'advanced_scores']:
            scores = user_progress.get(difficulty, [])
            recent_scores.extend(scores[-5:])  # Last 5 scores
        
        return np.mean(recent_scores) if recent_scores else 70

    def _calculate_learning_velocity(self, user_progress):
        """Calculate learning velocity (improvement rate)"""
        all_scores = []
        
        for difficulty in ['beginner_scores', 'intermediate_scores', 'advanced_scores']:
            scores = user_progress.get(difficulty, [])
            all_scores.extend(scores)
        
        if len(all_scores) < 2:
            return 0.5  # Default velocity
        
        # Calculate improvement over time
        first_half = np.mean(all_scores[:len(all_scores)//2])
        second_half = np.mean(all_scores[len(all_scores)//2:])
        
        velocity = (second_half - first_half) / max(len(all_scores), 1)
        return max(-2, min(2, velocity))  # Limit velocity

    def _predict_completion_timeline(self, learning_path, learning_velocity):
        """Predict completion timeline for learning path"""
        total_time = sum(item.get('recommended_time', 30) for item in learning_path.get('path', []))
        
        # Adjust based on learning velocity
        velocity_multiplier = max(0.5, 1.0 - learning_velocity * 0.1)
        adjusted_time = total_time * velocity_multiplier
        
        # Convert to days (assuming 60 minutes per day)
        return max(1, adjusted_time // 60)

    def _identify_potential_challenges(self, user_progress, learning_path):
        """Identify potential learning challenges"""
        challenges = []
        
        # Check consistency
        streak = user_progress.get('learning_streak', 0)
        if streak < 3:
            challenges.append({
                'type': 'consistency',
                'severity': 'medium',
                'description': 'Inconsistent learning pattern detected'
            })
        
        # Check performance variation
        all_scores = []
        for difficulty in ['beginner_scores', 'intermediate_scores', 'advanced_scores']:
            all_scores.extend(user_progress.get(difficulty, []))
        
        if all_scores and np.std(all_scores) > 20:
            challenges.append({
                'type': 'performance_variation',
                'severity': 'low',
                'description': 'High variation in quiz performance'
            })
        
        return challenges

    def _generate_outcome_recommendations(self, current_performance, learning_velocity, challenges):
        """Generate recommendations based on predicted outcomes"""
        recommendations = []
        
        if current_performance < 70:
            recommendations.append("Focus on fundamental concepts before advancing")
        
        if learning_velocity < 0:
            recommendations.append("Consider reviewing study methods and taking breaks")
        
        for challenge in challenges:
            if challenge['type'] == 'consistency':
                recommendations.append("Set up a regular study schedule and stick to it")
            elif challenge['type'] == 'performance_variation':
                recommendations.append("Identify weak areas and provide extra practice")
        
        return recommendations

    def _extract_preferred_learning_times(self, user_progress):
        """Extract preferred learning times from user data"""
        # This would analyze when user typically studies
        # For now, return default preferences
        return {'morning': 0.6, 'afternoon': 0.3, 'evening': 0.1}

    def _calculate_average_session_length(self, user_progress):
        """Calculate average learning session length"""
        total_time = user_progress.get('total_study_time', 0)
        sessions = user_progress.get('learning_sessions', 1)
        
        return total_time / sessions if sessions > 0 else 30

    def _identify_weak_areas(self, user_progress):
        """Identify areas where user performs poorly"""
        weak_areas = []
        
        # Check difficulty-specific performance
        for difficulty in ['beginner', 'intermediate', 'advanced']:
            scores = user_progress.get(f'{difficulty}_scores', [])
            if scores and np.mean(scores) < 70:
                weak_areas.append(difficulty)
        
        return weak_areas

    def _calculate_adjustment_confidence(self, current_session_data, user_performance_history):
        """Calculate confidence in difficulty adjustment recommendation"""
        # Simple confidence calculation based on data quality
        data_points = len(user_performance_history)
        session_completeness = current_session_data.get('completeness', 0.5)
        
        confidence = min(0.95, (data_points / 10) * 0.5 + session_completeness * 0.5)
        return confidence

    def _get_adjustment_actions(self, adjustment):
        """Get specific actions for difficulty adjustment"""
        actions = {
            'increase_difficulty': [
                'Add more complex problems',
                'Introduce advanced concepts',
                'Reduce hints and guidance',
                'Increase pace of content delivery'
            ],
            'decrease_difficulty': [
                'Provide more examples',
                'Break content into smaller chunks',
                'Add visual aids and explanations',
                'Increase practice opportunities'
            ],
            'provide_hints': [
                'Add contextual hints',
                'Provide step-by-step guidance',
                'Show worked examples',
                'Offer multiple solution paths'
            ],
            'maintain': [
                'Continue current approach',
                'Monitor performance closely',
                'Be ready to adjust if needed'
            ]
        }
        
        return actions.get(adjustment, ['Monitor and adjust as needed'])

    def _suggest_next_content_type(self, current_session_data):
        """Suggest next content type based on current session"""
        current_type = current_session_data.get('content_type', 'text')
        performance = current_session_data.get('accuracy', 0)
        
        if performance > 85:
            return 'challenging_practice'
        elif performance < 60:
            return 'explanatory_content'
        else:
            return 'mixed_practice'

    def _create_session_structure(self, daily_content, session_length_preference):
        """Create structure for daily study sessions"""
        total_content_time = sum(item.get('recommended_time', 30) for item in daily_content)
        
        if total_content_time <= session_length_preference:
            return [{
                'session': 1,
                'duration': total_content_time,
                'content': daily_content,
                'break_after': False
            }]
        else:
            # Split into multiple sessions
            sessions = []
            current_session_content = []
            current_session_time = 0
            session_num = 1
            
            for item in daily_content:
                item_time = item.get('recommended_time', 30)
                
                if current_session_time + item_time <= session_length_preference:
                    current_session_content.append(item)
                    current_session_time += item_time
                else:
                    # Finish current session
                    if current_session_content:
                        sessions.append({
                            'session': session_num,
                            'duration': current_session_time,
                            'content': current_session_content,
                            'break_after': True
                        })
                        session_num += 1
                    
                    # Start new session
                    current_session_content = [item]
                    current_session_time = item_time
            
            # Add final session
            if current_session_content:
                sessions.append({
                    'session': session_num,
                    'duration': current_session_time,
                    'content': current_session_content,
                    'break_after': False
                })
            
            return sessions

    def _calculate_total_path_time(self, learning_path):
        """Calculate total estimated time for learning path"""
        return sum(item.get('recommended_time', 30) for item in learning_path)

    def get_adaptation_summary(self):
        """NEW: Get summary of adaptive learning features and current state"""
        
        return {
            'adaptive_features': {
                'learning_style_detection': True,
                'difficulty_adaptation': True,
                'personalized_paths': True,
                'real_time_adjustment': True,
                'outcome_prediction': True,
                'schedule_generation': True
            },
            'supported_learning_styles': list(self.learning_styles.keys()),
            'difficulty_levels': list(self.difficulty_levels.keys()),
            'content_adaptation_types': [
                'Visual content optimization',
                'Auditory enhancement',
                'Interactive elements',
                'Text-based materials',
                'Mixed-modal content'
            ],
            'prediction_capabilities': [
                'Completion timeline',
                'Performance trends',
                'Risk factor identification',
                'Outcome probability'
            ]
        }

    def export_learning_analytics(self):
        """NEW: Export comprehensive learning analytics data"""
        
        return {
            'learning_patterns': self.learning_patterns,
            'style_preferences': self.learning_styles,
            'difficulty_configurations': self.difficulty_levels,
            'export_timestamp': datetime.now().isoformat(),
            'adaptive_engine_version': 'Enhanced v2.0'
        }
