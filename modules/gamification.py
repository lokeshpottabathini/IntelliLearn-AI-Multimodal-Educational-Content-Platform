import streamlit as st
from datetime import datetime, timedelta
import json
import random

# Enhanced imports for difficulty-aware gamification
try:
    from .gamification_enhanced import EnhancedGamificationEngine
    ENHANCED_MODULES_AVAILABLE = True
except ImportError:
    ENHANCED_MODULES_AVAILABLE = False

class GamificationEngine:
    def __init__(self):
        """Initialize enhanced gamification with difficulty levels and expanded achievements"""
        
        # Enhanced achievement system with difficulty multipliers
        self.badges = {
            # Basic Learning Achievements
            'first_chapter': {
                'name': '📚 First Reader', 
                'description': 'Completed your first chapter',
                'icon': '📚',
                'points': 50,
                'category': 'learning',
                'difficulty_multiplier': {'beginner': 1.2, 'intermediate': 1.0, 'advanced': 0.8}
            },
            'chapter_champion': {
                'name': '🏆 Chapter Champion', 
                'description': 'Completed 10 chapters',
                'icon': '🏆',
                'points': 500,
                'category': 'learning',
                'difficulty_multiplier': {'beginner': 1.5, 'intermediate': 1.0, 'advanced': 0.7}
            },
            'knowledge_marathon': {
                'name': '🎯 Knowledge Marathon',
                'description': 'Completed 25 chapters',
                'icon': '🎯',
                'points': 1000,
                'category': 'learning',
                'difficulty_multiplier': {'beginner': 2.0, 'intermediate': 1.0, 'advanced': 0.6}
            },
            
            # Streak Achievements
            'streak_3': {
                'name': '🔥 Getting Started',
                'description': '3-day learning streak',
                'icon': '🔥',
                'points': 75,
                'category': 'streak',
                'difficulty_multiplier': {'beginner': 1.3, 'intermediate': 1.0, 'advanced': 0.8}
            },
            'streak_7': {
                'name': '🔥 Week Warrior', 
                'description': '7-day learning streak',
                'icon': '🔥',
                'points': 200,
                'category': 'streak',
                'difficulty_multiplier': {'beginner': 1.5, 'intermediate': 1.0, 'advanced': 0.7}
            },
            'streak_30': {
                'name': '🔥 Monthly Master',
                'description': '30-day learning streak',
                'icon': '🔥',
                'points': 1000,
                'category': 'streak',
                'difficulty_multiplier': {'beginner': 2.0, 'intermediate': 1.0, 'advanced': 0.5}
            },
            
            # Quiz Achievements
            'first_quiz': {
                'name': '🎓 Quiz Rookie',
                'description': 'Completed your first quiz',
                'icon': '🎓',
                'points': 25,
                'category': 'quiz',
                'difficulty_multiplier': {'beginner': 1.5, 'intermediate': 1.0, 'advanced': 0.8}
            },
            'quiz_master': {
                'name': '🧠 Quiz Master', 
                'description': '90%+ average on quizzes',
                'icon': '🧠',
                'points': 300,
                'category': 'quiz',
                'difficulty_multiplier': {'beginner': 2.0, 'intermediate': 1.0, 'advanced': 0.6}
            },
            'perfect_scorer': {
                'name': '⭐ Perfect Scorer', 
                'description': 'Scored 100% on a quiz',
                'icon': '⭐',
                'points': 150,
                'category': 'quiz',
                'difficulty_multiplier': {'beginner': 1.8, 'intermediate': 1.0, 'advanced': 0.7}
            },
            'quiz_marathon': {
                'name': '📝 Quiz Marathon',
                'description': 'Completed 50 quizzes',
                'icon': '📝',
                'points': 750,
                'category': 'quiz',
                'difficulty_multiplier': {'beginner': 2.5, 'intermediate': 1.0, 'advanced': 0.5}
            },
            
            # Content Creation Achievements
            'video_creator': {
                'name': '🎬 Content Creator', 
                'description': 'Generated your first video',
                'icon': '🎬',
                'points': 100,
                'category': 'creation',
                'difficulty_multiplier': {'beginner': 1.0, 'intermediate': 1.0, 'advanced': 1.2}
            },
            'video_producer': {
                'name': '🎥 Video Producer',
                'description': 'Generated 10 educational videos',
                'icon': '🎥',
                'points': 500,
                'category': 'creation',
                'difficulty_multiplier': {'beginner': 0.8, 'intermediate': 1.0, 'advanced': 1.5}
            },
            
            # Interaction Achievements
            'knowledge_seeker': {
                'name': '🔍 Knowledge Seeker', 
                'description': 'Asked 50+ questions',
                'icon': '🔍',
                'points': 250,
                'category': 'interaction',
                'difficulty_multiplier': {'beginner': 1.2, 'intermediate': 1.0, 'advanced': 1.3}
            },
            'curious_mind': {
                'name': '🤔 Curious Mind',
                'description': 'Asked 100+ questions',
                'icon': '🤔',
                'points': 500,
                'category': 'interaction',
                'difficulty_multiplier': {'beginner': 1.5, 'intermediate': 1.0, 'advanced': 1.5}
            },
            
            # NEW: Voice Learning Achievements
            'voice_pioneer': {
                'name': '🎤 Voice Pioneer',
                'description': 'First voice interaction',
                'icon': '🎤',
                'points': 75,
                'category': 'voice',
                'difficulty_multiplier': {'beginner': 1.3, 'intermediate': 1.0, 'advanced': 1.0}
            },
            'voice_master': {
                'name': '🗣️ Voice Master',
                'description': '50+ voice interactions',
                'icon': '🗣️',
                'points': 400,
                'category': 'voice',
                'difficulty_multiplier': {'beginner': 1.8, 'intermediate': 1.0, 'advanced': 1.2}
            },
            
            # NEW: Difficulty-Specific Achievements
            'beginner_graduate': {
                'name': '🌱 Beginner Graduate',
                'description': 'Mastered beginner level content',
                'icon': '🌱',
                'points': 300,
                'category': 'difficulty',
                'difficulty_multiplier': {'beginner': 2.0, 'intermediate': 0.0, 'advanced': 0.0}
            },
            'intermediate_scholar': {
                'name': '📊 Intermediate Scholar',
                'description': 'Excelled at intermediate level',
                'icon': '📊',
                'points': 500,
                'category': 'difficulty',
                'difficulty_multiplier': {'beginner': 0.0, 'intermediate': 2.0, 'advanced': 0.0}
            },
            'advanced_expert': {
                'name': '🚀 Advanced Expert',
                'description': 'Conquered advanced challenges',
                'icon': '🚀',
                'points': 800,
                'category': 'difficulty',
                'difficulty_multiplier': {'beginner': 0.0, 'intermediate': 0.0, 'advanced': 2.0}
            },
            
            # NEW: Special Achievements
            'early_bird': {
                'name': '🌅 Early Bird',
                'description': 'Learned before 8 AM',
                'icon': '🌅',
                'points': 50,
                'category': 'special',
                'difficulty_multiplier': {'beginner': 1.0, 'intermediate': 1.0, 'advanced': 1.0}
            },
            'night_owl': {
                'name': '🦉 Night Owl',
                'description': 'Learned after 10 PM',
                'icon': '🦉',
                'points': 50,
                'category': 'special',
                'difficulty_multiplier': {'beginner': 1.0, 'intermediate': 1.0, 'advanced': 1.0}
            },
            'consistency_king': {
                'name': '👑 Consistency King',
                'description': 'Learned every day for a month',
                'icon': '👑',
                'points': 1500,
                'category': 'special',
                'difficulty_multiplier': {'beginner': 3.0, 'intermediate': 2.0, 'advanced': 1.5}
            }
        }
        
        # Enhanced level system with difficulty awareness
        self.levels = [
            {
                'name': 'Novice Explorer', 
                'icon': '🌱',
                'level': 1,
                'min_points': 0, 
                'max_points': 249,
                'color': '#95a5a6',
                'description': 'Just starting your learning journey'
            },
            {
                'name': 'Eager Student',
                'icon': '📚', 
                'level': 2,
                'min_points': 250, 
                'max_points': 499,
                'color': '#3498db',
                'description': 'Building foundational knowledge'
            },
            {
                'name': 'Dedicated Learner',
                'icon': '🎯',
                'level': 3, 
                'min_points': 500, 
                'max_points': 999,
                'color': '#2ecc71',
                'description': 'Making steady progress'
            },
            {
                'name': 'Knowledge Seeker',
                'icon': '🔍',
                'level': 4,
                'min_points': 1000, 
                'max_points': 1999,
                'color': '#f39c12',
                'description': 'Actively pursuing knowledge'
            },
            {
                'name': 'Skilled Scholar',
                'icon': '🧠',
                'level': 5,
                'min_points': 2000, 
                'max_points': 3499,
                'color': '#e74c3c',
                'description': 'Developing expertise'
            },
            {
                'name': 'Advanced Expert',
                'icon': '🚀',
                'level': 6,
                'min_points': 3500, 
                'max_points': 5999,
                'color': '#9b59b6',
                'description': 'Mastering complex concepts'
            },
            {
                'name': 'Learning Master',
                'icon': '👑',
                'level': 7,
                'min_points': 6000, 
                'max_points': 9999,
                'color': '#8e44ad',
                'description': 'Exceptional learning mastery'
            },
            {
                'name': 'Wisdom Keeper',
                'icon': '🧙‍♂️',
                'level': 8,
                'min_points': 10000, 
                'max_points': float('inf'),
                'color': '#2c3e50',
                'description': 'Legendary learning achievement'
            }
        ]
        
        # Enhanced tracking categories
        self.achievement_categories = {
            'learning': '📚 Learning Progress',
            'quiz': '🧠 Quiz Performance', 
            'streak': '🔥 Learning Streaks',
            'creation': '🎬 Content Creation',
            'interaction': '💬 Active Engagement',
            'voice': '🎤 Voice Learning',
            'difficulty': '🎯 Level Mastery',
            'special': '⭐ Special Achievements'
        }
        
        # Initialize enhanced gamification if available
        self.enhanced_engine = None
        if ENHANCED_MODULES_AVAILABLE:
            try:
                self.enhanced_engine = EnhancedGamificationEngine()
                st.success("✅ Enhanced gamification features loaded")
            except Exception as e:
                st.warning(f"Enhanced gamification not available: {e}")
    
    def initialize_user_progress(self, difficulty_level='intermediate'):
        """Initialize enhanced user progress tracking with difficulty awareness"""
        if 'user_progress' not in st.session_state:
            st.session_state.user_progress = {
                # Core Progress Tracking
                'total_points': 0,
                'badges_earned': [],
                'chapters_completed': 0,
                'quizzes_taken': 0,
                'videos_generated': 0,
                'questions_asked': 0,
                'learning_streak': 0,
                'last_activity_date': None,
                'quiz_scores': [],
                'achievements_history': [],
                
                # Enhanced Progress Tracking
                'difficulty_level': difficulty_level,
                'voice_interactions': 0,
                'ai_conversations': 0,
                'perfect_quizzes': 0,
                'textbooks_processed': 0,
                'image_analyses': 0,
                'learning_sessions': 0,
                'total_study_time': 0,
                
                # Difficulty-Specific Tracking
                'beginner_quizzes': 0,
                'intermediate_quizzes': 0,
                'advanced_quizzes': 0,
                'beginner_scores': [],
                'intermediate_scores': [],
                'advanced_scores': [],
                
                # Time-Based Tracking
                'daily_activities': {},
                'weekly_goals': {},
                'monthly_progress': {},
                
                # Enhanced Metrics
                'favorite_subjects': {},
                'learning_patterns': {},
                'achievement_streaks': {},
                
                # Notification Settings
                'notification_settings': {
                    'achievements': True,
                    'streaks': True,
                    'level_ups': True,
                    'daily_reminders': False
                }
            }
        
        # Update difficulty level if changed
        if st.session_state.user_progress.get('difficulty_level') != difficulty_level:
            st.session_state.user_progress['difficulty_level'] = difficulty_level

    def check_achievements_enhanced(self, user_progress, activity_type=None, activity_data=None):
        """Enhanced achievement checking with difficulty awareness and activity context"""
        
        # Use enhanced engine if available
        if self.enhanced_engine:
            return self.enhanced_engine.check_achievements(user_progress, activity_type, activity_data)
        
        # Fallback to enhanced standard checking
        new_badges = []
        current_difficulty = user_progress.get('difficulty_level', 'intermediate')
        
        # Basic Learning Achievements
        if (user_progress.get('chapters_completed', 0) >= 1 and 
            'first_chapter' not in user_progress.get('badges_earned', [])):
            new_badges.append('first_chapter')
        
        if (user_progress.get('chapters_completed', 0) >= 10 and 
            'chapter_champion' not in user_progress.get('badges_earned', [])):
            new_badges.append('chapter_champion')
        
        if (user_progress.get('chapters_completed', 0) >= 25 and 
            'knowledge_marathon' not in user_progress.get('badges_earned', [])):
            new_badges.append('knowledge_marathon')
        
        # Enhanced Streak Achievements
        streak = user_progress.get('learning_streak', 0)
        if streak >= 3 and 'streak_3' not in user_progress.get('badges_earned', []):
            new_badges.append('streak_3')
        if streak >= 7 and 'streak_7' not in user_progress.get('badges_earned', []):
            new_badges.append('streak_7')
        if streak >= 30 and 'streak_30' not in user_progress.get('badges_earned', []):
            new_badges.append('streak_30')
        
        # Quiz Achievements with Difficulty Awareness
        if (user_progress.get('quizzes_taken', 0) >= 1 and 
            'first_quiz' not in user_progress.get('badges_earned', [])):
            new_badges.append('first_quiz')
        
        # Difficulty-specific quiz scoring
        difficulty_scores = user_progress.get(f'{current_difficulty}_scores', [])
        all_scores = user_progress.get('quiz_scores', [])
        
        if (difficulty_scores and sum(difficulty_scores) / len(difficulty_scores) >= 90 and 
            'quiz_master' not in user_progress.get('badges_earned', [])):
            new_badges.append('quiz_master')
        
        if (all_scores and 100 in all_scores and 
            'perfect_scorer' not in user_progress.get('badges_earned', [])):
            new_badges.append('perfect_scorer')
        
        if (user_progress.get('quizzes_taken', 0) >= 50 and 
            'quiz_marathon' not in user_progress.get('badges_earned', [])):
            new_badges.append('quiz_marathon')
        
        # Content Creation Achievements
        if (user_progress.get('videos_generated', 0) >= 1 and 
            'video_creator' not in user_progress.get('badges_earned', [])):
            new_badges.append('video_creator')
        
        if (user_progress.get('videos_generated', 0) >= 10 and 
            'video_producer' not in user_progress.get('badges_earned', [])):
            new_badges.append('video_producer')
        
        # Interaction Achievements
        if (user_progress.get('questions_asked', 0) >= 50 and 
            'knowledge_seeker' not in user_progress.get('badges_earned', [])):
            new_badges.append('knowledge_seeker')
        
        if (user_progress.get('questions_asked', 0) >= 100 and 
            'curious_mind' not in user_progress.get('badges_earned', [])):
            new_badges.append('curious_mind')
        
        # NEW: Voice Learning Achievements
        if (user_progress.get('voice_interactions', 0) >= 1 and 
            'voice_pioneer' not in user_progress.get('badges_earned', [])):
            new_badges.append('voice_pioneer')
        
        if (user_progress.get('voice_interactions', 0) >= 50 and 
            'voice_master' not in user_progress.get('badges_earned', [])):
            new_badges.append('voice_master')
        
        # NEW: Difficulty-Specific Achievements
        self._check_difficulty_achievements(user_progress, new_badges, current_difficulty)
        
        # NEW: Time-Based Special Achievements
        self._check_special_achievements(user_progress, new_badges, activity_type, activity_data)
        
        # Award all new badges
        for badge_id in new_badges:
            self.award_badge_enhanced(badge_id, user_progress, current_difficulty)
        
        return new_badges

    def _check_difficulty_achievements(self, user_progress, new_badges, current_difficulty):
        """Check for difficulty-specific achievements"""
        
        # Beginner mastery (10+ beginner quizzes with 85%+ average)
        beginner_scores = user_progress.get('beginner_scores', [])
        if (len(beginner_scores) >= 10 and 
            sum(beginner_scores) / len(beginner_scores) >= 85 and
            'beginner_graduate' not in user_progress.get('badges_earned', [])):
            new_badges.append('beginner_graduate')
        
        # Intermediate mastery (15+ intermediate quizzes with 85%+ average)
        intermediate_scores = user_progress.get('intermediate_scores', [])
        if (len(intermediate_scores) >= 15 and 
            sum(intermediate_scores) / len(intermediate_scores) >= 85 and
            'intermediate_scholar' not in user_progress.get('badges_earned', [])):
            new_badges.append('intermediate_scholar')
        
        # Advanced mastery (20+ advanced quizzes with 80%+ average)
        advanced_scores = user_progress.get('advanced_scores', [])
        if (len(advanced_scores) >= 20 and 
            sum(advanced_scores) / len(advanced_scores) >= 80 and
            'advanced_expert' not in user_progress.get('badges_earned', [])):
            new_badges.append('advanced_expert')

    def _check_special_achievements(self, user_progress, new_badges, activity_type, activity_data):
        """Check for special time-based and activity-specific achievements"""
        
        current_hour = datetime.now().hour
        
        # Early Bird (activity before 8 AM)
        if (current_hour < 8 and activity_type and
            'early_bird' not in user_progress.get('badges_earned', [])):
            new_badges.append('early_bird')
        
        # Night Owl (activity after 10 PM)
        if (current_hour >= 22 and activity_type and
            'night_owl' not in user_progress.get('badges_earned', [])):
            new_badges.append('night_owl')
        
        # Consistency King (30-day streak with daily activities)
        daily_activities = user_progress.get('daily_activities', {})
        if (len(daily_activities) >= 30 and user_progress.get('learning_streak', 0) >= 30 and
            'consistency_king' not in user_progress.get('badges_earned', [])):
            # Check if activities are consecutive
            dates = sorted(daily_activities.keys())
            if len(dates) >= 30:
                # Check last 30 dates are consecutive
                recent_dates = dates[-30:]
                consecutive = True
                for i in range(1, len(recent_dates)):
                    date1 = datetime.fromisoformat(recent_dates[i-1]).date()
                    date2 = datetime.fromisoformat(recent_dates[i]).date()
                    if (date2 - date1).days > 1:
                        consecutive = False
                        break
                
                if consecutive:
                    new_badges.append('consistency_king')

    def award_badge_enhanced(self, badge_id, user_progress, difficulty_level='intermediate'):
        """Enhanced badge awarding with difficulty multipliers and better tracking"""
        
        if badge_id in self.badges:
            badge = self.badges[badge_id]
            
            # Calculate points with difficulty multiplier
            base_points = badge['points']
            multiplier = badge.get('difficulty_multiplier', {}).get(difficulty_level, 1.0)
            final_points = int(base_points * multiplier)
            
            # Add to earned badges
            if 'badges_earned' not in user_progress:
                user_progress['badges_earned'] = []
            user_progress['badges_earned'].append(badge_id)
            
            # Add points
            user_progress['total_points'] = user_progress.get('total_points', 0) + final_points
            
            # Add to achievements history with enhanced data
            if 'achievements_history' not in user_progress:
                user_progress['achievements_history'] = []
            
            achievement_record = {
                'badge_id': badge_id,
                'date': datetime.now().isoformat(),
                'points_earned': final_points,
                'base_points': base_points,
                'difficulty_multiplier': multiplier,
                'difficulty_level': difficulty_level,
                'category': badge.get('category', 'general'),
                'user_level': self.get_user_level(user_progress.get('total_points', 0))['name']
            }
            user_progress['achievements_history'].append(achievement_record)
            
            # Update achievement streaks
            category = badge.get('category', 'general')
            if 'achievement_streaks' not in user_progress:
                user_progress['achievement_streaks'] = {}
            
            user_progress['achievement_streaks'][category] = user_progress['achievement_streaks'].get(category, 0) + 1
            
            # Show enhanced celebration
            st.balloons()
            
            # Difficulty-specific celebration messages
            celebration_messages = {
                'beginner': f"🌟 Amazing progress! You've earned the {badge['icon']} {badge['name']} badge!",
                'intermediate': f"🎉 Great achievement! {badge['icon']} {badge['name']} badge earned!",
                'advanced': f"🚀 Outstanding! You've unlocked {badge['icon']} {badge['name']}!"
            }
            
            message = celebration_messages.get(difficulty_level, f"🎉 Badge Earned: {badge['icon']} {badge['name']}!")
            
            # Show points with multiplier info
            if multiplier != 1.0:
                st.success(f"{message}\n💰 {final_points} points ({base_points} × {multiplier:.1f} {difficulty_level} bonus)")
            else:
                st.success(f"{message}\n💰 +{final_points} points")
            
            # Check for level up
            old_level = self.get_user_level(user_progress.get('total_points', 0) - final_points)
            new_level = self.get_user_level(user_progress.get('total_points', 0))
            
            if old_level['level'] < new_level['level']:
                st.success(f"🎊 LEVEL UP! You're now a {new_level['icon']} {new_level['name']}!")
                st.balloons()

    def check_achievements(self, user_progress):
        """Original achievement checking method for backward compatibility"""
        return self.check_achievements_enhanced(user_progress)

    def award_badge(self, badge_id, user_progress):
        """Original badge awarding method for backward compatibility"""
        difficulty_level = user_progress.get('difficulty_level', 'intermediate')
        return self.award_badge_enhanced(badge_id, user_progress, difficulty_level)

    def get_user_level(self, total_points):
        """Get user's current level based on points with enhanced information"""
        for level in self.levels:
            if level['min_points'] <= total_points <= level['max_points']:
                return level
        return self.levels[0]  # Default to first level

    def get_next_level(self, total_points):
        """Get the next level information"""
        current_level = self.get_user_level(total_points)
        current_index = None
        
        for i, level in enumerate(self.levels):
            if level['level'] == current_level['level']:
                current_index = i
                break
        
        if current_index is not None and current_index < len(self.levels) - 1:
            return self.levels[current_index + 1]
        
        return None  # Already at max level

    def display_progress_dashboard_enhanced(self, user_progress):
        """Enhanced progress dashboard with difficulty awareness and comprehensive metrics"""
        
        # Use enhanced engine if available
        if self.enhanced_engine:
            return self.enhanced_engine.display_enhanced_progress_dashboard()
        
        # Enhanced dashboard display
        st.subheader("🏆 Your Enhanced Learning Progress")
        
        # Current level and points with enhanced display
        total_points = user_progress.get('total_points', 0)
        current_level = self.get_user_level(total_points)
        next_level = self.get_next_level(total_points)
        difficulty_level = user_progress.get('difficulty_level', 'intermediate')
        
        # Top metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label=f"{current_level['icon']} Level",
                value=current_level['name'],
                help=f"Level {current_level['level']} • {total_points:,} total points"
            )
        
        with col2:
            st.metric(
                label="📚 Chapters",
                value=user_progress.get('chapters_completed', 0),
                help="Total chapters completed across all difficulty levels"
            )
        
        with col3:
            streak = user_progress.get('learning_streak', 0)
            st.metric(
                label="🔥 Streak",
                value=f"{streak} days",
                help="Consecutive days of learning activity"
            )
        
        with col4:
            badges_count = len(user_progress.get('badges_earned', []))
            total_badges = len(self.badges)
            st.metric(
                label="🏅 Badges",
                value=f"{badges_count}/{total_badges}",
                help="Achievement badges earned"
            )
        
        with col5:
            # Difficulty-specific performance
            difficulty_scores = user_progress.get(f'{difficulty_level}_scores', [])
            if difficulty_scores:
                avg_score = sum(difficulty_scores) / len(difficulty_scores)
                st.metric(
                    label=f"📊 {difficulty_level.title()} Avg",
                    value=f"{avg_score:.1f}%",
                    help=f"Average score at {difficulty_level} difficulty"
                )
            else:
                st.metric(
                    label=f"🎯 Current Level",
                    value=difficulty_level.title(),
                    help="Current difficulty setting"
                )
        
        # Progress bar to next level with enhanced display
        if next_level:
            points_needed = next_level['min_points'] - total_points
            points_in_level = total_points - current_level['min_points']
            points_for_level = next_level['min_points'] - current_level['min_points']
            progress = points_in_level / points_for_level if points_for_level > 0 else 0
            
            st.markdown("### 📈 Progress to Next Level")
            st.progress(progress)
            st.info(f"🎯 {points_needed:,} more points to reach {next_level['icon']} {next_level['name']}!")
        else:
            st.success("🎉 Congratulations! You've reached the maximum level!")
        
        # Enhanced metrics section
        st.markdown("### 📊 Detailed Progress")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**📚 Learning Activity**")
            st.write(f"• Quizzes Taken: {user_progress.get('quizzes_taken', 0)}")
            st.write(f"• Videos Created: {user_progress.get('videos_generated', 0)}")
            st.write(f"• Questions Asked: {user_progress.get('questions_asked', 0)}")
            st.write(f"• Voice Chats: {user_progress.get('voice_interactions', 0)}")
        
        with col2:
            st.markdown("**🎯 Performance by Difficulty**")
            for diff in ['beginner', 'intermediate', 'advanced']:
                scores = user_progress.get(f'{diff}_scores', [])
                if scores:
                    avg = sum(scores) / len(scores)
                    st.write(f"• {diff.title()}: {avg:.1f}% ({len(scores)} quizzes)")
                else:
                    st.write(f"• {diff.title()}: No data")
        
        with col3:
            st.markdown("**⏰ Activity Patterns**")
            total_sessions = user_progress.get('learning_sessions', 0)
            total_time = user_progress.get('total_study_time', 0)
            
            if total_sessions > 0:
                avg_session = total_time / total_sessions if total_sessions > 0 else 0
                st.write(f"• Sessions: {total_sessions}")
                st.write(f"• Total Time: {total_time:.1f} min")
                st.write(f"• Avg/Session: {avg_session:.1f} min")
            else:
                st.write("• No session data yet")
            
            # Last activity
            last_activity = user_progress.get('last_activity_date')
            if last_activity:
                last_date = datetime.fromisoformat(last_activity).date()
                days_ago = (datetime.now().date() - last_date).days
                st.write(f"• Last Activity: {days_ago} days ago")
        
        with col4:
            st.markdown("**🏆 Achievement Categories**")
            category_counts = {}
            for badge_id in user_progress.get('badges_earned', []):
                if badge_id in self.badges:
                    category = self.badges[badge_id].get('category', 'general')
                    category_counts[category] = category_counts.get(category, 0) + 1
            
            for category, name in self.achievement_categories.items():
                count = category_counts.get(category, 0)
                total_in_category = sum(1 for badge in self.badges.values() if badge.get('category') == category)
                st.write(f"• {name}: {count}/{total_in_category}")
        
        # Enhanced badges display with categories
        st.markdown("### 🏅 Your Achievement Collection")
        
        earned_badges = user_progress.get('badges_earned', [])
        
        if earned_badges:
            # Group badges by category
            badges_by_category = {}
            for badge_id in earned_badges:
                if badge_id in self.badges:
                    category = self.badges[badge_id].get('category', 'general')
                    if category not in badges_by_category:
                        badges_by_category[category] = []
                    badges_by_category[category].append(badge_id)
            
            # Display badges by category
            for category, category_name in self.achievement_categories.items():
                if category in badges_by_category:
                    st.markdown(f"**{category_name}**")
                    
                    badge_cols = st.columns(min(len(badges_by_category[category]), 6))
                    for i, badge_id in enumerate(badges_by_category[category]):
                        badge = self.badges[badge_id]
                        with badge_cols[i % 6]:
                            # Calculate points earned with difficulty multiplier
                            difficulty = user_progress.get('difficulty_level', 'intermediate')
                            multiplier = badge.get('difficulty_multiplier', {}).get(difficulty, 1.0)
                            final_points = int(badge['points'] * multiplier)
                            
                            st.markdown(f"""
                            <div style="text-align: center; padding: 0.8rem; background: #f8f9fa; border-radius: 8px; margin: 0.3rem 0; border-left: 4px solid {current_level['color']};">
                                <div style="font-size: 1.8rem;">{badge['icon']}</div>
                                <div style="font-weight: bold; font-size: 0.85rem; margin: 0.2rem 0;">{badge['name']}</div>
                                <div style="font-size: 0.75rem; color: #666; margin: 0.2rem 0;">{badge['description']}</div>
                                <div style="font-size: 0.75rem; color: #e74c3c; font-weight: bold;">+{final_points} pts</div>
                                {f'<div style="font-size: 0.7rem; color: #3498db;">({multiplier:.1f}x {difficulty})</div>' if multiplier != 1.0 else ''}
                            </div>
                            """, unsafe_allow_html=True)
        else:
            st.info("🎯 Complete learning activities to earn your first badges!")
            
            # Show some example badges they can earn
            st.markdown("**🎯 Available Achievements:**")
            example_badges = ['first_chapter', 'first_quiz', 'streak_3', 'video_creator', 'voice_pioneer']
            
            badge_cols = st.columns(len(example_badges))
            for i, badge_id in enumerate(example_badges):
                if badge_id in self.badges:
                    badge = self.badges[badge_id]
                    with badge_cols[i]:
                        st.markdown(f"""
                        <div style="text-align: center; padding: 0.8rem; background: #f0f0f0; border-radius: 8px; opacity: 0.7;">
                            <div style="font-size: 1.5rem;">{badge['icon']}</div>
                            <div style="font-size: 0.8rem; font-weight: bold;">{badge['name']}</div>
                            <div style="font-size: 0.7rem; color: #666;">{badge['points']} pts</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Enhanced recent achievements with timeline
        achievements = user_progress.get('achievements_history', [])
        if achievements:
            st.markdown("### 📈 Recent Achievement Timeline")
            
            # Show last 5 achievements
            for achievement in achievements[-5:]:
                badge = self.badges.get(achievement['badge_id'], {})
                date_str = achievement.get('date', '')
                if date_str:
                    try:
                        date_obj = datetime.fromisoformat(date_str)
                        formatted_date = date_obj.strftime("%b %d, %Y")
                        time_str = date_obj.strftime("%I:%M %p")
                    except:
                        formatted_date = date_str[:10]
                        time_str = ""
                else:
                    formatted_date = "Unknown date"
                    time_str = ""
                
                points = achievement.get('points_earned', 0)
                difficulty = achievement.get('difficulty_level', 'intermediate')
                multiplier = achievement.get('difficulty_multiplier', 1.0)
                
                # Create timeline entry
                if multiplier != 1.0:
                    points_info = f"+{points} pts ({multiplier:.1f}x {difficulty})"
                else:
                    points_info = f"+{points} pts"
                
                st.markdown(f"""
                <div style="padding: 0.5rem; margin: 0.3rem 0; border-left: 3px solid {current_level['color']}; background: #f8f9fa;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-size: 1.2rem;">{badge.get('icon', '🏅')}</span>
                            <strong>{badge.get('name', 'Achievement')}</strong>
                            <span style="color: #e74c3c; font-weight: bold; margin-left: 10px;">{points_info}</span>
                        </div>
                        <div style="font-size: 0.8rem; color: #666;">
                            {formatted_date} {time_str}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    def display_progress_dashboard(self, user_progress):
        """Original progress dashboard method for backward compatibility"""
        return self.display_progress_dashboard_enhanced(user_progress)

    def update_learning_streak_enhanced(self, user_progress):
        """Enhanced learning streak tracking with detailed activity logging"""
        today = datetime.now().date()
        today_str = today.isoformat()
        last_activity = user_progress.get('last_activity_date')
        
        # Initialize daily activities tracking
        if 'daily_activities' not in user_progress:
            user_progress['daily_activities'] = {}
        
        # Log today's activity
        if today_str not in user_progress['daily_activities']:
            user_progress['daily_activities'][today_str] = {
                'date': today_str,
                'activities': [],
                'total_time': 0,
                'points_earned': 0
            }
        
        if last_activity:
            last_date = datetime.fromisoformat(last_activity).date()
            days_diff = (today - last_date).days
            
            if days_diff == 1:
                # Continue streak
                user_progress['learning_streak'] = user_progress.get('learning_streak', 0) + 1
            elif days_diff > 1:
                # Reset streak
                user_progress['learning_streak'] = 1
            # If days_diff == 0, same day activity, don't change streak
        else:
            # First activity
            user_progress['learning_streak'] = 1
        
        user_progress['last_activity_date'] = today.isoformat()
        user_progress['learning_sessions'] = user_progress.get('learning_sessions', 0) + 1
        
        return user_progress['learning_streak']

    def update_learning_streak(self, user_progress):
        """Original learning streak update method for backward compatibility"""
        return self.update_learning_streak_enhanced(user_progress)

    def log_activity(self, activity_type, activity_data=None, duration=0):
        """Log detailed learning activity for enhanced tracking"""
        
        if 'user_progress' not in st.session_state:
            self.initialize_user_progress()
        
        user_progress = st.session_state.user_progress
        today = datetime.now().date().isoformat()
        
        # Initialize daily activities if needed
        if 'daily_activities' not in user_progress:
            user_progress['daily_activities'] = {}
        
        if today not in user_progress['daily_activities']:
            user_progress['daily_activities'][today] = {
                'date': today,
                'activities': [],
                'total_time': 0,
                'points_earned': 0
            }
        
        # Log the activity
        activity_record = {
            'type': activity_type,
            'timestamp': datetime.now().isoformat(),
            'duration': duration,
            'data': activity_data or {}
        }
        
        user_progress['daily_activities'][today]['activities'].append(activity_record)
        user_progress['daily_activities'][today]['total_time'] += duration
        
        # Update global counters
        user_progress['total_study_time'] = user_progress.get('total_study_time', 0) + duration
        
        # Update streak
        self.update_learning_streak_enhanced(user_progress)
        
        # Check for achievements with activity context
        new_achievements = self.check_achievements_enhanced(user_progress, activity_type, activity_data)
        
        return new_achievements

    def get_learning_stats(self):
        """Get comprehensive learning statistics"""
        
        if 'user_progress' not in st.session_state:
            return {}
        
        user_progress = st.session_state.user_progress
        
        # Calculate various statistics
        stats = {
            'total_points': user_progress.get('total_points', 0),
            'current_level': self.get_user_level(user_progress.get('total_points', 0)),
            'badges_earned': len(user_progress.get('badges_earned', [])),
            'total_badges': len(self.badges),
            'completion_rate': len(user_progress.get('badges_earned', [])) / len(self.badges) * 100,
            'learning_streak': user_progress.get('learning_streak', 0),
            'total_activities': len(user_progress.get('daily_activities', {})),
            'difficulty_level': user_progress.get('difficulty_level', 'intermediate')
        }
        
        # Calculate difficulty-specific stats
        for difficulty in ['beginner', 'intermediate', 'advanced']:
            scores = user_progress.get(f'{difficulty}_scores', [])
            stats[f'{difficulty}_quizzes'] = len(scores)
            stats[f'{difficulty}_average'] = sum(scores) / len(scores) if scores else 0
        
        return stats
    
    def export_progress(self):
        """Export user progress data"""
        
        if 'user_progress' not in st.session_state:
            return {}
        
        export_data = {
            'user_progress': st.session_state.user_progress,
            'badges': self.badges,
            'levels': self.levels,
            'export_date': datetime.now().isoformat(),
            'platform_version': 'IntelliLearn AI Enhanced v2.0'
        }
        
        return json.dumps(export_data, indent=2)
