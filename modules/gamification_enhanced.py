import streamlit as st
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np

class EnhancedGamificationEngine:
    def __init__(self):
        """Initialize enhanced gamification engine"""
        
        # Enhanced achievement system with difficulty levels
        self.achievements = {
            # Learning Milestones
            'first_steps': {
                'id': 'first_steps',
                'name': 'First Steps',
                'description': 'Complete your first learning session',
                'icon': '🎯',
                'points': 50,
                'requirements': {'sessions_completed': 1},
                'category': 'milestone',
                'difficulty': 'all'
            },
            'knowledge_seeker': {
                'id': 'knowledge_seeker',
                'name': 'Knowledge Seeker',
                'description': 'Ask 10 questions to the AI tutor',
                'icon': '🔍',
                'points': 100,
                'requirements': {'questions_asked': 10},
                'category': 'engagement',
                'difficulty': 'all'
            },
            'chapter_master': {
                'id': 'chapter_master',
                'name': 'Chapter Master',
                'description': 'Complete 5 chapters',
                'icon': '📚',
                'points': 200,
                'requirements': {'chapters_completed': 5},
                'category': 'progress',
                'difficulty': 'all'
            },
            
            # Difficulty-specific achievements
            'beginner_champion': {
                'id': 'beginner_champion',
                'name': 'Beginner Champion',
                'description': 'Excel at beginner level learning',
                'icon': '🌟',
                'points': 150,
                'requirements': {'beginner_quizzes': 5, 'beginner_avg_score': 80},
                'category': 'difficulty',
                'difficulty': 'beginner'
            },
            'intermediate_scholar': {
                'id': 'intermediate_scholar',
                'name': 'Intermediate Scholar',
                'description': 'Master intermediate concepts',
                'icon': '🎓',
                'points': 250,
                'requirements': {'intermediate_quizzes': 5, 'intermediate_avg_score': 85},
                'category': 'difficulty',
                'difficulty': 'intermediate'
            },
            'advanced_expert': {
                'id': 'advanced_expert',
                'name': 'Advanced Expert',
                'description': 'Conquer advanced challenges',
                'icon': '🏆',
                'points': 350,
                'requirements': {'advanced_quizzes': 5, 'advanced_avg_score': 90},
                'category': 'difficulty',
                'difficulty': 'advanced'
            },
            
            # Quiz Performance
            'quiz_rookie': {
                'id': 'quiz_rookie',
                'name': 'Quiz Rookie',
                'description': 'Take your first quiz',
                'icon': '📝',
                'points': 25,
                'requirements': {'quizzes_taken': 1},
                'category': 'assessment',
                'difficulty': 'all'
            },
            'perfect_score': {
                'id': 'perfect_score',
                'name': 'Perfect Score',
                'description': 'Achieve 100% on a quiz',
                'icon': '💯',
                'points': 300,
                'requirements': {'perfect_quizzes': 1},
                'category': 'excellence',
                'difficulty': 'all'
            },
            'quiz_master': {
                'id': 'quiz_master',
                'name': 'Quiz Master',
                'description': 'Maintain 90%+ average across 10 quizzes',
                'icon': '🎖️',
                'points': 500,
                'requirements': {'quizzes_taken': 10, 'quiz_average': 90},
                'category': 'mastery',
                'difficulty': 'all'
            },
            
            # Learning Streaks
            'week_warrior': {
                'id': 'week_warrior',
                'name': 'Week Warrior',
                'description': 'Maintain a 7-day learning streak',
                'icon': '🔥',
                'points': 200,
                'requirements': {'learning_streak': 7},
                'category': 'consistency',
                'difficulty': 'all'
            },
            'month_master': {
                'id': 'month_master',
                'name': 'Month Master',
                'description': 'Maintain a 30-day learning streak',
                'icon': '🌟',
                'points': 1000,
                'requirements': {'learning_streak': 30},
                'category': 'dedication',
                'difficulty': 'all'
            },
            
            # Video and Voice
            'content_creator': {
                'id': 'content_creator',
                'name': 'Content Creator',
                'description': 'Generate 5 educational videos',
                'icon': '🎬',
                'points': 300,
                'requirements': {'videos_generated': 5},
                'category': 'creation',
                'difficulty': 'all'
            },
            'voice_learner': {
                'id': 'voice_learner',
                'name': 'Voice Learner',
                'description': 'Use voice chat 10 times',
                'icon': '🎤',
                'points': 150,
                'requirements': {'voice_interactions': 10},
                'category': 'interaction',
                'difficulty': 'all'
            },
            
            # Advanced Achievements
            'knowledge_architect': {
                'id': 'knowledge_architect',
                'name': 'Knowledge Architect',
                'description': 'Process and master 10 textbooks',
                'icon': '🏗️',
                'points': 2000,
                'requirements': {'textbooks_processed': 10},
                'category': 'mastery',
                'difficulty': 'advanced'
            },
            'ai_collaborator': {
                'id': 'ai_collaborator',
                'name': 'AI Collaborator',
                'description': 'Have 100+ AI tutor conversations',
                'icon': '🤝',
                'points': 750,
                'requirements': {'ai_conversations': 100},
                'category': 'engagement',
                'difficulty': 'all'
            }
        }
        
        # Enhanced level system with difficulty bonuses
        self.levels = [
            {'level': 1, 'name': 'Learner', 'min_points': 0, 'icon': '📖', 'color': '#4CAF50'},
            {'level': 2, 'name': 'Student', 'min_points': 500, 'icon': '🎓', 'color': '#2196F3'},
            {'level': 3, 'name': 'Scholar', 'min_points': 1500, 'icon': '📚', 'color': '#FF9800'},
            {'level': 4, 'name': 'Expert', 'min_points': 3000, 'icon': '🔬', 'color': '#9C27B0'},
            {'level': 5, 'name': 'Master', 'min_points': 6000, 'icon': '🏆', 'color': '#F44336'},
            {'level': 6, 'name': 'Sage', 'min_points': 10000, 'icon': '🧙‍♂️', 'color': '#673AB7'},
            {'level': 7, 'name': 'Guru', 'min_points': 15000, 'icon': '✨', 'color': '#FF5722'},
            {'level': 8, 'name': 'Legend', 'min_points': 25000, 'icon': '🌟', 'color': '#795548'},
            {'level': 9, 'name': 'Grandmaster', 'min_points': 40000, 'icon': '👑', 'color': '#607D8B'},
            {'level': 10, 'name': 'Enlightened', 'min_points': 60000, 'icon': '🌠', 'color': '#E91E63'}
        ]
        
        # Points system with difficulty multipliers
        self.points_system = {
            'textbook_processed': {'base': 100, 'beginner': 1.0, 'intermediate': 1.2, 'advanced': 1.5},
            'quiz_completed': {'base': 50, 'beginner': 1.0, 'intermediate': 1.3, 'advanced': 1.6},
            'perfect_quiz': {'base': 300, 'beginner': 1.0, 'intermediate': 1.5, 'advanced': 2.0},
            'chapter_completed': {'base': 100, 'beginner': 1.0, 'intermediate': 1.2, 'advanced': 1.4},
            'video_generated': {'base': 75, 'beginner': 1.0, 'intermediate': 1.1, 'advanced': 1.3},
            'voice_interaction': {'base': 10, 'beginner': 1.0, 'intermediate': 1.1, 'advanced': 1.2},
            'streak_day': {'base': 20, 'beginner': 1.0, 'intermediate': 1.0, 'advanced': 1.0},
            'ai_conversation': {'base': 5, 'beginner': 1.0, 'intermediate': 1.1, 'advanced': 1.2}
        }
        
        # Badge categories for organization
        self.badge_categories = {
            'milestone': {'name': 'Milestones', 'color': '#4CAF50', 'icon': '🎯'},
            'excellence': {'name': 'Excellence', 'color': '#FFD700', 'icon': '⭐'},
            'consistency': {'name': 'Consistency', 'color': '#FF6B6B', 'icon': '🔥'},
            'engagement': {'name': 'Engagement', 'color': '#4ECDC4', 'icon': '💬'},
            'mastery': {'name': 'Mastery', 'color': '#A8E6CF', 'icon': '🏆'},
            'difficulty': {'name': 'Difficulty', 'color': '#FFE66D', 'icon': '📊'}
        }
    
    def initialize_user_progress(self, difficulty_level="intermediate"):
        """Initialize enhanced user progress with difficulty tracking"""
        
        if 'user_progress' not in st.session_state:
            st.session_state.user_progress = {}
        
        # Initialize all tracking fields
        defaults = {
            'total_points': 0,
            'current_level': 1,
            'badges_earned': [],
            'achievements_history': [],
            
            # Basic metrics
            'sessions_completed': 0,
            'questions_asked': 0,
            'chapters_completed': 0,
            'quizzes_taken': 0,
            'videos_generated': 0,
            'voice_interactions': 0,
            'ai_conversations': 0,
            'textbooks_processed': 0,
            
            # Quiz performance
            'quiz_scores': [],
            'perfect_quizzes': 0,
            'quiz_average': 0,
            
            # Difficulty-specific tracking
            'beginner_quizzes': 0,
            'intermediate_quizzes': 0,
            'advanced_quizzes': 0,
            'beginner_scores': [],
            'intermediate_scores': [],
            'advanced_scores': [],
            'beginner_avg_score': 0,
            'intermediate_avg_score': 0,
            'advanced_avg_score': 0,
            
            # Streak tracking
            'learning_streak': 0,
            'longest_streak': 0,
            'last_activity_date': None,
            'streak_freeze_used': 0,
            
            # Time tracking
            'total_learning_time': 0,
            'session_start_time': None,
            'daily_goal': 30,  # minutes
            'weekly_goal': 300,  # minutes
            
            # Preferences
            'preferred_difficulty': difficulty_level,
            'notification_settings': {
                'achievements': True,
                'streaks': True,
                'level_ups': True
            }
        }
        
        # Initialize missing fields
        for key, value in defaults.items():
            if key not in st.session_state.user_progress:
                st.session_state.user_progress[key] = value
        
        # Update current level based on points
        self._update_user_level()
    
    def award_points(self, action_type, difficulty_level="intermediate", bonus_multiplier=1.0):
        """Award points with difficulty multipliers and bonuses"""
        
        if action_type not in self.points_system:
            return 0
        
        point_config = self.points_system[action_type]
        base_points = point_config['base']
        difficulty_multiplier = point_config.get(difficulty_level, 1.0)
        
        # Calculate final points
        final_points = int(base_points * difficulty_multiplier * bonus_multiplier)
        
        # Award points
        st.session_state.user_progress['total_points'] += final_points
        
        # Check for level up
        old_level = st.session_state.user_progress['current_level']
        self._update_user_level()
        new_level = st.session_state.user_progress['current_level']
        
        # Show point award notification
        if final_points > 0:
            self._show_points_notification(final_points, action_type, difficulty_level)
        
        # Show level up notification
        if new_level > old_level:
            self._show_level_up_notification(new_level)
        
        return final_points
    
    def update_quiz_performance(self, score, difficulty_level="intermediate"):
        """Update quiz performance with difficulty tracking"""
        
        # Update overall quiz stats
        st.session_state.user_progress['quizzes_taken'] += 1
        st.session_state.user_progress['quiz_scores'].append(score)
        
        # Calculate new average
        scores = st.session_state.user_progress['quiz_scores']
        st.session_state.user_progress['quiz_average'] = sum(scores) / len(scores)
        
        # Update difficulty-specific stats
        difficulty_key = f'{difficulty_level}_quizzes'
        scores_key = f'{difficulty_level}_scores'
        avg_key = f'{difficulty_level}_avg_score'
        
        st.session_state.user_progress[difficulty_key] += 1
        st.session_state.user_progress[scores_key].append(score)
        
        # Calculate difficulty-specific average
        difficulty_scores = st.session_state.user_progress[scores_key]
        if difficulty_scores:
            st.session_state.user_progress[avg_key] = sum(difficulty_scores) / len(difficulty_scores)
        
        # Check for perfect score
        if score >= 100:
            st.session_state.user_progress['perfect_quizzes'] += 1
            self.award_points('perfect_quiz', difficulty_level)
            self._show_perfect_score_celebration()
        else:
            self.award_points('quiz_completed', difficulty_level)
        
        # Check achievements
        self.check_achievements()
    
    def update_learning_streak(self):
        """Update learning streak with date tracking"""
        
        today = datetime.now().date()
        last_activity = st.session_state.user_progress.get('last_activity_date')
        
        if last_activity:
            if isinstance(last_activity, str):
                last_activity = datetime.fromisoformat(last_activity).date()
            
            days_diff = (today - last_activity).days
            
            if days_diff == 1:
                # Continue streak
                st.session_state.user_progress['learning_streak'] += 1
                self.award_points('streak_day')
            elif days_diff == 0:
                # Same day, no change
                pass
            else:
                # Streak broken
                if st.session_state.user_progress['learning_streak'] > 0:
                    self._show_streak_broken_notification(st.session_state.user_progress['learning_streak'])
                st.session_state.user_progress['learning_streak'] = 1
        else:
            # First day
            st.session_state.user_progress['learning_streak'] = 1
        
        # Update last activity date
        st.session_state.user_progress['last_activity_date'] = today.isoformat()
        
        # Update longest streak
        current_streak = st.session_state.user_progress['learning_streak']
        if current_streak > st.session_state.user_progress.get('longest_streak', 0):
            st.session_state.user_progress['longest_streak'] = current_streak
        
        # Check for streak achievements
        self.check_achievements()
        
        # Show streak milestone notifications
        if current_streak in [3, 7, 14, 30, 50, 100]:
            self._show_streak_milestone(current_streak)
    
    def check_achievements(self):
        """Check and award new achievements"""
        
        new_achievements = []
        progress = st.session_state.user_progress
        
        for achievement_id, achievement in self.achievements.items():
            if achievement_id not in progress['badges_earned']:
                if self._check_achievement_requirements(achievement, progress):
                    # Award achievement
                    progress['badges_earned'].append(achievement_id)
                    progress['total_points'] += achievement['points']
                    
                    # Record achievement history
                    achievement_record = {
                        'badge_id': achievement_id,
                        'name': achievement['name'],
                        'date': datetime.now().isoformat(),
                        'points_earned': achievement['points']
                    }
                    progress['achievements_history'].append(achievement_record)
                    
                    new_achievements.append(achievement)
        
        # Show achievement notifications
        for achievement in new_achievements:
            self._show_achievement_notification(achievement)
        
        return new_achievements
    
    def _check_achievement_requirements(self, achievement, progress):
        """Check if achievement requirements are met"""
        
        requirements = achievement['requirements']
        
        for req_key, req_value in requirements.items():
            if req_key not in progress:
                return False
            
            progress_value = progress[req_key]
            
            # Handle different requirement types
            if req_key.endswith('_average') or req_key.endswith('_avg_score'):
                if progress_value < req_value:
                    return False
            else:
                if progress_value < req_value:
                    return False
        
        return True
    
    def _update_user_level(self):
        """Update user level based on points"""
        
        current_points = st.session_state.user_progress['total_points']
        
        for level in reversed(self.levels):
            if current_points >= level['min_points']:
                st.session_state.user_progress['current_level'] = level['level']
                break
    
    def get_current_level_info(self):
        """Get current level information"""
        
        current_level_num = st.session_state.user_progress.get('current_level', 1)
        current_points = st.session_state.user_progress.get('total_points', 0)
        
        current_level = next((l for l in self.levels if l['level'] == current_level_num), self.levels[0])
        next_level = next((l for l in self.levels if l['level'] == current_level_num + 1), None)
        
        if next_level:
            points_to_next = next_level['min_points'] - current_points
            progress_to_next = (current_points - current_level['min_points']) / (next_level['min_points'] - current_level['min_points'])
        else:
            points_to_next = 0
            progress_to_next = 1.0
        
        return {
            'current_level': current_level,
            'next_level': next_level,
            'current_points': current_points,
            'points_to_next': max(0, points_to_next),
            'progress_to_next': min(1.0, progress_to_next)
        }
    
    def display_enhanced_progress_dashboard(self):
        """Display comprehensive progress dashboard"""
        
        st.subheader("🏆 Your Learning Journey")
        
        # Level and progress display
        level_info = self.get_current_level_info()
        current_level = level_info['current_level']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style='text-align: center; padding: 20px; background: {current_level['color']}20; border-radius: 10px;'>
                <h2>{current_level['icon']}</h2>
                <h3>{current_level['name']}</h3>
                <p>Level {current_level['level']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("📊 Total Points", f"{level_info['current_points']:,}")
            if level_info['next_level']:
                st.caption(f"Next level: {level_info['points_to_next']} points")
        
        with col3:
            streak = st.session_state.user_progress.get('learning_streak', 0)
            st.metric("🔥 Learning Streak", f"{streak} days")
            longest = st.session_state.user_progress.get('longest_streak', 0)
            st.caption(f"Longest: {longest} days")
        
        with col4:
            badges_count = len(st.session_state.user_progress.get('badges_earned', []))
            st.metric("🏅 Achievements", badges_count)
            st.caption(f"of {len(self.achievements)} total")
        
        # Progress bar to next level
        if level_info['next_level']:
            st.markdown("### 📈 Level Progress")
            progress = level_info['progress_to_next']
            st.progress(progress)
            st.caption(f"Progress to {level_info['next_level']['name']}: {progress:.1%}")
        
        # Achievement showcase
        self._display_achievement_showcase()
        
        # Performance analytics by difficulty
        self._display_difficulty_analytics()
    
    def _display_achievement_showcase(self):
        """Display earned achievements organized by category"""
        
        st.markdown("### 🏅 Achievement Gallery")
        
        earned_badges = st.session_state.user_progress.get('badges_earned', [])
        
        if not earned_badges:
            st.info("Complete learning activities to earn your first achievements!")
            return
        
        # Group achievements by category
        categorized_achievements = {}
        for badge_id in earned_badges:
            if badge_id in self.achievements:
                achievement = self.achievements[badge_id]
                category = achievement.get('category', 'other')
                if category not in categorized_achievements:
                    categorized_achievements[category] = []
                categorized_achievements[category].append(achievement)
        
        # Display by category
        for category, achievements in categorized_achievements.items():
            category_info = self.badge_categories.get(category, {'name': category.title(), 'icon': '🏆'})
            
            with st.expander(f"{category_info['icon']} {category_info['name']} ({len(achievements)})"):
                cols = st.columns(min(3, len(achievements)))
                
                for i, achievement in enumerate(achievements):
                    with cols[i % 3]:
                        st.markdown(f"""
                        <div style='text-align: center; padding: 10px; background: #f0f2f6; border-radius: 8px; margin: 5px;'>
                            <h3>{achievement['icon']}</h3>
                            <h4>{achievement['name']}</h4>
                            <p style='font-size: 0.8em;'>{achievement['description']}</p>
                            <p style='font-weight: bold; color: #1f77b4;'>+{achievement['points']} points</p>
                        </div>
                        """, unsafe_allow_html=True)
    
    def _display_difficulty_analytics(self):
        """Display performance analytics by difficulty level"""
        
        st.markdown("### 📊 Performance by Difficulty Level")
        
        progress = st.session_state.user_progress
        
        difficulty_data = []
        for difficulty in ['beginner', 'intermediate', 'advanced']:
            quizzes_key = f'{difficulty}_quizzes'
            avg_key = f'{difficulty}_avg_score'
            
            quizzes_taken = progress.get(quizzes_key, 0)
            avg_score = progress.get(avg_key, 0)
            
            difficulty_data.append({
                'difficulty': difficulty.title(),
                'quizzes': quizzes_taken,
                'average': round(avg_score, 1) if avg_score > 0 else 0,
                'color': {'beginner': '#4CAF50', 'intermediate': '#FF9800', 'advanced': '#F44336'}[difficulty]
            })
        
        if any(d['quizzes'] > 0 for d in difficulty_data):
            col1, col2, col3 = st.columns(3)
            
            for i, data in enumerate(difficulty_data):
                with [col1, col2, col3][i]:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 15px; background: {data['color']}20; border-radius: 10px;'>
                        <h4>{data['difficulty']}</h4>
                        <p><strong>{data['quizzes']}</strong> quizzes</p>
                        <p><strong>{data['average']}%</strong> average</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Take quizzes at different difficulty levels to see your performance analytics!")
    
    def _show_points_notification(self, points, action_type, difficulty_level):
        """Show points earned notification"""
        
        difficulty_bonus = ""
        if difficulty_level != "intermediate":
            difficulty_bonus = f" ({difficulty_level} bonus!)"
        
        st.success(f"🎉 +{points} points earned for {action_type.replace('_', ' ')}{difficulty_bonus}")
    
    def _show_level_up_notification(self, new_level):
        """Show level up celebration"""
        
        level_info = next((l for l in self.levels if l['level'] == new_level), None)
        if level_info:
            st.balloons()
            st.success(f"🎉 **LEVEL UP!** You are now a {level_info['name']} (Level {new_level}) {level_info['icon']}")
    
    def _show_achievement_notification(self, achievement):
        """Show achievement earned notification"""
        
        st.balloons()
        st.success(f"🏆 **ACHIEVEMENT UNLOCKED!** {achievement['icon']} {achievement['name']}")
        st.info(f"📝 {achievement['description']} (+{achievement['points']} points)")
    
    def _show_perfect_score_celebration(self):
        """Show perfect score celebration"""
        
        st.balloons()
        st.success("🎉 **PERFECT SCORE!** 💯 Outstanding performance!")
    
    def _show_streak_milestone(self, streak_days):
        """Show streak milestone celebration"""
        
        milestones = {
            3: "🔥 3-day streak! You're on fire!",
            7: "⭐ Week warrior! 7 days strong!",
            14: "🌟 Two weeks of excellence!",
            30: "🏆 Monthly master! 30 days achieved!",
            50: "💎 50-day legend! Incredible dedication!",
            100: "👑 Century streak! You're unstoppable!"
        }
        
        message = milestones.get(streak_days, f"🔥 {streak_days}-day streak!")
        st.success(message)
    
    def _show_streak_broken_notification(self, broken_streak):
        """Show streak broken notification with encouragement"""
        
        if broken_streak >= 7:
            st.warning(f"💔 Your {broken_streak}-day streak ended, but don't give up! Start a new one today! 💪")
        else:
            st.info(f"Your {broken_streak}-day streak ended. Ready to start fresh? 🌟")
    
    def get_achievement_progress(self):
        """Get progress toward unearned achievements"""
        
        progress = st.session_state.user_progress
        earned_badges = set(progress.get('badges_earned', []))
        
        achievement_progress = []
        
        for achievement_id, achievement in self.achievements.items():
            if achievement_id not in earned_badges:
                requirements = achievement['requirements']
                progress_info = {
                    'achievement': achievement,
                    'progress': {},
                    'completion': 0.0
                }
                
                total_requirements = len(requirements)
                completed_requirements = 0
                
                for req_key, req_value in requirements.items():
                    current_value = progress.get(req_key, 0)
                    progress_info['progress'][req_key] = {
                        'current': current_value,
                        'required': req_value,
                        'percentage': min(100, (current_value / req_value) * 100) if req_value > 0 else 0
                    }
                    
                    if current_value >= req_value:
                        completed_requirements += 1
                
                progress_info['completion'] = (completed_requirements / total_requirements) * 100
                achievement_progress.append(progress_info)
        
        # Sort by completion percentage
        achievement_progress.sort(key=lambda x: x['completion'], reverse=True)
        
        return achievement_progress[:10]  # Return top 10 closest achievements
    
    def display_achievement_progress(self):
        """Display progress toward unearned achievements"""
        
        st.markdown("### 🎯 Achievement Progress")
        
        progress_list = self.get_achievement_progress()
        
        if not progress_list:
            st.success("🎉 Congratulations! You've earned all available achievements!")
            return
        
        for progress_info in progress_list[:5]:  # Show top 5
            achievement = progress_info['achievement']
            completion = progress_info['completion']
            
            with st.expander(f"{achievement['icon']} {achievement['name']} - {completion:.0f}% complete"):
                st.write(achievement['description'])
                
                for req_key, req_progress in progress_info['progress'].items():
                    current = req_progress['current']
                    required = req_progress['required']
                    percentage = req_progress['percentage']
                    
                    st.write(f"**{req_key.replace('_', ' ').title()}:** {current}/{required}")
                    st.progress(percentage / 100)
                
                st.caption(f"Reward: +{achievement['points']} points")
    
    def export_gamification_data(self):
        """Export gamification data for backup/analysis"""
        
        export_data = {
            'user_progress': st.session_state.user_progress,
            'export_timestamp': datetime.now().isoformat(),
            'achievements_earned': len(st.session_state.user_progress.get('badges_earned', [])),
            'total_achievements': len(self.achievements),
            'level_info': self.get_current_level_info(),
            'achievement_progress': self.get_achievement_progress()
        }
        
        return json.dumps(export_data, indent=2, default=str)

# Global gamification engine instance
enhanced_gamification = EnhancedGamificationEngine()
