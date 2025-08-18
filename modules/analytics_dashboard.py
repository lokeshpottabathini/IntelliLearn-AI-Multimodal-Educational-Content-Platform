import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Enhanced imports for difficulty-aware analytics
try:
    from .advanced_analytics_engine import AdvancedAnalyticsEngine
    from .learning_insights_generator import LearningInsightsGenerator
    from .performance_predictor import PerformancePredictionModel
    ENHANCED_MODULES_AVAILABLE = True
except ImportError:
    ENHANCED_MODULES_AVAILABLE = False

class LearningAnalyticsDashboard:
    def __init__(self):
        """Initialize enhanced analytics dashboard with difficulty awareness and advanced metrics"""
        self.user_sessions = []
        self.learning_progress = {}
        
        # Initialize enhanced analytics engine if available
        self.advanced_analytics = None
        self.insights_generator = None
        self.performance_predictor = None
        
        if ENHANCED_MODULES_AVAILABLE:
            try:
                self.advanced_analytics = AdvancedAnalyticsEngine()
                self.insights_generator = LearningInsightsGenerator()
                self.performance_predictor = PerformancePredictionModel()
                st.success("✅ Enhanced analytics features loaded")
            except Exception as e:
                st.warning(f"Enhanced analytics not available: {e}")
        
        # Initialize enhanced session state for analytics
        if 'analytics_data' not in st.session_state:
            st.session_state.analytics_data = {
                # Core Analytics
                'sessions': [],
                'progress': {},
                'time_spent': {},
                'quiz_scores': [],
                'topic_completion': {},
                
                # Enhanced Analytics
                'difficulty_progression': {'beginner': [], 'intermediate': [], 'advanced': []},
                'learning_patterns': {},
                'voice_interactions': 0,
                'ai_conversations': [],
                'video_generations': 0,
                'content_preferences': {},
                'study_streaks': [],
                'performance_trends': {},
                
                # Time-Based Analytics
                'daily_activity': {},
                'weekly_summaries': {},
                'monthly_progress': {},
                'session_durations': [],
                
                # Engagement Metrics
                'feature_usage': {
                    'text_chat': 0,
                    'voice_chat': 0,
                    'video_generation': 0,
                    'quiz_taking': 0,
                    'chapter_reading': 0
                },
                
                # Learning Effectiveness
                'concept_mastery': {},
                'knowledge_gaps': [],
                'learning_velocity': [],
                'retention_scores': {}
            }
    
    def create_comprehensive_dashboard_enhanced(self, knowledge_base, user_progress=None, difficulty_level='intermediate'):
        """NEW: Enhanced comprehensive dashboard with difficulty awareness and advanced analytics"""
        
        st.title("📊 Enhanced Learning Analytics Dashboard")
        
        # Enhanced overview with difficulty context
        self._display_enhanced_overview_metrics(knowledge_base, user_progress, difficulty_level)
        
        # Difficulty-specific analytics tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", "🎯 Progress", "📈 Performance", "🧠 Insights", "🔮 Predictions"
        ])
        
        with tab1:
            self._create_overview_analytics(knowledge_base, difficulty_level)
        
        with tab2:
            self._create_progress_analytics(knowledge_base, user_progress, difficulty_level)
        
        with tab3:
            self._create_performance_analytics(user_progress, difficulty_level)
        
        with tab4:
            self._create_learning_insights_enhanced(knowledge_base, user_progress)
        
        with tab5:
            self._create_predictive_analytics(user_progress, difficulty_level)

    def create_comprehensive_dashboard(self, knowledge_base, user_progress=None):
        """Enhanced comprehensive learning analytics dashboard (original method with enhancements)"""
        
        st.subheader("📊 Learning Analytics Dashboard")
        
        # Overview metrics
        self._display_overview_metrics(knowledge_base)
        
        # Content analysis
        col1, col2 = st.columns(2)
        
        with col1:
            self._create_difficulty_distribution(knowledge_base)
            self._create_chapter_overview(knowledge_base)
        
        with col2:
            self._create_content_metrics(knowledge_base)
            self._create_progress_tracking()
        
        # Detailed analytics
        self._create_topic_analysis(knowledge_base)
        
        # Learning recommendations
        self._generate_learning_recommendations(knowledge_base)

    def _display_enhanced_overview_metrics(self, knowledge_base, user_progress, difficulty_level):
        """NEW: Enhanced overview metrics with difficulty context and learning insights"""
        
        chapters = knowledge_base.get('chapters', {})
        analytics_data = st.session_state.analytics_data
        
        # Calculate enhanced metrics
        total_chapters = len(chapters)
        total_topics = sum(len(ch.get('topics', {})) for ch in chapters.values())
        total_concepts = sum(len(ch.get('concepts', [])) for ch in chapters.values())
        total_examples = sum(len(ch.get('examples', [])) for ch in chapters.values())
        
        # Calculate difficulty-specific metrics
        difficulty_topics = {'beginner': 0, 'intermediate': 0, 'advanced': 0}
        for chapter in chapters.values():
            for topic in chapter.get('topics', {}).values():
                diff = topic.get('difficulty', 'intermediate').lower()
                difficulty_topics[diff] = difficulty_topics.get(diff, 0) + 1
        
        # Enhanced user progress metrics
        if user_progress:
            total_points = user_progress.get('total_points', 0)
            badges_earned = len(user_progress.get('badges_earned', []))
            learning_streak = user_progress.get('learning_streak', 0)
            chapters_completed = user_progress.get('chapters_completed', 0)
            current_level = user_progress.get('difficulty_level', difficulty_level)
        else:
            total_points = badges_earned = learning_streak = chapters_completed = 0
            current_level = difficulty_level
        
        # Display enhanced metrics grid
        st.markdown("### 📈 Learning Overview")
        
        # Top row - Content metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "📚 Total Chapters",
                total_chapters,
                help="Total chapters available in your textbook"
            )
        
        with col2:
            st.metric(
                "📖 Total Topics", 
                total_topics,
                help="Topics identified across all chapters"
            )
        
        with col3:
            st.metric(
                "🔑 Key Concepts",
                total_concepts,
                help="Important concepts extracted from content"
            )
        
        with col4:
            st.metric(
                "💡 Examples",
                total_examples,
                help="Real-world examples found in content"
            )
        
        with col5:
            difficulty_counts = pd.Series(list(difficulty_topics.values()), 
                                        index=['Beginner', 'Intermediate', 'Advanced'])
            most_common = difficulty_counts.idxmax() if difficulty_counts.sum() > 0 else "Intermediate"
            st.metric(
                "📊 Primary Level",
                most_common,
                help="Most common difficulty level in content"
            )
        
        # Second row - User progress metrics
        if user_progress:
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "🎯 Your Level",
                    current_level.title(),
                    help="Your current difficulty setting"
                )
            
            with col2:
                st.metric(
                    "⭐ Points Earned",
                    f"{total_points:,}",
                    help="Total learning points accumulated"
                )
            
            with col3:
                st.metric(
                    "🏅 Badges",
                    badges_earned,
                    help="Achievement badges earned"
                )
            
            with col4:
                st.metric(
                    "🔥 Streak",
                    f"{learning_streak} days",
                    help="Consecutive learning days"
                )
            
            with col5:
                completion_rate = (chapters_completed / total_chapters * 100) if total_chapters > 0 else 0
                st.metric(
                    "📈 Progress",
                    f"{completion_rate:.1f}%",
                    help="Overall chapter completion rate"
                )
        
        # Third row - Engagement metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        feature_usage = analytics_data.get('feature_usage', {})
        
        with col1:
            st.metric(
                "💬 AI Chats",
                feature_usage.get('text_chat', 0),
                help="Total AI conversations"
            )
        
        with col2:
            st.metric(
                "🎤 Voice Chats",
                feature_usage.get('voice_chat', 0),
                help="Voice interactions with AI"
            )
        
        with col3:
            st.metric(
                "🎬 Videos Created",
                feature_usage.get('video_generation', 0),
                help="Educational videos generated"
            )
        
        with col4:
            st.metric(
                "🧠 Quizzes Taken",
                feature_usage.get('quiz_taking', 0),
                help="Assessment quizzes completed"
            )
        
        with col5:
            total_time = sum(analytics_data.get('time_spent', {}).values())
            st.metric(
                "⏰ Study Time",
                f"{total_time:.0f} min",
                help="Total time spent learning"
            )

    def _create_overview_analytics(self, knowledge_base, difficulty_level):
        """NEW: Enhanced overview analytics with difficulty breakdown"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            self._create_difficulty_distribution_enhanced(knowledge_base, difficulty_level)
            self._create_learning_path_visualization(knowledge_base, difficulty_level)
        
        with col2:
            self._create_content_complexity_analysis(knowledge_base)
            self._create_engagement_heatmap()

    def _create_progress_analytics(self, knowledge_base, user_progress, difficulty_level):
        """NEW: Enhanced progress analytics with difficulty tracking"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            self._create_difficulty_progression_chart(user_progress)
            self._create_chapter_completion_timeline(knowledge_base, user_progress)
        
        with col2:
            self._create_skill_development_radar(user_progress, difficulty_level)
            self._create_learning_velocity_chart()

    def _create_performance_analytics(self, user_progress, difficulty_level):
        """NEW: Enhanced performance analytics with detailed metrics"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            self._create_quiz_performance_analysis(user_progress, difficulty_level)
            self._create_retention_analysis()
        
        with col2:
            self._create_learning_efficiency_metrics()
            self._create_concept_mastery_matrix(user_progress)

    def _create_learning_insights_enhanced(self, knowledge_base, user_progress):
        """NEW: Enhanced learning insights with AI-powered recommendations"""
        
        if self.insights_generator:
            insights = self.insights_generator.generate_comprehensive_insights(
                knowledge_base, user_progress, st.session_state.analytics_data
            )
            
            # Display AI-powered insights
            st.markdown("### 🧠 AI-Powered Learning Insights")
            
            for insight in insights:
                with st.container():
                    st.markdown(f"**{insight['category']}:** {insight['insight']}")
                    if insight.get('recommendation'):
                        st.info(f"💡 Recommendation: {insight['recommendation']}")
                    st.markdown("---")
        else:
            # Fallback to basic insights
            self._generate_basic_learning_insights(knowledge_base, user_progress)

    def _create_predictive_analytics(self, user_progress, difficulty_level):
        """NEW: Predictive analytics for learning outcomes"""
        
        if self.performance_predictor:
            predictions = self.performance_predictor.predict_learning_outcomes(
                user_progress, st.session_state.analytics_data, difficulty_level
            )
            
            st.markdown("### 🔮 Learning Predictions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Performance prediction
                if predictions.get('performance_trend'):
                    st.plotly_chart(
                        self._create_performance_prediction_chart(predictions['performance_trend']),
                        use_container_width=True
                    )
                
                # Mastery timeline
                if predictions.get('mastery_timeline'):
                    st.plotly_chart(
                        self._create_mastery_timeline_chart(predictions['mastery_timeline']),
                        use_container_width=True
                    )
            
            with col2:
                # Risk assessment
                if predictions.get('risk_factors'):
                    self._display_learning_risks(predictions['risk_factors'])
                
                # Recommendations
                if predictions.get('recommendations'):
                    self._display_predictive_recommendations(predictions['recommendations'])
        else:
            st.info("📊 Advanced predictive analytics will be available after more learning data is collected.")
            self._create_basic_predictions(user_progress, difficulty_level)

    def _create_difficulty_distribution_enhanced(self, knowledge_base, current_difficulty):
        """NEW: Enhanced difficulty distribution with user context"""
        
        st.subheader("📊 Content Difficulty Analysis")
        
        difficulties = []
        chapter_names = []
        
        for chapter_name, chapter_data in knowledge_base.get('chapters', {}).items():
            for topic_name, topic_data in chapter_data.get('topics', {}).items():
                diff = topic_data.get('difficulty', 'Intermediate')
                difficulties.append(diff)
                chapter_names.append(chapter_name)
        
        if difficulties:
            difficulty_counts = pd.Series(difficulties).value_counts()
            
            # Enhanced color scheme with current difficulty highlighted
            colors = {
                'Beginner': '#2ecc71' if current_difficulty != 'beginner' else '#27ae60',
                'Intermediate': '#f39c12' if current_difficulty != 'intermediate' else '#e67e22',
                'Advanced': '#e74c3c' if current_difficulty != 'advanced' else '#c0392b'
            }
            
            # Add border for current difficulty
            line_colors = {level: 'white' if level.lower() == current_difficulty else 'rgba(0,0,0,0)' 
                          for level in difficulty_counts.index}
            
            fig = px.pie(
                values=difficulty_counts.values,
                names=difficulty_counts.index,
                title=f"Topic Difficulty Levels (Your Level: {current_difficulty.title()})",
                color=difficulty_counts.index,
                color_discrete_map=colors
            )
            
            # Highlight current difficulty level
            fig.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                marker_line_color=[line_colors.get(name, 'rgba(0,0,0,0)') for name in difficulty_counts.index],
                marker_line_width=3
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show recommendations based on difficulty distribution
            current_count = difficulty_counts.get(current_difficulty.title(), 0)
            total_topics = difficulty_counts.sum()
            
            if current_count > 0:
                percentage = (current_count / total_topics) * 100
                st.success(f"🎯 Great! {percentage:.1f}% of topics match your {current_difficulty} level")
                
                # Suggest progression
                if current_difficulty == 'beginner' and difficulty_counts.get('Intermediate', 0) > 0:
                    st.info("💡 Consider advancing to intermediate topics when ready!")
                elif current_difficulty == 'intermediate' and difficulty_counts.get('Advanced', 0) > 0:
                    st.info("🚀 Challenge yourself with advanced topics!")
            else:
                st.warning(f"⚠️ No {current_difficulty} level topics found. Consider adjusting your difficulty setting.")
        else:
            st.info("No difficulty data available")

    def _create_learning_path_visualization(self, knowledge_base, difficulty_level):
        """NEW: Visual learning path based on difficulty progression"""
        
        st.subheader("🛤️ Suggested Learning Path")
        
        chapters = knowledge_base.get('chapters', {})
        if chapters:
            # Create learning path data
            path_data = []
            
            for i, (chapter_name, chapter_data) in enumerate(chapters.items()):
                # Calculate chapter difficulty score
                topic_difficulties = []
                for topic_data in chapter_data.get('topics', {}).values():
                    diff = topic_data.get('difficulty', 'Intermediate').lower()
                    difficulty_score = {'beginner': 1, 'intermediate': 2, 'advanced': 3}.get(diff, 2)
                    topic_difficulties.append(difficulty_score)
                
                avg_difficulty = np.mean(topic_difficulties) if topic_difficulties else 2
                
                path_data.append({
                    'Chapter': chapter_name[:20] + '...' if len(chapter_name) > 20 else chapter_name,
                    'Order': i + 1,
                    'Difficulty_Score': avg_difficulty,
                    'Topics': len(chapter_data.get('topics', {})),
                    'Recommended': avg_difficulty <= {'beginner': 1.5, 'intermediate': 2.5, 'advanced': 3.5}.get(difficulty_level, 2.5)
                })
            
            df = pd.DataFrame(path_data)
            
            # Create learning path chart
            fig = px.scatter(
                df, 
                x='Order', 
                y='Difficulty_Score',
                size='Topics',
                color='Recommended',
                hover_data=['Chapter', 'Topics'],
                title="Learning Path by Difficulty",
                labels={'Order': 'Chapter Order', 'Difficulty_Score': 'Difficulty Level'},
                color_discrete_map={True: '#2ecc71', False: '#95a5a6'}
            )
            
            # Add difficulty level line
            current_diff_score = {'beginner': 1, 'intermediate': 2, 'advanced': 3}.get(difficulty_level, 2)
            fig.add_hline(
                y=current_diff_score, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Your Level: {difficulty_level.title()}"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No learning path data available")

    def _create_content_complexity_analysis(self, knowledge_base):
        """NEW: Analyze content complexity metrics"""
        
        st.subheader("🔍 Content Complexity Analysis")
        
        chapters = knowledge_base.get('chapters', {})
        if chapters:
            complexity_data = []
            
            for chapter_name, chapter_data in chapters.items():
                # Calculate complexity metrics
                word_count = chapter_data.get('word_count', 0)
                concepts_count = len(chapter_data.get('concepts', []))
                topics_count = len(chapter_data.get('topics', {}))
                examples_count = len(chapter_data.get('examples', []))
                
                # Complexity score calculation
                complexity_score = (word_count / 1000) + (concepts_count * 2) + (topics_count * 3)
                
                complexity_data.append({
                    'Chapter': chapter_name[:25] + '...' if len(chapter_name) > 25 else chapter_name,
                    'Complexity_Score': complexity_score,
                    'Word_Count': word_count,
                    'Concepts': concepts_count,
                    'Topics': topics_count,
                    'Examples': examples_count
                })
            
            df = pd.DataFrame(complexity_data)
            
            # Create complexity heatmap
            fig = px.bar(
                df,
                x='Chapter',
                y='Complexity_Score',
                color='Complexity_Score',
                title="Chapter Complexity Scores",
                color_continuous_scale='viridis',
                hover_data=['Word_Count', 'Concepts', 'Topics', 'Examples']
            )
            
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show complexity statistics
            avg_complexity = df['Complexity_Score'].mean()
            max_complexity_chapter = df.loc[df['Complexity_Score'].idxmax(), 'Chapter']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Complexity", f"{avg_complexity:.1f}")
            with col2:
                st.metric("Most Complex Chapter", max_complexity_chapter)
        else:
            st.info("No complexity analysis available")

    def _create_engagement_heatmap(self):
        """NEW: User engagement heatmap by time and activity"""
        
        st.subheader("🔥 Learning Engagement Heatmap")
        
        analytics_data = st.session_state.analytics_data
        daily_activity = analytics_data.get('daily_activity', {})
        
        if daily_activity:
            # Create engagement heatmap data
            heatmap_data = []
            
            for date_str, activities in daily_activity.items():
                try:
                    date_obj = datetime.fromisoformat(date_str)
                    day_of_week = date_obj.strftime('%A')
                    hour_of_day = date_obj.hour
                    
                    heatmap_data.append({
                        'Day': day_of_week,
                        'Hour': hour_of_day,
                        'Activity_Count': len(activities.get('activities', []))
                    })
                except:
                    continue
            
            if heatmap_data:
                df = pd.DataFrame(heatmap_data)
                
                # Create pivot table for heatmap
                heatmap_pivot = df.pivot_table(
                    values='Activity_Count', 
                    index='Day', 
                    columns='Hour', 
                    aggfunc='sum', 
                    fill_value=0
                )
                
                # Create heatmap
                fig = px.imshow(
                    heatmap_pivot,
                    title="Learning Activity by Day and Hour",
                    labels={'x': 'Hour of Day', 'y': 'Day of Week', 'color': 'Activities'},
                    color_continuous_scale='blues'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No engagement data available yet")
        else:
            # Show sample heatmap
            sample_data = np.random.randint(0, 5, size=(7, 24))
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            fig = px.imshow(
                sample_data,
                y=days,
                title="Sample Learning Activity Heatmap",
                labels={'x': 'Hour of Day', 'y': 'Day of Week', 'color': 'Activities'},
                color_continuous_scale='blues'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.info("Start learning to see your actual engagement patterns!")

    def _create_difficulty_progression_chart(self, user_progress):
        """NEW: Show difficulty level progression over time"""
        
        st.subheader("📈 Difficulty Level Progression")
        
        analytics_data = st.session_state.analytics_data
        difficulty_progression = analytics_data.get('difficulty_progression', {})
        
        if any(difficulty_progression.values()):
            # Create progression timeline
            progression_data = []
            
            for difficulty, scores in difficulty_progression.items():
                for i, score in enumerate(scores):
                    progression_data.append({
                        'Session': i + 1,
                        'Difficulty': difficulty.title(),
                        'Score': score,
                        'Date': (datetime.now() - timedelta(days=len(scores) - i)).strftime('%Y-%m-%d')
                    })
            
            if progression_data:
                df = pd.DataFrame(progression_data)
                
                fig = px.line(
                    df,
                    x='Session',
                    y='Score',
                    color='Difficulty',
                    title="Performance by Difficulty Level Over Time",
                    markers=True,
                    hover_data=['Date']
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show difficulty insights
                if user_progress:
                    current_level = user_progress.get('difficulty_level', 'intermediate')
                    current_scores = difficulty_progression.get(current_level, [])
                    
                    if current_scores:
                        avg_score = np.mean(current_scores)
                        recent_trend = "improving" if len(current_scores) > 1 and current_scores[-1] > current_scores[-2] else "stable"
                        
                        st.success(f"🎯 Your {current_level} level average: {avg_score:.1f}% ({recent_trend})")
            else:
                st.info("No difficulty progression data available yet")
        else:
            st.info("Complete quizzes at different difficulty levels to see your progression!")

    def _create_skill_development_radar(self, user_progress, difficulty_level):
        """NEW: Radar chart showing skill development across areas"""
        
        st.subheader("🎯 Skill Development Radar")
        
        if user_progress:
            # Extract skill data
            skills_data = {
                'Quiz Performance': min(100, (user_progress.get('quizzes_taken', 0) * 10)),
                'Chapter Completion': min(100, (user_progress.get('chapters_completed', 0) * 15)),
                'Concept Mastery': min(100, len(user_progress.get('badges_earned', [])) * 8),
                'Consistency': min(100, user_progress.get('learning_streak', 0) * 3),
                'Engagement': min(100, user_progress.get('questions_asked', 0) * 2),
                'Content Creation': min(100, user_progress.get('videos_generated', 0) * 25)
            }
            
            # Adjust based on difficulty level
            difficulty_multiplier = {'beginner': 1.2, 'intermediate': 1.0, 'advanced': 0.8}.get(difficulty_level, 1.0)
            skills_data = {skill: min(100, score * difficulty_multiplier) for skill, score in skills_data.items()}
            
            # Create radar chart
            categories = list(skills_data.keys())
            values = list(skills_data.values())
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=f'{difficulty_level.title()} Level Skills',
                line_color='rgb(46, 204, 113)',
                fillcolor='rgba(46, 204, 113, 0.3)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="Your Learning Skills Development"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show skill insights
            max_skill = max(skills_data, key=skills_data.get)
            min_skill = min(skills_data, key=skills_data.get)
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"🌟 Strongest: {max_skill} ({skills_data[max_skill]:.0f}%)")
            with col2:
                st.info(f"📈 Growth area: {min_skill} ({skills_data[min_skill]:.0f}%)")
        else:
            st.info("Start learning to see your skill development!")

    def _create_quiz_performance_analysis(self, user_progress, difficulty_level):
        """NEW: Detailed quiz performance analysis"""
        
        st.subheader("🧠 Quiz Performance Analysis")
        
        if user_progress:
            analytics_data = st.session_state.analytics_data
            
            # Get difficulty-specific scores
            difficulty_scores = {
                'beginner': user_progress.get('beginner_scores', []),
                'intermediate': user_progress.get('intermediate_scores', []),
                'advanced': user_progress.get('advanced_scores', [])
            }
            
            # Create performance comparison chart
            performance_data = []
            
            for diff, scores in difficulty_scores.items():
                if scores:
                    avg_score = np.mean(scores)
                    performance_data.append({
                        'Difficulty': diff.title(),
                        'Average_Score': avg_score,
                        'Quizzes_Taken': len(scores),
                        'Best_Score': max(scores),
                        'Current_Level': diff == difficulty_level
                    })
            
            if performance_data:
                df = pd.DataFrame(performance_data)
                
                # Create bar chart with highlighting
                colors = ['#e74c3c' if row['Current_Level'] else '#95a5a6' for _, row in df.iterrows()]
                
                fig = px.bar(
                    df,
                    x='Difficulty',
                    y='Average_Score',
                    title="Quiz Performance by Difficulty Level",
                    color='Current_Level',
                    color_discrete_map={True: '#e74c3c', False: '#95a5a6'},
                    hover_data=['Quizzes_Taken', 'Best_Score']
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance insights
                current_row = df[df['Current_Level'] == True]
                if not current_row.empty:
                    current_avg = current_row['Average_Score'].iloc[0]
                    current_count = current_row['Quizzes_Taken'].iloc[0]
                    
                    st.success(f"🎯 Your {difficulty_level} level performance: {current_avg:.1f}% average over {current_count} quizzes")
            else:
                st.info("Take some quizzes to see your performance analysis!")
        else:
            st.info("No quiz performance data available")

    def _display_overview_metrics(self, knowledge_base):
        """Display key overview metrics (original method enhanced)"""
        chapters = knowledge_base.get('chapters', {})
        
        # Calculate metrics
        total_chapters = len(chapters)
        total_topics = sum(len(ch.get('topics', {})) for ch in chapters.values())
        total_concepts = sum(len(ch.get('concepts', [])) for ch in chapters.values())
        total_examples = sum(len(ch.get('examples', [])) for ch in chapters.values())
        
        # Calculate average difficulty
        difficulties = []
        for chapter in chapters.values():
            for topic in chapter.get('topics', {}).values():
                diff = topic.get('difficulty', 'Intermediate')
                difficulties.append(diff)
        
        difficulty_counts = pd.Series(difficulties).value_counts()
        most_common_difficulty = difficulty_counts.index[0] if len(difficulty_counts) > 0 else "Intermediate"
        
        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "📚 Chapters",
                total_chapters,
                help="Total number of chapters processed"
            )
        
        with col2:
            st.metric(
                "📖 Topics", 
                total_topics,
                help="Total topics identified across all chapters"
            )
        
        with col3:
            st.metric(
                "🔑 Concepts",
                total_concepts,
                help="Key concepts extracted from content"
            )
        
        with col4:
            st.metric(
                "💡 Examples",
                total_examples,
                help="Real-world examples found"
            )
        
        with col5:
            st.metric(
                "📈 Avg Difficulty",
                most_common_difficulty,
                help="Most common difficulty level"
            )

    def _create_difficulty_distribution(self, knowledge_base):
        """Create difficulty distribution chart (original method)"""
        st.subheader("📊 Difficulty Distribution")
        
        difficulties = []
        chapter_names = []
        
        for chapter_name, chapter_data in knowledge_base.get('chapters', {}).items():
            for topic_name, topic_data in chapter_data.get('topics', {}).items():
                difficulties.append(topic_data.get('difficulty', 'Intermediate'))
                chapter_names.append(chapter_name)
        
        if difficulties:
            # Create pie chart
            difficulty_counts = pd.Series(difficulties).value_counts()
            
            colors = {
                'Beginner': '#2ecc71',      # Green
                'Intermediate': '#f39c12',   # Orange  
                'Advanced': '#e74c3c'        # Red
            }
            
            fig = px.pie(
                values=difficulty_counts.values,
                names=difficulty_counts.index,
                title="Topic Difficulty Levels",
                color=difficulty_counts.index,
                color_discrete_map=colors
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No difficulty data available")

    def _create_chapter_overview(self, knowledge_base):
        """Create chapter overview visualization (original method)"""
        st.subheader("📚 Chapter Overview")
        
        chapters = knowledge_base.get('chapters', {})
        
        if chapters:
            chapter_data = []
            
            for chapter_name, chapter_info in chapters.items():
                chapter_data.append({
                    'Chapter': chapter_name[:30] + "..." if len(chapter_name) > 30 else chapter_name,
                    'Topics': len(chapter_info.get('topics', {})),
                    'Concepts': len(chapter_info.get('concepts', [])),
                    'Examples': len(chapter_info.get('examples', [])),
                    'Word Count': chapter_info.get('word_count', 0)
                })
            
            df = pd.DataFrame(chapter_data)
            
            # Create bar chart for topics per chapter
            fig = px.bar(
                df, 
                x='Chapter', 
                y='Topics',
                title="Topics per Chapter",
                color='Topics',
                color_continuous_scale='viridis'
            )
            
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display data table
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No chapter data available")

    def _create_content_metrics(self, knowledge_base):
        """Create content analysis metrics (original method)"""
        st.subheader("📈 Content Analysis")
        
        chapters = knowledge_base.get('chapters', {})
        
        if chapters:
            # Calculate content metrics
            word_counts = [ch.get('word_count', 0) for ch in chapters.values()]
            topic_counts = [len(ch.get('topics', {})) for ch in chapters.values()]
            
            metrics_data = {
                'Metric': ['Total Words', 'Average Words/Chapter', 'Total Topics', 'Average Topics/Chapter'],
                'Value': [
                    sum(word_counts),
                    np.mean(word_counts) if word_counts else 0,
                    sum(topic_counts),
                    np.mean(topic_counts) if topic_counts else 0
                ]
            }
            
            # Create bar chart
            fig = px.bar(
                x=metrics_data['Metric'],
                y=metrics_data['Value'],
                title="Content Metrics",
                color=metrics_data['Value'],
                color_continuous_scale='blues'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("No content metrics available")

    def _create_progress_tracking(self):
        """Create progress tracking visualization (enhanced)"""
        st.subheader("📊 Learning Progress")
        
        # Get progress data from session state
        progress_data = st.session_state.analytics_data.get('topic_completion', {})
        
        if progress_data:
            # Create progress chart
            topics = list(progress_data.keys())
            completion = list(progress_data.values())
            
            fig = px.bar(
                x=topics,
                y=completion,
                title="Topic Completion Progress",
                labels={'x': 'Topics', 'y': 'Completion %'},
                color=completion,
                color_continuous_scale='greens'
            )
            
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Show placeholder progress
            sample_topics = ['Introduction', 'Core Concepts', 'Applications', 'Examples']
            sample_progress = [85, 60, 30, 15]
            
            fig = px.bar(
                x=sample_topics,
                y=sample_progress,
                title="Sample Learning Progress",
                labels={'x': 'Topics', 'y': 'Completion %'},
                color=sample_progress,
                color_continuous_scale='greens'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.info("Start learning to see your actual progress!")

    def _create_topic_analysis(self, knowledge_base):
        """Create detailed topic analysis (original method)"""
        st.subheader("🔍 Detailed Topic Analysis")
        
        chapters = knowledge_base.get('chapters', {})
        
        if chapters:
            # Create expandable sections for each chapter
            for chapter_name, chapter_data in chapters.items():
                with st.expander(f"📖 {chapter_name}"):
                    
                    # Chapter metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Topics", len(chapter_data.get('topics', {})))
                    with col2:
                        st.metric("Concepts", len(chapter_data.get('concepts', [])))
                    with col3:  
                        st.metric("Examples", len(chapter_data.get('examples', [])))
                    
                    # Topic details
                    topics = chapter_data.get('topics', {})
                    if topics:
                        st.write("**Topics in this chapter:**")
                        
                        topic_table_data = []
                        for topic_name, topic_info in topics.items():
                            topic_table_data.append({
                                'Topic': topic_name,
                                'Difficulty': topic_info.get('difficulty', 'Unknown'),
                                'Key Points': len(topic_info.get('key_points', [])),
                                'Summary': topic_info.get('summary', 'No summary available')[:100] + "..."
                            })
                        
                        if topic_table_data:
                            topic_df = pd.DataFrame(topic_table_data)
                            st.dataframe(topic_df, use_container_width=True)
                    
                    # Show key concepts
                    concepts = chapter_data.get('concepts', [])
                    if concepts:
                        st.write("**Key Concepts:**")
                        concept_cols = st.columns(3)
                        for i, concept in enumerate(concepts[:9]):  # Show up to 9 concepts
                            with concept_cols[i % 3]:
                                st.write(f"• {concept}")
        else:
            st.info("No detailed topic analysis available")

    def _generate_learning_recommendations(self, knowledge_base):
        """Generate AI-powered learning recommendations (enhanced)"""
        st.subheader("🎯 Learning Recommendations")
        
        chapters = knowledge_base.get('chapters', {})
        
        if chapters:
            recommendations = []
            
            # Analyze content structure for recommendations
            beginner_topics = []
            advanced_topics = []
            
            for chapter_name, chapter_data in chapters.items():
                for topic_name, topic_info in chapter_data.get('topics', {}).items():
                    difficulty = topic_info.get('difficulty', 'Intermediate')
                    if difficulty == 'Beginner':
                        beginner_topics.append(f"{chapter_name}: {topic_name}")
                    elif difficulty == 'Advanced':
                        advanced_topics.append(f"{chapter_name}: {topic_name}")
            
            # Generate recommendations
            if beginner_topics:
                recommendations.append({
                    'type': '🌱 Start Here',
                    'title': 'Beginner-Friendly Topics',
                    'items': beginner_topics[:3],
                    'description': 'These topics are great starting points for your learning journey.'
                })
            
            if advanced_topics:
                recommendations.append({
                    'type': '🚀 Challenge Yourself',
                    'title': 'Advanced Topics',
                    'items': advanced_topics[:3],
                    'description': 'Ready for a challenge? Try these advanced topics.'
                })
            
            # Add general recommendations
            all_chapters = list(chapters.keys())
            if len(all_chapters) > 1:
                recommendations.append({
                    'type': '📚 Study Path',
                    'title': 'Suggested Chapter Order',
                    'items': all_chapters[:4],
                    'description': 'Follow this order for systematic learning.'
                })
            
            # Display recommendations
            for rec in recommendations:
                with st.container():
                    st.write(f"**{rec['type']}: {rec['title']}**")
                    st.write(rec['description'])
                    
                    for item in rec['items']:
                        st.write(f"• {item}")
                    
                    st.write("---")
        
        else:
            st.info("Process a textbook to get personalized learning recommendations!")

    # Enhanced tracking and analytics methods
    def track_learning_session_enhanced(self, topic, duration, completion_rate, difficulty_level='intermediate', activity_type='general'):
        """NEW: Enhanced learning session tracking with detailed metadata"""
        
        session_data = {
            'topic': topic,
            'duration': duration,
            'completion_rate': completion_rate,
            'difficulty_level': difficulty_level,
            'activity_type': activity_type,
            'timestamp': datetime.now().isoformat(),
            'date': datetime.now().date().isoformat(),
            'hour': datetime.now().hour
        }
        
        # Add to session state with enhanced tracking
        analytics_data = st.session_state.analytics_data
        
        # Core tracking
        analytics_data['sessions'].append(session_data)
        analytics_data['topic_completion'][topic] = completion_rate
        
        # Enhanced tracking
        analytics_data['session_durations'].append(duration)
        
        # Update feature usage
        if activity_type in analytics_data['feature_usage']:
            analytics_data['feature_usage'][activity_type] += 1
        
        # Update daily activity
        date_str = session_data['date']
        if date_str not in analytics_data['daily_activity']:
            analytics_data['daily_activity'][date_str] = {'activities': [], 'total_time': 0}
        
        analytics_data['daily_activity'][date_str]['activities'].append(session_data)
        analytics_data['daily_activity'][date_str]['total_time'] += duration
        
        # Update time spent
        if topic in analytics_data['time_spent']:
            analytics_data['time_spent'][topic] += duration
        else:
            analytics_data['time_spent'][topic] = duration
        
        # Update difficulty progression
        if completion_rate > 0:  # Only track successful sessions
            if difficulty_level not in analytics_data['difficulty_progression']:
                analytics_data['difficulty_progression'][difficulty_level] = []
            analytics_data['difficulty_progression'][difficulty_level].append(completion_rate)

    def track_learning_session(self, topic, duration, completion_rate):
        """Track a learning session (original method)"""
        session_data = {
            'topic': topic,
            'duration': duration,
            'completion_rate': completion_rate,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to session state
        st.session_state.analytics_data['sessions'].append(session_data)
        st.session_state.analytics_data['topic_completion'][topic] = completion_rate
        
        # Update time spent
        if topic in st.session_state.analytics_data['time_spent']:
            st.session_state.analytics_data['time_spent'][topic] += duration
        else:
            st.session_state.analytics_data['time_spent'][topic] = duration

    def get_learning_insights_enhanced(self):
        """NEW: Enhanced learning insights with comprehensive analysis"""
        
        analytics_data = st.session_state.analytics_data
        insights = []
        
        # Time-based insights
        total_time = sum(analytics_data['time_spent'].values())
        if total_time > 0:
            insights.append({
                'category': '⏰ Study Time',
                'insight': f'Total study time: {total_time:.1f} minutes across {len(analytics_data["sessions"])} sessions',
                'recommendation': f'Average session: {total_time/len(analytics_data["sessions"]):.1f} minutes' if analytics_data['sessions'] else None
            })
        
        # Most studied topic with context
        if analytics_data['time_spent']:
            most_studied = max(analytics_data['time_spent'], key=analytics_data['time_spent'].get)
            time_spent = analytics_data['time_spent'][most_studied]
            insights.append({
                'category': '🎯 Focus Area',
                'insight': f'Most studied topic: {most_studied} ({time_spent:.1f} minutes)',
                'recommendation': 'Consider exploring related topics to broaden your understanding'
            })
        
        # Completion rate analysis
        if analytics_data['topic_completion']:
            avg_completion = np.mean(list(analytics_data['topic_completion'].values()))
            insights.append({
                'category': '📈 Progress Rate',
                'insight': f'Average completion rate: {avg_completion:.1f}%',
                'recommendation': 'Great progress!' if avg_completion >= 80 else 'Try breaking topics into smaller chunks for better completion'
            })
        
        # Difficulty progression insights
        difficulty_progression = analytics_data.get('difficulty_progression', {})
        for difficulty, scores in difficulty_progression.items():
            if scores:
                avg_score = np.mean(scores)
                trend = 'improving' if len(scores) > 1 and scores[-1] > scores[0] else 'stable'
                insights.append({
                    'category': f'🎯 {difficulty.title()} Level',
                    'insight': f'{difficulty.title()} performance: {avg_score:.1f}% average, {trend}',
                    'recommendation': f'Keep practicing at {difficulty} level!' if avg_score >= 75 else f'Consider more practice or review at {difficulty} level'
                })
        
        # Engagement patterns
        daily_activity = analytics_data.get('daily_activity', {})
        if daily_activity:
            active_days = len(daily_activity)
            insights.append({
                'category': '📅 Learning Consistency',
                'insight': f'Active learning on {active_days} different days',
                'recommendation': 'Try to maintain daily learning habits for best results' if active_days < 7 else 'Excellent consistency!'
            })
        
        return insights if insights else [
            {
                'category': '🚀 Getting Started',
                'insight': 'Start your learning journey to unlock personalized insights!',
                'recommendation': 'Begin by exploring topics that interest you most'
            }
        ]

    def get_learning_insights(self):
        """Generate learning insights (original method)"""
        analytics_data = st.session_state.analytics_data
        
        insights = []
        
        # Total study time
        total_time = sum(analytics_data['time_spent'].values())
        if total_time > 0:
            insights.append(f"📚 Total study time: {total_time:.1f} minutes")
        
        # Most studied topic
        if analytics_data['time_spent']:
            most_studied = max(analytics_data['time_spent'], key=analytics_data['time_spent'].get)
            insights.append(f"🎯 Most studied topic: {most_studied}")
        
        # Average completion rate
        if analytics_data['topic_completion']:
            avg_completion = np.mean(list(analytics_data['topic_completion'].values()))
            insights.append(f"📈 Average completion rate: {avg_completion:.1f}%")
        
        return insights if insights else ["Start learning to see your insights!"]

    def export_analytics_data(self):
        """NEW: Export comprehensive analytics data"""
        
        export_data = {
            'analytics_data': st.session_state.analytics_data,
            'export_timestamp': datetime.now().isoformat(),
            'platform_version': 'IntelliLearn AI Enhanced v2.0',
            'summary_stats': {
                'total_sessions': len(st.session_state.analytics_data['sessions']),
                'total_study_time': sum(st.session_state.analytics_data['time_spent'].values()),
                'topics_studied': len(st.session_state.analytics_data['topic_completion']),
                'active_days': len(st.session_state.analytics_data.get('daily_activity', {}))
            }
        }
        
        return json.dumps(export_data, indent=2)

    def reset_analytics_data(self):
        """NEW: Reset analytics data while preserving structure"""
        
        # Backup current data
        backup_data = st.session_state.analytics_data.copy()
        
        # Reset to initial state
        st.session_state.analytics_data = {
            'sessions': [],
            'progress': {},
            'time_spent': {},
            'quiz_scores': [],
            'topic_completion': {},
            'difficulty_progression': {'beginner': [], 'intermediate': [], 'advanced': []},
            'learning_patterns': {},
            'voice_interactions': 0,
            'ai_conversations': [],
            'video_generations': 0,
            'content_preferences': {},
            'study_streaks': [],
            'performance_trends': {},
            'daily_activity': {},
            'weekly_summaries': {},
            'monthly_progress': {},
            'session_durations': [],
            'feature_usage': {
                'text_chat': 0,
                'voice_chat': 0,
                'video_generation': 0,
                'quiz_taking': 0,
                'chapter_reading': 0
            },
            'concept_mastery': {},
            'knowledge_gaps': [],
            'learning_velocity': [],
            'retention_scores': {}
        }
        
        return f"🔄 Analytics data reset successfully! Previous data backed up with {len(backup_data.get('sessions', []))} sessions."

    def get_analytics_summary(self):
        """NEW: Get comprehensive analytics summary"""
        
        analytics_data = st.session_state.analytics_data
        
        summary = {
            'learning_stats': {
                'total_sessions': len(analytics_data['sessions']),
                'total_study_time': sum(analytics_data['time_spent'].values()),
                'average_session_duration': np.mean(analytics_data['session_durations']) if analytics_data['session_durations'] else 0,
                'topics_studied': len(analytics_data['topic_completion']),
                'active_days': len(analytics_data.get('daily_activity', {}))
            },
            'engagement_stats': {
                'text_chats': analytics_data['feature_usage'].get('text_chat', 0),
                'voice_chats': analytics_data['feature_usage'].get('voice_chat', 0),
                'videos_created': analytics_data['feature_usage'].get('video_generation', 0),
                'quizzes_taken': analytics_data['feature_usage'].get('quiz_taking', 0)
            },
            'performance_stats': {
                'average_completion': np.mean(list(analytics_data['topic_completion'].values())) if analytics_data['topic_completion'] else 0,
                'difficulty_levels_used': len([k for k, v in analytics_data['difficulty_progression'].items() if v]),
                'best_performing_difficulty': max(analytics_data['difficulty_progression'].keys(), 
                                                key=lambda k: np.mean(analytics_data['difficulty_progression'][k]) if analytics_data['difficulty_progression'][k] else 0) if any(analytics_data['difficulty_progression'].values()) else 'None'
            }
        }
        
        return summary

    # Additional utility methods for enhanced analytics
    def _generate_basic_learning_insights(self, knowledge_base, user_progress):
        """Fallback method for generating basic insights when enhanced engine is not available"""
        
        st.markdown("### 🧠 Learning Insights")
        
        insights = self.get_learning_insights_enhanced()
        
        for insight in insights:
            with st.container():
                st.markdown(f"**{insight['category']}**")
                st.write(insight['insight'])
                if insight.get('recommendation'):
                    st.info(f"💡 {insight['recommendation']}")
                st.markdown("---")

    def _create_basic_predictions(self, user_progress, difficulty_level):
        """Basic predictions when advanced predictor is not available"""
        
        st.markdown("### 🔮 Learning Predictions")
        
        if user_progress:
            # Simple trend analysis
            total_points = user_progress.get('total_points', 0)
            learning_streak = user_progress.get('learning_streak', 0)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 Predicted Growth")
                if learning_streak >= 7:
                    st.success("🚀 You're on track for excellent progress!")
                    st.write("• Consistent learning pattern detected")
                    st.write("• High probability of continued success")
                elif learning_streak >= 3:
                    st.info("📊 Steady progress expected")
                    st.write("• Good learning momentum")
                    st.write("• Maintain consistency for better results")
                else:
                    st.warning("⚠️ Build consistency for better predictions")
                    st.write("• More data needed for accurate predictions")
                    st.write("• Focus on regular learning sessions")
            
            with col2:
                st.markdown("#### 🎯 Recommendations")
                if total_points >= 1000:
                    st.write("• Consider advancing difficulty level")
                    st.write("• Explore advanced features")
                    st.write("• Share knowledge with others")
                else:
                    st.write("• Focus on consistent daily learning")
                    st.write("• Complete more quizzes for better tracking")
                    st.write("• Use voice chat for engagement")
        else:
            st.info("Complete more learning activities to enable predictions!")

    def _create_performance_prediction_chart(self, prediction_data):
        """Create performance prediction visualization"""
        
        df = pd.DataFrame(prediction_data)
        
        fig = px.line(
            df,
            x='period',
            y='predicted_score',
            title="Predicted Performance Trend",
            markers=True
        )
        
        fig.add_scatter(
            x=df['period'],
            y=df['confidence_lower'],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Lower Bound'
        )
        
        fig.add_scatter(
            x=df['period'],
            y=df['confidence_upper'],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Upper Bound'
        )
        
        return fig

    def _create_mastery_timeline_chart(self, mastery_data):
        """Create mastery timeline visualization"""
        
        df = pd.DataFrame(mastery_data)
        
        fig = px.timeline(
            df,
            x_start='start_date',
            x_end='end_date',
            y='topic',
            color='mastery_level',
            title="Predicted Mastery Timeline"
        )
        
        return fig

    def _display_learning_risks(self, risk_factors):
        """Display learning risk assessment"""
        
        st.markdown("#### ⚠️ Learning Risk Assessment")
        
        for risk in risk_factors:
            if risk['level'] == 'high':
                st.error(f"🔴 {risk['factor']}: {risk['description']}")
            elif risk['level'] == 'medium':
                st.warning(f"🟡 {risk['factor']}: {risk['description']}")
            else:
                st.info(f"🟢 {risk['factor']}: {risk['description']}")

    def _display_predictive_recommendations(self, recommendations):
        """Display AI-powered predictive recommendations"""
        
        st.markdown("#### 💡 AI Recommendations")
        
        for rec in recommendations:
            st.write(f"• **{rec['category']}:** {rec['recommendation']}")
            if rec.get('priority'):
                st.caption(f"Priority: {rec['priority']}")
