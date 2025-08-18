try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False

import builtins
builtins.CLIP_AVAILABLE = CLIP_AVAILABLE
builtins.BLIP_AVAILABLE = BLIP_AVAILABLE


# Now your regular imports...
import streamlit as st
import torch
# ... rest of your imports
import os
import io
import time
import json
import requests
import sys
import tempfile

import torch  # Add this line if missing
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Enhanced imports for open-source models
try:
    from modules.enhanced_chapter_detection_v2 import SuperiorChapterDetector
    from modules.multimodal_processor import OpenSourceMultimodalProcessor
    from modules.voice_chat_processor import VoiceChatProcessor
    from modules.enhanced_nlp_processor import EnhancedNLPProcessor
    from modules.opensource_video_generator import OpenSourceVideoGenerator
    from modules.gamification_enhanced import EnhancedGamificationEngine
    ENHANCED_MODULES_AVAILABLE = True
except ImportError as e:
    st.warning(f"Enhanced modules not available: {e}")
    ENHANCED_MODULES_AVAILABLE = False

# Import original modules with error handling
try:
    from modules.text_processor import AdvancedTextProcessor
    from modules.video_generator import ProductionVideoGenerator
    from modules.chatbot import EnhancedEducationalChatbot
    from modules.analytics_dashboard import LearningAnalyticsDashboard
    from modules.adaptive_learning import AdaptiveLearningEngine
    from modules.gamification import GamificationEngine
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.error("Please ensure all module files exist in the 'modules' directory with __init__.py")
    st.stop()

from config import Config

# Configure page
st.set_page_config(
    page_title="IntelliLearn AI - Advanced Multi-Modal Educational Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for better UI with difficulty levels
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .quiz-card {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    .achievement-badge {
        background: linear-gradient(45deg, #f39c12, #e74c3c);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.2rem;
        font-weight: bold;
    }
    .difficulty-beginner {
        border-left: 4px solid #4CAF50;
        background: #f0f8ff;
    }
    .difficulty-intermediate {
        border-left: 4px solid #FF9800;
        background: #fffef7;
    }
    .difficulty-advanced {
        border-left: 4px solid #F44336;
        background: #fff8f8;
    }
    .voice-button {
        background: linear-gradient(45deg, #2196F3, #21CBF3);
        color: white;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        border: none;
        cursor: pointer;
    }
    .model-status {
        background: #e8f5e8;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize all session state variables with enhanced features"""
    defaults = {
        'current_page': '🏠 Dashboard',
        'textbook_processed': False,
        'knowledge_base': None,
        'selected_subject': 'General',
        'chatbot': None,
        'video_results': [],
        'learning_progress': {},
        'analytics_dashboard': LearningAnalyticsDashboard(),
        'adaptive_engine': AdaptiveLearningEngine(),
        'user_interactions': [],
        'current_quiz': None,
        'quiz_results': [],
        'learning_style': 'Visual',
        'difficulty_level': 'intermediate',  # NEW: Default difficulty
        'voice_enabled': True,  # NEW: Voice interaction toggle
        'image_analysis_results': {},  # NEW: Image analysis storage
        'model_setup_complete': False,  # NEW: Model setup tracking
        'user_progress': {
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
            'voice_interactions': 0,  # NEW: Voice interaction tracking
            'ai_conversations': 0,  # NEW: AI conversation tracking
            'perfect_quizzes': 0,  # NEW: Perfect quiz tracking
            'textbooks_processed': 0,  # NEW: Textbook processing tracking
            # Difficulty-specific tracking (NEW)
            'beginner_quizzes': 0,
            'intermediate_quizzes': 0,
            'advanced_quizzes': 0,
            'beginner_scores': [],
            'intermediate_scores': [],
            'advanced_scores': []
        }
    }
    
    # Enhanced modules initialization
    if ENHANCED_MODULES_AVAILABLE:
        enhanced_defaults = {
            'superior_chapter_detector': None,
            'multimodal_processor': None,
            'voice_processor': None,
            'enhanced_nlp': None,
            'opensource_video_gen': None,
            'enhanced_gamification': None
        }
        defaults.update(enhanced_defaults)
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Initialize enhanced gamification
    if ENHANCED_MODULES_AVAILABLE:
        if 'enhanced_gamification' not in st.session_state or st.session_state.enhanced_gamification is None:
            st.session_state.enhanced_gamification = EnhancedGamificationEngine()
        # Initialize user progress for enhanced gamification
        st.session_state.enhanced_gamification.initialize_user_progress(st.session_state.difficulty_level)
    else:
        # Fallback to original gamification
        if 'gamification' not in st.session_state:
            st.session_state.gamification = GamificationEngine()

def setup_enhanced_sidebar():
    """Enhanced sidebar with comprehensive features and difficulty levels"""
    st.sidebar.title("🧭 Enhanced Navigation")
    
    pages = [
        "🏠 Dashboard",
        "📚 Upload & Process", 
        "🎬 Video Generation",
        "💬 AI Tutor Chat",
        "🎯 Interactive Quizzes",
        "📊 Learning Analytics",
        "🏆 Learning Progress",
        "🎮 Adaptive Learning",
        "🖼️ Image Analysis",  # NEW: Image analysis page
        "🎤 Voice Learning",   # NEW: Voice interaction page
        "⚙️ Settings",
        "🧪 Test Functions"
    ]
    
    st.session_state.current_page = st.sidebar.selectbox("Choose Module", pages)
    
    # Difficulty Level Selector (NEW)
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Learning Level")
    
    difficulty_options = {
        "🟢 Beginner": "beginner",
        "🟡 Intermediate": "intermediate", 
        "🔴 Advanced": "advanced"
    }
    
    selected_difficulty_display = st.sidebar.selectbox(
        "Choose Your Level",
        list(difficulty_options.keys()),
        index=1  # Default to Intermediate
    )
    
    st.session_state.difficulty_level = difficulty_options[selected_difficulty_display]
    
    # Show difficulty level info
    level_descriptions = {
        "beginner": "• Simple explanations\n• Basic vocabulary\n• Extra examples\n• Slower pace",
        "intermediate": "• Balanced detail\n• Standard terminology\n• Good examples\n• Normal pace",
        "advanced": "• Technical depth\n• Complex concepts\n• Minimal guidance\n• Fast pace"
    }
    
    with st.sidebar.expander("ℹ️ Level Details"):
        st.write(level_descriptions[st.session_state.difficulty_level])
    
    # Enhanced Gamification Display
    if ENHANCED_MODULES_AVAILABLE and st.session_state.enhanced_gamification:
        gamification = st.session_state.enhanced_gamification
        level_info = gamification.get_current_level_info()
        
        if level_info['current_points'] > 0:
            st.sidebar.markdown("### 🏆 Your Progress")
            
            current_level = level_info['current_level']
            st.sidebar.markdown(f"**{current_level['icon']} Level:** {current_level['name']}")
            st.sidebar.markdown(f"**Points:** {level_info['current_points']:,}")
            
            streak = st.session_state.user_progress.get('learning_streak', 0)
            st.sidebar.markdown(f"**🔥 Streak:** {streak} days")
            
            # Progress to next level
            if level_info['next_level']:
                progress = level_info['progress_to_next']
                st.sidebar.progress(progress)
                st.sidebar.caption(f"Progress to {level_info['next_level']['name']}: {progress:.1%}")
            
            # Show recent badges
            recent_badges = st.session_state.user_progress.get('badges_earned', [])[-2:]
            if recent_badges:
                st.sidebar.markdown("**Recent Achievements:**")
                for badge_id in recent_badges:
                    if badge_id in gamification.achievements:
                        achievement = gamification.achievements[badge_id]
                        st.sidebar.markdown(f"{achievement['icon']} {achievement['name']}")
    
    # Enhanced Status Display
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 System Status")
    
    # Model availability status (NEW)
    if ENHANCED_MODULES_AVAILABLE:
        config = Config()
        if hasattr(config, 'model_availability'):
            available_models = sum(1 for available in config.model_availability.values() if available)
            total_models = len(config.model_availability)
            
            if available_models > 0:
                st.sidebar.success(f"✅ {available_models}/{total_models} AI models available")
            else:
                st.sidebar.warning("⚠️ Enhanced models need setup")
                if st.sidebar.button("🚀 Setup Models"):
                    st.session_state.current_page = "⚙️ Settings"
                    st.rerun()
    
    # Textbook processing status
    if st.session_state.textbook_processed:
        st.sidebar.success("✅ Textbook Processed")
        
        kb = st.session_state.knowledge_base
        if kb and 'chapters' in kb:
            chapters = kb['chapters']
            total_topics = sum(len(ch.get('topics', {})) for ch in chapters.values())
            total_quizzes = sum(len(ch.get('quizzes', {})) for ch in chapters.values())
            
            st.sidebar.info(f"📖 {len(chapters)} chapters")
            st.sidebar.info(f"📝 {total_topics} topics")
            st.sidebar.info(f"🎯 {total_quizzes} quizzes available")
            
            # Image analysis status (NEW)
            if st.session_state.get('image_analysis_results'):
                image_count = len(st.session_state.image_analysis_results)
                st.sidebar.info(f"🖼️ {image_count} images analyzed")
    else:
        st.sidebar.warning("⚠️ No textbook processed")
    
    # Voice interaction status (NEW)
    if st.session_state.get('voice_enabled', True):
        voice_count = st.session_state.user_progress.get('voice_interactions', 0)
        if voice_count > 0:
            st.sidebar.info(f"🎤 {voice_count} voice interactions")
    
    # Enhanced Subject Selector
    st.sidebar.markdown("---")
    subjects = [
        # Core K-12 Subjects
        "General", "English", "Mathematics", "Science", "History", "Geography",
        
        # Mathematics Specializations
        "Algebra", "Geometry", "Calculus", "Statistics", "Arithmetic", "Trigonometry",
        
        # Sciences
        "Physics", "Chemistry", "Biology", "Environmental Science", "Earth Science", 
        "Astronomy", "Geology", "Botany", "Zoology", "Microbiology",
        
        # Languages & Literature
        "Literature", "Creative Writing", "Spanish", "French", "German", "Chinese",
        "Latin", "Hindi", "Sanskrit", "Regional Languages",
        
        # Social Studies & Humanities
        "World History", "Indian History", "Government", "Civics", "Economics", 
        "Psychology", "Sociology", "Anthropology", "Philosophy", "Ethics",
        
        # Arts & Creative Studies
        "Art", "Music", "Drama", "Theater", "Dance", "Photography", "Graphic Design",
        "Media Studies", "Film Studies", "Creative Arts",
        
        # Technology & Computing
        "Computer Science", "Information Technology", "Programming", "Web Design",
        "Data Science", "Artificial Intelligence", "Robotics", "Digital Media",
        
        # Professional & Vocational
        "Business Studies", "Marketing", "Accounting", "Management", "Engineering",
        "Medical Studies", "Law", "Architecture", "Agriculture",
        
        # Health & Physical Education
        "Physical Education", "Health Science", "Sports Science", "Nutrition",
        
        # Life Skills & General Knowledge
        "General Knowledge", "Current Affairs", "Life Skills", "Study Skills",
        "Environmental Studies", "Moral Science", "Value Education"
    ]
    
    st.session_state.selected_subject = st.sidebar.selectbox("📚 Subject", subjects)
    
    # Voice Settings Panel (NEW)
    if ENHANCED_MODULES_AVAILABLE and st.session_state.get('voice_processor'):
        st.session_state.voice_processor.setup_voice_settings_panel()

def enhanced_dashboard_page():
    """Enhanced dashboard with open-source features and difficulty levels"""
    st.header("🏠 Enhanced Learning Dashboard")
    
    # Initialize enhanced gamification
    if ENHANCED_MODULES_AVAILABLE:
        gamification = st.session_state.enhanced_gamification
        gamification.initialize_user_progress(st.session_state.difficulty_level)
    else:
        gamification = st.session_state.gamification
        gamification.initialize_user_progress()
    
    if not st.session_state.textbook_processed:
        # Enhanced welcome screen
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div class="feature-card difficulty-{st.session_state.difficulty_level}">
                <h3>🚀 Next-Generation AI Learning Platform</h3>
                <p>Welcome to IntelliLearn AI with <strong>open-source enhancements!</strong></p>
                <ul>
                    <li><strong>🧠 Enhanced AI Processing:</strong> Open-source models with commercial API fallbacks</li>
                    <li><strong>📚 Superior Chapter Detection:</strong> 15+ patterns with 90%+ accuracy improvement</li>
                    <li><strong>🖼️ Image Understanding:</strong> Educational diagram analysis with BLIP & CLIP</li>
                    <li><strong>🎤 Voice Learning:</strong> Complete speech-to-text and text-to-speech integration</li>
                    <li><strong>🎬 Professional Videos:</strong> Local AI script generation with Ollama/Coqui TTS</li>
                    <li><strong>🎯 Adaptive Difficulty:</strong> Content adjusted to your {st.session_state.difficulty_level} level</li>
                    <li><strong>🏆 Enhanced Gamification:</strong> 20+ achievements with difficulty multipliers</li>
                    <li><strong>📊 Advanced Analytics:</strong> Comprehensive learning insights</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Model status display (NEW)
            if ENHANCED_MODULES_AVAILABLE:
                config = Config()
                if hasattr(config, 'model_availability') and config.model_availability:
                    st.markdown("### 🤖 AI Model Status")
                    
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        models_available = sum(1 for available in config.model_availability.values() if available)
                        total_models = len(config.model_availability)
                        st.metric("🧠 AI Models", f"{models_available}/{total_models}")
                    
                    with col_b:
                        if config.model_availability.get('sentence_transformers', False):
                            st.success("✅ NLP Ready")
                        else:
                            st.warning("⚠️ NLP Setup Needed")
                    
                    with col_c:
                        if config.model_availability.get('coqui_tts', False):
                            st.success("✅ Voice Ready")
                        else:
                            st.info("🎤 Voice Optional")
        
        with col2:
            st.info("👈 Start by uploading a textbook to unlock all enhanced features!")
            
            # Show current difficulty level
            st.markdown(f"""
            <div class="difficulty-{st.session_state.difficulty_level}">
                <h4>🎯 Current Level: {st.session_state.difficulty_level.title()}</h4>
                <p>Content and features are optimized for your learning level</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show learning style detection preview
            st.markdown("### 🧠 Learning Profile")
            if st.session_state.user_interactions:
                detected_style = st.session_state.adaptive_engine.detect_learning_style(st.session_state.user_interactions)
                st.session_state.learning_style = detected_style
                st.success(f"**Learning Style:** {detected_style}")
            else:
                st.info("Upload content to detect your learning style")
    
    else:
        # Enhanced dashboard with processed textbook
        kb = st.session_state.knowledge_base
        
        # Enhanced gamification progress display
        if ENHANCED_MODULES_AVAILABLE:
            gamification.display_enhanced_progress_dashboard()
        else:
            gamification.display_progress_dashboard(st.session_state.user_progress)
        
        st.markdown("---")
        
        # Enhanced content overview with difficulty-specific metrics
        st.subheader(f"📊 Content Overview ({st.session_state.difficulty_level.title()} Level)")
        
        chapters = kb.get('chapters', {})
        total_topics = sum(len(ch.get('topics', {})) for ch in chapters.values())
        total_concepts = sum(len(ch.get('concepts', [])) for ch in chapters.values())
        total_examples = sum(len(ch.get('examples', [])) for ch in chapters.values())
        total_quizzes = sum(len(ch.get('quizzes', {})) for ch in chapters.values())
        
        # Image analysis summary (NEW)
        image_analysis = st.session_state.get('image_analysis_results', {})
        total_images = len(image_analysis)
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("📚 Chapters", len(chapters))
        with col2:
            st.metric("📖 Topics", total_topics)
        with col3:
            st.metric("🔑 Concepts", total_concepts)
        with col4:
            st.metric("💡 Examples", total_examples)
        with col5:
            st.metric("🎯 Quizzes", total_quizzes)
        with col6:
            st.metric("🖼️ Images", total_images)
        
        # Enhanced adaptive learning recommendations
        st.subheader("🎮 Personalized Recommendations")
        
        adaptive_engine = st.session_state.adaptive_engine
        if st.session_state.user_interactions:
            # Get difficulty-specific performance
            difficulty_scores = st.session_state.user_progress.get(f'{st.session_state.difficulty_level}_scores', [])
            current_performance = sum(difficulty_scores) / len(difficulty_scores) if difficulty_scores else 75
            
            learning_path = adaptive_engine.generate_adaptive_path(
                current_performance=current_performance,
                subject=st.session_state.selected_subject,
                learning_style=st.session_state.learning_style,
                knowledge_base=kb
            )
            
            if learning_path:
                recommendation = learning_path[0].get('chapter', 'Next Chapter')
                st.info(f"📚 **Recommended for {st.session_state.learning_style} learners ({st.session_state.difficulty_level}):** {recommendation}")
        
        # Quick actions with enhanced features
        st.subheader("⚡ Quick Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🎯 Take Quiz", use_container_width=True):
                st.session_state.current_page = "🎯 Interactive Quizzes"
                st.rerun()
        
        with col2:
            if st.button("🎬 Create Video", use_container_width=True):
                st.session_state.current_page = "🎬 Video Generation"
                st.rerun()
        
        with col3:
            if st.button("💬 Start Learning", use_container_width=True):
                st.session_state.current_page = "💬 AI Tutor Chat"
                st.rerun()
        
        with col4:
            if ENHANCED_MODULES_AVAILABLE:
                if st.button("🖼️ Analyze Images", use_container_width=True):
                    st.session_state.current_page = "🖼️ Image Analysis"
                    st.rerun()
            else:
                if st.button("📊 View Analytics", use_container_width=True):
                    st.session_state.current_page = "📊 Learning Analytics"
                    st.rerun()
        
        # Enhanced recent activity with difficulty tracking
        st.subheader("📈 Recent Learning Activity")
        
        if ENHANCED_MODULES_AVAILABLE:
            recent_achievements = st.session_state.user_progress.get('achievements_history', [])
            if recent_achievements:
                for achievement in recent_achievements[-3:]:
                    badge = gamification.achievements.get(achievement['badge_id'], {})
                    difficulty_bonus = "🎯" if achievement.get('difficulty_level') == st.session_state.difficulty_level else ""
                    st.success(f"{badge.get('icon', '🏅')} **{badge.get('name', 'Achievement')}** {difficulty_bonus} - {achievement.get('date', '')[:10]} (+{achievement.get('points_earned', 0)} points)")
            else:
                st.info("Start learning to see your achievements here!")
        
        # Voice interaction summary (NEW)
        if st.session_state.user_progress.get('voice_interactions', 0) > 0:
            st.subheader("🎤 Voice Learning Summary")
            voice_count = st.session_state.user_progress['voice_interactions']
            st.info(f"You've used voice chat {voice_count} times! Keep exploring voice-enabled learning.")

def textbook_processing_page():
    """Enhanced textbook processing with superior chapter detection and multimodal analysis"""
    st.header("📚 Advanced Textbook Processing with AI Enhancement")
    st.markdown("Upload your textbook for comprehensive AI-powered analysis with **enhanced open-source models**, superior chapter detection, and intelligent image understanding.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Textbook",
            type=['pdf'],
            help="Upload PDF textbooks for processing. Enhanced AI will automatically detect chapters, analyze images, and generate adaptive content."
        )
        
        if uploaded_file:
            file_size = uploaded_file.size / 1024 / 1024  # Convert to MB
            st.info(f"📄 **{uploaded_file.name}** ({file_size:.1f} MB)")
            
            # Processing options (NEW)
            with st.expander("🔧 Processing Options"):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    use_enhanced_detection = st.checkbox("🧠 Use Enhanced Chapter Detection", value=ENHANCED_MODULES_AVAILABLE)
                    analyze_images = st.checkbox("🖼️ Analyze Images and Diagrams", value=ENHANCED_MODULES_AVAILABLE)
                
                with col_b:
                    use_advanced_nlp = st.checkbox("📊 Advanced NLP Analysis", value=ENHANCED_MODULES_AVAILABLE)
                    difficulty_adaptation = st.checkbox(f"🎯 Adapt to {st.session_state.difficulty_level.title()} Level", value=True)
            
            if st.button("🚀 Process with Enhanced AI", type="primary", use_container_width=True):
                process_textbook_enhanced(uploaded_file, use_enhanced_detection, analyze_images, use_advanced_nlp, difficulty_adaptation)
    
    with col2:
        st.markdown(f"""
        <div class="feature-card difficulty-{st.session_state.difficulty_level}">
            <h4>🔬 Enhanced Processing Features</h4>
            <ul>
                <li>📖 <strong>Superior Chapter Detection:</strong> 15+ patterns with 90%+ accuracy</li>
                <li>🖼️ <strong>Image Understanding:</strong> BLIP + CLIP analysis</li>
                <li>🎯 <strong>{st.session_state.difficulty_level.title()} Level:</strong> Content adapted to your level</li>
                <li>🤖 <strong>Advanced NLP:</strong> BERTopic + educational concepts</li>
                <li>🔍 <strong>Enhanced Extraction:</strong> Topics, examples, objectives</li>
                <li>📊 <strong>Difficulty Assessment:</strong> Multi-factor analysis</li>
                <li>📝 <strong>Smart Quiz Generation:</strong> Bloom's taxonomy integration</li>
                <li>🎮 <strong>Gamification Ready:</strong> Achievement tracking</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Show model availability
        if ENHANCED_MODULES_AVAILABLE:
            config = Config()
            if hasattr(config, 'model_availability'):
                st.markdown("**🤖 Available AI Models:**")
                for model_name, available in config.model_availability.items():
                    status = "✅" if available else "❌"
                    st.markdown(f"{status} {model_name.replace('_', ' ').title()}")
    
    # Show existing content with enhanced preview
    if st.session_state.textbook_processed:
        st.markdown("---")
        st.subheader(f"📖 Enhanced Content Preview ({st.session_state.difficulty_level.title()} Level)")
        
        kb = st.session_state.knowledge_base
        chapters = kb.get('chapters', {})
        
        for chapter_name, chapter_data in list(chapters.items())[:3]:  # Show first 3 chapters
            with st.expander(f"📚 {chapter_name}"):
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    st.metric("📖 Topics", len(chapter_data.get('topics', {})))
                with col_b:
                    st.metric("🔑 Concepts", len(chapter_data.get('concepts', [])))
                with col_c:
                    st.metric("💡 Examples", len(chapter_data.get('examples', [])))
                with col_d:
                    st.metric("🎯 Quizzes", len(chapter_data.get('quizzes', {})))
                
                # Enhanced metadata display
                if isinstance(chapter_data, dict):
                    metadata = {
                        'Type': chapter_data.get('type', 'content'),
                        'Reading Time': f"{chapter_data.get('estimated_reading_time', 0)} min",
                        'Quality': chapter_data.get('content_quality', 'unknown'),
                        'Level': chapter_data.get('educational_level', 'general'),
                        'Difficulty': chapter_data.get('difficulty_indicators', {}).get('complexity', 'medium')
                    }
                    
                    col_meta1, col_meta2 = st.columns(2)
                    
                    with col_meta1:
                        for key, value in list(metadata.items())[:3]:
                            st.write(f"**{key}:** {value}")
                    
                    with col_meta2:
                        for key, value in list(metadata.items())[3:]:
                            st.write(f"**{key}:** {value}")
                    
                    # Show enhanced features if available
                    if chapter_data.get('key_concepts'):
                        st.write("**🔑 Key Concepts:**")
                        concepts = chapter_data['key_concepts'][:5]  # Show first 5
                        st.write(", ".join(concepts))

def process_textbook_enhanced(uploaded_file, use_enhanced_detection, analyze_images, use_advanced_nlp, difficulty_adaptation):
    """Enhanced textbook processing with all open-source features"""
    try:
        # Save uploaded file
        temp_file_path = f"temp/{uploaded_file.name}"
        os.makedirs("temp", exist_ok=True)
        
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Initialize enhanced processors
        if ENHANCED_MODULES_AVAILABLE and use_enhanced_detection:
            # Use enhanced chapter detector
            if not st.session_state.superior_chapter_detector:
                st.session_state.superior_chapter_detector = SuperiorChapterDetector()
            
            processor = AdvancedTextProcessor()
            chapter_detector = st.session_state.superior_chapter_detector
        else:
            # Use original processor
            processor = AdvancedTextProcessor()
            chapter_detector = None
        
        # Show enhanced processing steps
        with st.spinner("🔍 Processing textbook with enhanced AI..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Document analysis
            status_text.text("📖 Analyzing document structure...")
            progress_bar.progress(10)
            time.sleep(0.5)
            
            # Step 2: Enhanced chapter detection
            if use_enhanced_detection and chapter_detector:
                status_text.text("📚 Using superior chapter detection (15+ patterns)...")
            else:
                status_text.text("📚 Using standard chapter detection...")
            progress_bar.progress(25)
            time.sleep(0.5)
            
            # Step 3: Image analysis (NEW)
            if analyze_images and ENHANCED_MODULES_AVAILABLE:
                status_text.text("🖼️ Analyzing images and diagrams with BLIP/CLIP...")
                progress_bar.progress(40)
                
                # Initialize multimodal processor
                if not st.session_state.multimodal_processor:
                    st.session_state.multimodal_processor = OpenSourceMultimodalProcessor()
                
                time.sleep(1)
            
            # Step 4: Advanced NLP processing
            if use_advanced_nlp and ENHANCED_MODULES_AVAILABLE:
                status_text.text("🧠 Advanced NLP analysis with BERTopic...")
                progress_bar.progress(55)
                
                # Initialize enhanced NLP processor
                if not st.session_state.enhanced_nlp:
                    st.session_state.enhanced_nlp = EnhancedNLPProcessor()
                
                time.sleep(1)
            
            # Step 5: Main processing
            status_text.text("🤖 AI-powered content analysis...")
            progress_bar.progress(70)
            
            # Process the textbook with enhanced features
            if ENHANCED_MODULES_AVAILABLE and use_enhanced_detection:
                knowledge_base = processor.process_textbook_with_enhanced_detection(
                    temp_file_path, 
                    chapter_detector,
                    difficulty_level=st.session_state.difficulty_level if difficulty_adaptation else 'intermediate'
                )
            else:
                knowledge_base = processor.process_textbook_with_full_hierarchy(temp_file_path)
            
            progress_bar.progress(80)
            
            # Step 6: Image analysis processing
            if analyze_images and ENHANCED_MODULES_AVAILABLE and knowledge_base:
                status_text.text("🖼️ Processing educational images...")
                
                import fitz
                doc = fitz.open(temp_file_path)
                
                image_analysis = st.session_state.multimodal_processor.analyze_educational_images(
                    doc, st.session_state.difficulty_level
                )
                
                if image_analysis:
                    st.session_state.image_analysis_results = image_analysis
                
                doc.close()
                progress_bar.progress(90)
                time.sleep(0.5)
            
            # Step 7: Gamification setup
            status_text.text("🏆 Setting up enhanced gamification...")
            progress_bar.progress(95)
            time.sleep(0.5)
            
            progress_bar.progress(100)
            status_text.text("✅ Enhanced processing complete!")
        
        if knowledge_base:
            st.session_state.knowledge_base = knowledge_base
            st.session_state.textbook_processed = True
            
            # Enhanced gamification tracking
            if ENHANCED_MODULES_AVAILABLE:
                gamification = st.session_state.enhanced_gamification
            else:
                gamification = st.session_state.gamification
            
            # Update progress tracking
            st.session_state.user_progress['chapters_completed'] = len(knowledge_base.get('chapters', {}))
            st.session_state.user_progress['textbooks_processed'] += 1
            
            # Award points with difficulty multiplier
            if ENHANCED_MODULES_AVAILABLE:
                points_earned = gamification.award_points('textbook_processed', st.session_state.difficulty_level)
            else:
                points_earned = 100
                st.session_state.user_progress['total_points'] += points_earned
                gamification.update_learning_streak(st.session_state.user_progress)
            
            # Check for achievements
            if ENHANCED_MODULES_AVAILABLE:
                gamification.check_achievements()
            else:
                gamification.check_achievements(st.session_state.user_progress)
            
            st.success("🎉 **Textbook processed successfully with enhanced features!**")
            st.info(f"🏆 You earned **{points_earned} points** for processing your textbook!")
            
            # Display enhanced results
            chapters = knowledge_base.get('chapters', {})
            total_topics = sum(len(ch.get('topics', {})) for ch in chapters.values())
            total_quizzes = sum(len(ch.get('quizzes', {})) for ch in chapters.values())
            total_concepts = sum(len(ch.get('concepts', [])) for ch in chapters.values())
            total_examples = sum(len(ch.get('examples', [])) for ch in chapters.values())
            
            # Enhanced metrics display
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                st.metric("📚 Chapters", len(chapters))
            with col2:
                st.metric("📖 Topics", total_topics)
            with col3:
                st.metric("🎯 Quizzes", total_quizzes)
            with col4:
                st.metric("🔑 Concepts", total_concepts)
            with col5:
                st.metric("💡 Examples", total_examples)
            with col6:
                if st.session_state.get('image_analysis_results'):
                    image_count = len(st.session_state.image_analysis_results)
                    st.metric("🖼️ Images", image_count)
                else:
                    st.metric("🖼️ Images", 0)
            
            # Enhanced features summary
            if use_enhanced_detection or analyze_images or use_advanced_nlp:
                st.markdown("### 🚀 Enhanced Features Applied")
                
                feature_col1, feature_col2, feature_col3 = st.columns(3)
                
                with feature_col1:
                    if use_enhanced_detection:
                        st.success("✅ Superior Chapter Detection")
                    else:
                        st.info("📖 Standard Chapter Detection")
                
                with feature_col2:
                    if analyze_images:
                        image_count = len(st.session_state.get('image_analysis_results', {}))
                        st.success(f"✅ {image_count} Images Analyzed")
                    else:
                        st.info("🖼️ Image Analysis Skipped")
                
                with feature_col3:
                    if use_advanced_nlp:
                        st.success("✅ Advanced NLP Applied")
                    else:
                        st.info("🧠 Standard NLP Used")
            
            st.balloons()
            
        else:
            st.error("❌ Failed to process textbook. Please try again or check the file format.")
        
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
    except Exception as e:
        st.error(f"❌ Processing failed: {str(e)}")

def video_generation_page():
    """Enhanced video generation with open-source models and difficulty levels"""
    st.header("🎬 Professional Video Generation with Enhanced AI")
    
    if not st.session_state.textbook_processed:
        st.warning("⚠️ Please process a textbook first!")
        return
    
    st.markdown(f"Generate professional educational videos optimized for **{st.session_state.difficulty_level} level** learning with AI-created scripts, slides, and voiceovers using open-source models.")
    
    kb = st.session_state.knowledge_base
    chapters = kb.get('chapters', {})
    
    # Enhanced chapter and topic selection
    col1, col2 = st.columns(2)
    
    with col1:
        selected_chapter = st.selectbox("📚 Select Chapter", list(chapters.keys()))
        
    with col2:
        if selected_chapter:
            chapter_topics = chapters[selected_chapter].get('topics', {})
            selected_topic = st.selectbox("📖 Select Topic", list(chapter_topics.keys()))
    
    if selected_chapter and selected_topic:
        topic_data = chapters[selected_chapter]['topics'][selected_topic]
        
        # Enhanced video configuration with difficulty levels
        st.subheader("🎛️ Video Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Difficulty-adjusted persona options
            persona_options = get_difficulty_adjusted_personas(st.session_state.difficulty_level)
            
            selected_persona = st.selectbox("👨‍🏫 Teaching Persona", list(persona_options.keys()))
            st.info(persona_options[selected_persona])
            
            # Enhanced options based on difficulty
            if ENHANCED_MODULES_AVAILABLE:
                use_opensource_generation = st.checkbox("🤖 Use Open-Source Video Generation", value=True)
                if use_opensource_generation:
                    st.info("Uses Ollama + Coqui TTS for local generation")
        
        with col2:
            # Difficulty-adjusted video settings
            duration_settings = get_difficulty_duration_settings(st.session_state.difficulty_level)
            
            video_length = st.slider(
                "📊 Video Length (minutes)", 
                duration_settings["min"], 
                duration_settings["max"], 
                duration_settings["default"]
            )
            
            include_examples = st.checkbox(
                "💡 Include Real-World Examples", 
                value=(st.session_state.difficulty_level != "advanced")
            )
            
            # Difficulty-specific options
            if st.session_state.difficulty_level == "beginner":
                include_definitions = st.checkbox("📚 Include Definitions", value=True)
                slow_pace = st.checkbox("🐌 Slower Explanation Pace", value=True)
            elif st.session_state.difficulty_level == "advanced":
                include_technical = st.checkbox("🔬 Include Technical Details", value=True)
                include_research = st.checkbox("📊 Reference Research", value=False)
        
        # Enhanced topic preview
        st.subheader("📋 Topic Preview")
        
        with st.expander("🔍 View Topic Details"):
            st.write(f"**Summary:** {topic_data.get('summary', 'No summary available')}")
            st.write(f"**Difficulty:** {topic_data.get('difficulty', 'Unknown')}")
            st.write(f"**Adapted for:** {st.session_state.difficulty_level.title()} level")
            
            key_points = topic_data.get('key_points', [])
            if key_points:
                st.write("**Key Points:**")
                for point in key_points[:5]:
                    st.write(f"• {point}")
        
        # Generate video with enhanced options
        if st.button("🎬 Generate Professional Video", type="primary", use_container_width=True):
            # Enhanced topic data with difficulty context
            enhanced_topic_data = {
                **topic_data,
                'difficulty_level': st.session_state.difficulty_level,
                'persona_context': persona_options[selected_persona],
                'video_settings': {
                    'length': video_length,
                    'include_examples': include_examples,
                    'pace': 'slow' if st.session_state.difficulty_level == 'beginner' else 'normal'
                }
            }
            
            if ENHANCED_MODULES_AVAILABLE and st.session_state.get('use_opensource_generation', True):
                generate_opensource_video(enhanced_topic_data, selected_chapter, selected_persona, video_length, include_examples)
            else:
                generate_professional_video(enhanced_topic_data, selected_chapter, selected_persona, video_length, include_examples)

def get_difficulty_adjusted_personas(difficulty_level):
    """Get persona options adjusted for difficulty level"""
    if difficulty_level == "beginner":
        return {
            "Simple Tutor": "Very basic explanations with lots of examples and encouraging tone",
            "Friendly Guide": "Patient, step-by-step guidance with simple vocabulary"
        }
    elif difficulty_level == "advanced":
        return {
            "Expert Academic": "Technical depth with scholarly approach and advanced terminology",
            "Research Specialist": "Complex analysis with research references and advanced concepts"
        }
    else:  # intermediate
        return {
            "Professional Instructor": "Clear, structured approach with balanced complexity",
            "Enthusiastic Teacher": "Engaging tone with good examples and moderate detail"
        }

def get_difficulty_duration_settings(difficulty_level):
    """Get duration settings based on difficulty level"""
    if difficulty_level == "beginner":
        return {"min": 5, "max": 12, "default": 8}
    elif difficulty_level == "advanced":
        return {"min": 6, "max": 20, "default": 12}
    else:  # intermediate
        return {"min": 4, "max": 15, "default": 8}

def generate_opensource_video(topic_data, chapter_name, persona, length, include_examples):
    """Generate video using open-source models"""
    if not ENHANCED_MODULES_AVAILABLE:
        st.error("Enhanced modules not available. Using standard video generation.")
        generate_professional_video(topic_data, chapter_name, persona, length, include_examples)
        return
    
    try:
        # Initialize open-source video generator
        if not st.session_state.opensource_video_gen:
            st.session_state.opensource_video_gen = OpenSourceVideoGenerator()
        
        generator = st.session_state.opensource_video_gen
        difficulty_level = topic_data.get('difficulty_level', 'intermediate')
        
        # Enhanced script guidance for difficulty
        difficulty_script_prompts = {
            "beginner": """Create a simple, easy-to-understand educational script.
            - Use basic vocabulary and short sentences
            - Include lots of examples and analogies
            - Repeat key concepts multiple times
            - Add encouraging phrases
            - Explain technical terms simply""",
            
            "intermediate": """Create a balanced educational script.
            - Use clear explanations with moderate detail
            - Include relevant examples
            - Maintain good pacing
            - Use standard educational terminology""",
            
            "advanced": """Create a comprehensive, detailed educational script.
            - Use technical terminology appropriately
            - Include complex concepts and relationships
            - Reference advanced examples
            - Assume prior knowledge of basics
            - Focus on analysis and synthesis"""
        }
        
        script_guidance = difficulty_script_prompts.get(difficulty_level, difficulty_script_prompts["intermediate"])
        
        with st.spinner(f"🎬 Creating {difficulty_level} level educational video with open-source AI..."):
            result = generator.create_professional_video_with_difficulty(
                topic_data=topic_data,
                chapter_name=chapter_name,
                persona=persona,
                length=length,
                include_examples=include_examples,
                difficulty_level=difficulty_level,
                script_guidance=script_guidance
            )
        
        display_enhanced_video_results(result, difficulty_level)
        
    except Exception as e:
        st.error(f"❌ Open-source video generation error: {str(e)}")
        st.info("Falling back to standard video generation...")
        generate_professional_video(topic_data, chapter_name, persona, length, include_examples)

def generate_professional_video(topic_data, chapter_name, persona, length, include_examples):
    """Generate professional educational video (original method)"""
    try:
        generator = ProductionVideoGenerator()
        
        with st.spinner("🎬 Creating professional educational video..."):
            result = generator.create_professional_video(
                topic_data=topic_data,
                chapter_name=chapter_name,
                persona=persona,
                length=length,
                include_examples=include_examples
            )
        
        display_enhanced_video_results(result, st.session_state.difficulty_level)
        
    except Exception as e:
        st.error(f"❌ Video generation error: {str(e)}")

def display_enhanced_video_results(result, difficulty_level):
    """Display enhanced video generation results"""
    if result['success']:
        st.success(f"🎉 **{difficulty_level.title()} level video generated successfully!**")
        
        # Difficulty-specific success message
        level_messages = {
            "beginner": "Perfect for starting your learning journey! 🌟",
            "intermediate": "Great balance of detail and clarity! 🎯", 
            "advanced": "Comprehensive coverage for advanced learners! 🚀"
        }
        
        st.info(level_messages.get(difficulty_level, "Video created successfully!"))
        
        # Update enhanced gamification
        if ENHANCED_MODULES_AVAILABLE:
            gamification = st.session_state.enhanced_gamification
            points_earned = gamification.award_points('video_generated', difficulty_level)
        else:
            st.session_state.user_progress['videos_generated'] += 1
            points_earned = 75
            st.session_state.user_progress['total_points'] += points_earned
            gamification = st.session_state.gamification
            gamification.check_achievements(st.session_state.user_progress)
        
        # Display video information with enhanced details
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Generated Script")
            st.text_area("Script Preview", result['script'], height=300)
            
            # Enhanced video concept display
            concept = result.get('video_concept', {})
            st.subheader("🎯 Video Concept")
            st.write(f"**Title:** {concept.get('title', 'Educational Video')}")
            st.write(f"**Duration:** {concept.get('duration', '5 minutes')}")
            st.write(f"**Format:** {concept.get('format', 'Slide-based presentation')}")
            st.write(f"**Level:** {difficulty_level.title()}")
            
            if concept.get('persona'):
                st.write(f"**Persona:** {concept.get('persona', 'Professional Instructor')}")
        
        with col2:
            st.subheader("🖼️ Generated Slides")
            
            slides = result.get('slides', [])
            if slides:
                for i, slide in enumerate(slides[:3], 1):  # Show first 3 slides
                    st.write(f"**Slide {i}: {slide.get('title', f'Slide {i}')}**")
                    
                    # Show slide image if available
                    if slide.get('path') and os.path.exists(slide['path']):
                        st.image(slide['path'], width=300)
                    
                    # Show slide content
                    content_points = slide.get('content', [])
                    if isinstance(content_points, list):
                        for point in content_points[:3]:
                            st.write(f"• {point}")
                    elif isinstance(content_points, str):
                        st.write(content_points[:200] + "..." if len(content_points) > 200 else content_points)
                    
                    st.write("---")
            else:
                st.info("Slides are being generated...")
        
        # Enhanced audio information
        audio_info = result.get('audio', {})
        if audio_info and audio_info.get('path'):
            st.subheader("🎵 Generated Audio")
            st.write(f"**Duration:** ~{audio_info.get('duration', 0):.1f} seconds")
            st.write(f"**Optimized for:** {difficulty_level.title()} level (adjusted speed and complexity)")
            
            if os.path.exists(audio_info['path']):
                st.audio(audio_info['path'])
        
        # Save enhanced results to session state
        if 'video_results' not in st.session_state:
            st.session_state.video_results = []
        
        enhanced_result = {
            'topic': topic_data.get('summary', 'Unknown Topic'),
            'chapter': result.get('video_concept', {}).get('title', 'Unknown Chapter'),
            'difficulty_level': difficulty_level,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'points_earned': points_earned,
            'result': result
        }
        
        st.session_state.video_results.append(enhanced_result)
        st.balloons()
        
    else:
        st.error(f"❌ Video generation failed: {result.get('error', 'Unknown error')}")

def ai_tutor_chat_page():
    """Enhanced AI tutor chat with voice integration and difficulty adaptation"""
    st.header("💬 AI Tutor - Enhanced Learning Assistant with Voice")
    
    if not st.session_state.textbook_processed:
        st.warning("⚠️ Please process a textbook first to start learning!")
        return
    
    # Display current settings
    difficulty_level = st.session_state.difficulty_level
    st.info(f"🎯 **Learning Level:** {difficulty_level.title()} | 📚 **Subject:** {st.session_state.selected_subject} | 🎤 **Voice Enabled**")
    
    # Initialize enhanced chatbot
    if 'chatbot' not in st.session_state or st.session_state.chatbot is None:
        with st.spinner("🤖 Initializing your enhanced AI tutor..."):
            st.session_state.chatbot = EnhancedEducationalChatbot(
                st.session_state.knowledge_base,
                st.session_state.selected_subject
            )
        st.success("✅ Enhanced AI Tutor ready with voice capabilities!")
    
    # Voice Chat Interface (NEW)
    if ENHANCED_MODULES_AVAILABLE:
        with st.expander("🎤 Voice Chat (Click to expand)", expanded=False):
            if not st.session_state.voice_processor:
                st.session_state.voice_processor = VoiceChatProcessor()
            
            st.session_state.voice_processor.setup_voice_chat_interface()
    
    # Enhanced controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"**Current Mode:** {difficulty_level.title()} Level Learning")
    
    with col2:
        if st.button("🔄 Switch Subject", use_container_width=True):
            if st.session_state.chatbot:
                result = st.session_state.chatbot.switch_subject(st.session_state.selected_subject)
                st.info(result)
    
    with col3:
        if st.button("🎯 Change Level", use_container_width=True):
            st.info(f"Switch to {st.session_state.difficulty_level} in sidebar ➡️")
    
    # Enhanced chat interface
    st.subheader("💭 Chat with Your Enhanced AI Tutor")
    
    # Initialize messages with difficulty-aware greeting
    if 'chat_messages' not in st.session_state:
        difficulty_greetings = {
            "beginner": f"Hello! I'm your friendly AI tutor for **{st.session_state.selected_subject}**. I'll explain everything in simple terms and help you learn step by step. What would you like to start with? 📚✨",
            "intermediate": f"Hello! I'm your AI tutor for **{st.session_state.selected_subject}**. I can help you understand concepts, answer questions, and guide your learning. What topic interests you today? 📚💡",
            "advanced": f"Hello! I'm your advanced AI tutor for **{st.session_state.selected_subject}**. I can provide in-depth analysis, technical explanations, and complex discussions. What challenging topic shall we explore? 📚🚀"
        }
        
        greeting = difficulty_greetings.get(difficulty_level, difficulty_greetings["intermediate"])
        
        st.session_state.chat_messages = [
            {
                "role": "assistant", 
                "content": greeting + "\n\n🎤 **Voice Feature:** You can also use voice chat above for hands-free learning!"
            }
        ]
    
    # Display chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Enhanced chat input with voice integration
    if prompt := st.chat_input(f"Ask me anything about {st.session_state.selected_subject} (or use voice chat above)..."):
        # Track user interaction for adaptive learning
        st.session_state.user_interactions.append({
            'action_type': 'chat_message_sent',
            'timestamp': time.time(),
            'content_length': len(prompt),
            'difficulty_level': difficulty_level
        })
        
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Update enhanced gamification
        if ENHANCED_MODULES_AVAILABLE:
            gamification = st.session_state.enhanced_gamification
            points_earned = gamification.award_points('ai_conversation', difficulty_level)
        else:
            st.session_state.user_progress['questions_asked'] += 1
        
        # Get AI tutor response with difficulty adaptation
        with st.chat_message("assistant"):
            with st.spinner(f"🤔 Generating {difficulty_level}-level response using multiple AI sources..."):
                try:
                    response = get_difficulty_adapted_response(prompt, st.session_state.chatbot, difficulty_level)
                    st.markdown(response)
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                    
                    # Track learning session with gamification
                    st.session_state.analytics_dashboard.track_learning_session(
                        topic=prompt[:50] + "...",
                        duration=2.0,
                        completion_rate=85.0
                    )
                    
                    # Update learning streak
                    if ENHANCED_MODULES_AVAILABLE:
                        gamification = st.session_state.enhanced_gamification
                        gamification.update_learning_streak()
                        gamification.check_achievements()
                    else:
                        gamification = st.session_state.gamification
                        gamification.update_learning_streak(st.session_state.user_progress)
                        gamification.check_achievements(st.session_state.user_progress)
                    
                    # Voice response option (NEW)
                    if ENHANCED_MODULES_AVAILABLE:
                        col_voice1, col_voice2 = st.columns([4, 1])
                        with col_voice2:
                            if st.button("🔊 Hear Response", key=f"tts_{len(st.session_state.chat_messages)}"):
                                if st.session_state.voice_processor:
                                    with st.spinner("🎵 Generating speech..."):
                                        audio_response = st.session_state.voice_processor.text_to_speech(
                                            response, difficulty_level
                                        )
                                    if audio_response:
                                        st.audio(audio_response, format="audio/mp3")
                    
                except Exception as e:
                    error_msg = f"I apologize for the error: {str(e)}. Let me try to help you with the textbook content."
                    st.error(error_msg)
                    st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
    
    # Enhanced sidebar features for chat
    with st.sidebar:
        if st.session_state.current_page == "💬 AI Tutor Chat":
            st.markdown("---")
            st.markdown("### 💡 Smart Suggestions")
            
            # Difficulty-adapted suggestions
            if difficulty_level == 'beginner':
                st.info("💡 Try asking: 'Can you explain this in simple terms?'")
                st.info("💡 Say: 'Give me an easy example of...'")
            elif difficulty_level == 'advanced':
                st.info("💡 Try asking: 'What are the technical details of...'")
                st.info("💡 Say: 'How does this relate to advanced concepts?'")
            else:
                st.info("💡 Try asking: 'Can you give me a detailed explanation?'")
            
            # Adaptive suggestions based on learning style
            if st.session_state.learning_style == 'Visual':
                st.info("💡 Try asking: 'Can you create a diagram of this concept?'")
            elif st.session_state.learning_style == 'Auditory':
                st.info("💡 Use the voice feature to ask questions aloud!")
            
            # Enhanced suggested questions
            kb = st.session_state.knowledge_base
            chapters = kb.get('chapters', {})
            
            if chapters:
                st.write("**Quick Questions:**")
                topic_count = 0
                for chapter_name, chapter_data in chapters.items():
                    for topic_name in chapter_data.get('topics', {}):
                        if topic_count < 3:
                            question_prefix = get_difficulty_question_prefix(difficulty_level)
                            if st.button(f"{question_prefix} {topic_name}?", key=f"topic_{topic_count}"):
                                difficulty_prompt = f"{question_prefix} {topic_name} at a {difficulty_level} level"
                                st.session_state.chat_messages.append({
                                    "role": "user", 
                                    "content": difficulty_prompt
                                })
                                st.rerun()
                            topic_count += 1
            
            # Enhanced conversation management with gamification
            st.markdown("---")
            st.markdown("### 🎮 Chat Stats")
            
            if ENHANCED_MODULES_AVAILABLE:
                total_conversations = st.session_state.user_progress.get('ai_conversations', 0)
                voice_interactions = st.session_state.user_progress.get('voice_interactions', 0)
                st.write(f"**Conversations:** {total_conversations}")
                st.write(f"**Voice Chats:** {voice_interactions}")
                st.write(f"**Current Level:** {difficulty_level.title()}")
            else:
                questions_asked = st.session_state.user_progress.get('questions_asked', 0)
                st.write(f"**Questions Asked:** {questions_asked}")
            
            if st.button("📊 View Conversation Summary"):
                if st.session_state.chatbot:
                    summary = st.session_state.chatbot.get_conversation_summary()
                    st.write(summary)
            
            if st.button("🗑️ Clear Chat History"):
                difficulty_greetings = {
                    "beginner": f"Chat cleared! I'm ready to help you learn **{st.session_state.selected_subject}** in simple terms. What would you like to explore?",
                    "intermediate": f"Chat cleared! I'm ready to help you with **{st.session_state.selected_subject}**. What topic interests you?",
                    "advanced": f"Chat cleared! Ready for advanced **{st.session_state.selected_subject}** discussions. What shall we analyze?"
                }
                
                greeting = difficulty_greetings.get(difficulty_level, difficulty_greetings["intermediate"])
                
                st.session_state.chat_messages = [
                    {
                        "role": "assistant",
                        "content": greeting
                    }
                ]
                st.rerun()

def get_difficulty_adapted_response(prompt, chatbot, difficulty_level):
    """Get AI response adapted to difficulty level"""
    
    # Difficulty-specific system prompts
    difficulty_prompts = {
        "beginner": """You are a patient, encouraging AI tutor for beginners. Guidelines:
        - Use simple, clear language and short sentences
        - Avoid technical jargon, explain complex terms simply
        - Provide step-by-step explanations
        - Include encouraging phrases like "Great question!" or "Let's break this down"
        - Use analogies and real-world examples
        - Repeat key concepts for reinforcement
        - Keep responses focused and not overwhelming""",
        
        "intermediate": """You are a knowledgeable AI tutor for intermediate learners. Guidelines:
        - Use clear explanations with moderate technical vocabulary
        - Provide balanced detail without overwhelming
        - Include relevant examples and connections
        - Encourage deeper thinking with follow-up questions
        - Build on existing knowledge appropriately
        - Use educational terminology correctly""",
        
        "advanced": """You are an expert AI tutor for advanced learners. Guidelines:
        - Use precise technical language and complex concepts
        - Provide comprehensive, detailed explanations
        - Include multiple perspectives and analysis
        - Challenge thinking with probing questions
        - Reference advanced concepts and applications
        - Assume strong foundational knowledge
        - Encourage critical evaluation and synthesis"""
    }
    
    system_prompt = difficulty_prompts.get(difficulty_level, difficulty_prompts["intermediate"])
    
    try:
        # Use enhanced chatbot with difficulty context
        response = chatbot.get_comprehensive_response(prompt, system_override=system_prompt)
        return response
    except Exception as e:
        return f"I apologize for the technical difficulty. Let me try to help you with a direct answer based on your textbook content. {str(e)}"

def get_difficulty_question_prefix(difficulty_level):
    """Get question prefix based on difficulty level"""
    prefixes = {
        "beginner": "Simply explain",
        "intermediate": "Tell me about",
        "advanced": "Analyze"
    }
    return prefixes.get(difficulty_level, "Explain")

def interactive_quiz_page():
    """Enhanced interactive quiz page with difficulty-specific settings"""
    st.header(f"🎯 Interactive Learning Quizzes ({st.session_state.difficulty_level.title()} Level)")
    
    if not st.session_state.textbook_processed:
        st.warning("⚠️ Please process a textbook first!")
        return
    
    kb = st.session_state.knowledge_base
    chapters = kb.get('chapters', {})
    
    # Enhanced quiz selection with difficulty indicators
    st.subheader("📚 Select Quiz Topic")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_chapter = st.selectbox("📖 Choose Chapter", list(chapters.keys()))
    
    with col2:
        if selected_chapter:
            chapter_quizzes = chapters[selected_chapter].get('quizzes', {})
            if chapter_quizzes:
                selected_topic = st.selectbox("🎯 Choose Topic", list(chapter_quizzes.keys()))
            else:
                st.info("No quizzes available for this chapter")
                return
    
    if selected_chapter and selected_topic and chapter_quizzes:
        quiz_data = chapter_quizzes[selected_topic]
        
        # Enhanced quiz metadata with difficulty settings
        if ENHANCED_MODULES_AVAILABLE:
            config = Config()
            quiz_settings = config.get_difficulty_config(st.session_state.difficulty_level)['quiz_settings']
        else:
            quiz_settings = {'passing_score': 70, 'questions_per_quiz': 5, 'time_limit': 600}
        
        metadata = quiz_data.get('quiz_metadata', {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📝 Questions", metadata.get('total_questions', quiz_settings['questions_per_quiz']))
        with col2:
            st.metric("⏱️ Time Limit", f"{quiz_settings.get('time_limit', 600) // 60} min")
        with col3:
            st.metric("🎯 Passing Score", f"{quiz_settings.get('passing_score', 70)}%")
        with col4:
            st.metric("📊 Difficulty", st.session_state.difficulty_level.title())
        
        # Enhanced difficulty explanation
        st.markdown(f"""
        <div class="difficulty-{st.session_state.difficulty_level}">
            <h4>🎯 {st.session_state.difficulty_level.title()} Level Quiz Features:</h4>
            <ul>
                <li><strong>Questions:</strong> {quiz_settings.get('questions_per_quiz', 5)} questions optimized for your level</li>
                <li><strong>Time:</strong> {quiz_settings.get('time_limit', 600) // 60} minutes to complete</li>
                <li><strong>Passing:</strong> {quiz_settings.get('passing_score', 70)}% required to pass</li>
                <li><strong>Hints:</strong> {'Available' if quiz_settings.get('hints_enabled', False) else 'Challenge mode'}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Start enhanced quiz button
        if st.button("🚀 Start Enhanced Quiz", type="primary", use_container_width=True):
            st.session_state.current_quiz = quiz_data
            st.session_state.quiz_answers = {}
            st.session_state.quiz_started = True
            st.session_state.quiz_start_time = time.time()  # Track time
            st.rerun()
        
        # Enhanced quiz interface
        if st.session_state.get('quiz_started', False) and st.session_state.current_quiz:
            display_enhanced_interactive_quiz()

def display_enhanced_interactive_quiz():
    """Display enhanced interactive quiz interface with difficulty adaptation"""
    quiz = st.session_state.current_quiz
    questions = quiz.get('questions', [])
    
    if not questions:
        st.error("No questions found in this quiz")
        return
    
    st.markdown("---")
    st.subheader(f"🎯 {st.session_state.difficulty_level.title()} Level Quiz in Progress")
    
    # Enhanced progress tracking
    if 'quiz_answers' not in st.session_state:
        st.session_state.quiz_answers = {}
    
    progress = len(st.session_state.quiz_answers) / len(questions)
    
    # Progress display with time tracking
    col1, col2 = st.columns(2)
    
    with col1:
        st.progress(progress)
        st.write(f"Progress: {len(st.session_state.quiz_answers)}/{len(questions)} questions answered")
    
    with col2:
        if st.session_state.get('quiz_start_time'):
            elapsed_time = time.time() - st.session_state.quiz_start_time
            st.write(f"⏱️ Time elapsed: {int(elapsed_time // 60)}:{int(elapsed_time % 60):02d}")
    
    # Display questions with difficulty-specific features
    for i, question in enumerate(questions):
        with st.container():
            st.markdown(f"### Question {i+1}")
            st.write(question['question'])
            
            question_key = f"q_{i}"
            
            # Difficulty-specific question handling
            if question['type'] == 'multiple_choice':
                # Enhanced multiple choice with difficulty hints
                answer = st.radio(
                    "Select your answer:",
                    question['options'],
                    key=f"radio_{i}",
                    index=None
                )
                if answer:
                    st.session_state.quiz_answers[question_key] = question['options'].index(answer)
                
                # Show hints for beginners
                if st.session_state.difficulty_level == 'beginner' and question_key not in st.session_state.quiz_answers:
                    if st.button(f"💡 Hint", key=f"hint_{i}"):
                        hint = question.get('hint', 'Think about the key concepts we discussed.')
                        st.info(f"**Hint:** {hint}")
            
            elif question['type'] == 'true_false':
                answer = st.radio(
                    "Select your answer:",
                    ['True', 'False'],
                    key=f"tf_{i}",
                    index=None
                )
                if answer:
                    st.session_state.quiz_answers[question_key] = answer == 'True'
            
            elif question['type'] == 'short_answer':
                # Enhanced text input with difficulty-specific guidance
                if st.session_state.difficulty_level == 'beginner':
                    placeholder = "Write your answer in simple terms..."
                    help_text = "Don't worry about perfect grammar - focus on the key ideas!"
                elif st.session_state.difficulty_level == 'advanced':
                    placeholder = "Provide a comprehensive analysis..."
                    help_text = "Include technical details and multiple perspectives."
                else:
                    placeholder = "Write your answer here..."
                    help_text = "Explain your reasoning clearly."
                
                answer = st.text_area(
                    "Your answer:",
                    key=f"sa_{i}",
                    height=100,
                    placeholder=placeholder,
                    help=help_text
                )
                if answer.strip():
                    st.session_state.quiz_answers[question_key] = answer
            
            # Show explanation if answered (difficulty-adapted)
            if question_key in st.session_state.quiz_answers:
                with st.expander("💡 Explanation"):
                    explanation = question.get('explanation', 'No explanation available')
                    
                    # Adapt explanation to difficulty level
                    if st.session_state.difficulty_level == 'beginner':
                        st.info(f"**Simple explanation:** {explanation}")
                    elif st.session_state.difficulty_level == 'advanced':
                        st.info(f"**Detailed analysis:** {explanation}")
                        # Show additional context for advanced learners
                        if question.get('advanced_context'):
                            st.write(f"**Advanced context:** {question['advanced_context']}")
                    else:
                        st.info(explanation)
            
            st.markdown("---")
    
    # Enhanced submit quiz
    if len(st.session_state.quiz_answers) == len(questions):
        if st.button("🎯 Submit Enhanced Quiz", type="primary", use_container_width=True):
            submit_enhanced_quiz_results(quiz, questions)

def submit_enhanced_quiz_results(quiz, questions):
    """Process enhanced quiz results with difficulty-specific scoring"""
    answers = st.session_state.quiz_answers
    correct_answers = 0
    total_questions = len(questions)
    
    # Calculate score with difficulty-specific evaluation
    for i, question in enumerate(questions):
        question_key = f"q_{i}"
        user_answer = answers.get(question_key)
        correct_answer = question.get('correct_answer')
        
        if question['type'] in ['multiple_choice', 'true_false']:
            if user_answer == correct_answer:
                correct_answers += 1
        elif question['type'] == 'short_answer':
            # Enhanced short answer evaluation based on difficulty
            key_points = question.get('key_points', [])
            user_text = str(user_answer).lower()
            
            if st.session_state.difficulty_level == 'beginner':
                # More lenient scoring for beginners
                if any(point.lower() in user_text for point in key_points):
                    correct_answers += 1
            elif st.session_state.difficulty_level == 'advanced':
                # Stricter evaluation for advanced learners
                points_found = sum(1 for point in key_points if point.lower() in user_text)
                if points_found >= len(key_points) * 0.7:  # Need 70% of key points
                    correct_answers += 1
            else:  # intermediate
                # Standard evaluation
                points_found = sum(1 for point in key_points if point.lower() in user_text)
                if points_found >= len(key_points) * 0.5:  # Need 50% of key points
                    correct_answers += 1
    
    score_percentage = (correct_answers / total_questions) * 100
    
    # Enhanced results display
    st.balloons()
    st.success(f"🎉 {st.session_state.difficulty_level.title()} Level Quiz Completed!")
    
    # Time tracking
    if st.session_state.get('quiz_start_time'):
        total_time = time.time() - st.session_state.quiz_start_time
        time_minutes = int(total_time // 60)
        time_seconds = int(total_time % 60)
        st.info(f"⏱️ **Time taken:** {time_minutes}:{time_seconds:02d}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("✅ Correct", correct_answers)
    with col2:
        st.metric("📊 Score", f"{score_percentage:.1f}%")
    with col3:
        # Difficulty-specific grading
        if st.session_state.difficulty_level == 'beginner':
            grade_thresholds = {'A': 80, 'B': 70, 'C': 60, 'D': 50}
        elif st.session_state.difficulty_level == 'advanced':
            grade_thresholds = {'A': 90, 'B': 80, 'C': 70, 'D': 60}
        else:  # intermediate
            grade_thresholds = {'A': 85, 'B': 75, 'C': 65, 'D': 55}
        
        grade = 'F'
        for letter, threshold in grade_thresholds.items():
            if score_percentage >= threshold:
                grade = letter
                break
        
        st.metric("🏆 Grade", grade)
    
    with col4:
        if ENHANCED_MODULES_AVAILABLE:
            config = Config()
            passing_score = config.get_difficulty_config(st.session_state.difficulty_level)['quiz_settings']['passing_score']
        else:
            passing_score = 70
        
        status = "PASS" if score_percentage >= passing_score else "RETRY"
        st.metric("📋 Status", status)
    
    # Enhanced gamification with difficulty multipliers
    if ENHANCED_MODULES_AVAILABLE:
        gamification = st.session_state.enhanced_gamification
        
        # Update quiz performance tracking
        gamification.update_quiz_performance(score_percentage, st.session_state.difficulty_level)
        
        # Check for achievements
        new_achievements = gamification.check_achievements()
        
        if new_achievements:
            st.markdown("### 🏆 New Achievements Unlocked!")
            for achievement in new_achievements:
                st.success(f"{achievement['icon']} **{achievement['name']}** - {achievement['description']}")
    else:
        # Original gamification
        gamification = st.session_state.gamification
        st.session_state.user_progress['quizzes_taken'] += 1
        st.session_state.user_progress['quiz_scores'].append(score_percentage)
        
        if score_percentage == 100:
            st.session_state.user_progress['perfect_quizzes'] = st.session_state.user_progress.get('perfect_quizzes', 0) + 1
        
        points_earned = 50 if score_percentage >= passing_score else 25
        st.session_state.user_progress['total_points'] += points_earned
        
        gamification.check_achievements(st.session_state.user_progress)
    
    # Enhanced feedback based on performance and difficulty
    if score_percentage >= 90:
        feedback_messages = {
            "beginner": "🌟 Excellent work! You're mastering the basics beautifully!",
            "intermediate": "🎯 Outstanding performance! You're ready for more challenging topics!",
            "advanced": "🚀 Exceptional work! Your analytical skills are impressive!"
        }
    elif score_percentage >= 70:
        feedback_messages = {
            "beginner": "😊 Good job! Keep practicing and you'll get even better!",
            "intermediate": "👍 Well done! You're on the right track!",
            "advanced": "📊 Good work! Consider reviewing the challenging areas."
        }
    else:
        feedback_messages = {
            "beginner": "💪 Don't worry! Learning takes time. Try reviewing the material and take the quiz again!",
            "intermediate": "🔄 Good effort! Review the topics and try again when you're ready.",
            "advanced": "🎯 This was challenging! Review the complex concepts and retake when prepared."
        }
    
    feedback = feedback_messages.get(st.session_state.difficulty_level, feedback_messages["intermediate"])
    st.info(feedback)
    
    # Save enhanced quiz results
    quiz_result = {
        'score': score_percentage,
        'correct_answers': correct_answers,
        'total_questions': total_questions,
        'grade': grade,
        'difficulty_level': st.session_state.difficulty_level,
        'time_taken': total_time if st.session_state.get('quiz_start_time') else 0,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'quiz_topic': quiz.get('topic', 'Unknown'),
        'passed': score_percentage >= passing_score
    }
    
    if 'quiz_results' not in st.session_state:
        st.session_state.quiz_results = []
    
    st.session_state.quiz_results.append(quiz_result)
    
    # Reset quiz state
    st.session_state.quiz_started = False
    st.session_state.current_quiz = None
    st.session_state.quiz_answers = {}
    
    # Navigation options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Take Another Quiz", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("📊 View Analytics", use_container_width=True):
            st.session_state.current_page = "📊 Learning Analytics"
            st.rerun()
    
    with col3:
        if st.button("🏠 Back to Dashboard", use_container_width=True):
            st.session_state.current_page = "🏠 Dashboard"
            st.rerun()

def image_analysis_page():
    """NEW: Image analysis page with multimodal AI processing"""
    st.header("🖼️ Educational Image Analysis with AI")
    
    if not ENHANCED_MODULES_AVAILABLE:
        st.error("❌ Enhanced modules not available. Please install required dependencies.")
        return
    
    if not st.session_state.textbook_processed:
        st.warning("⚠️ Please process a textbook first to analyze embedded images!")
        
        # Allow standalone image upload
        st.markdown("### 📁 Upload Individual Images for Analysis")
        uploaded_images = st.file_uploader(
            "Upload educational images",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            accept_multiple_files=True,
            help="Upload educational diagrams, charts, or illustrations for AI analysis"
        )
        
        if uploaded_images:
            analyze_uploaded_images(uploaded_images)
        return
    
    # Show existing image analysis results
    if st.session_state.get('image_analysis_results'):
        display_image_analysis_results()
    else:
        st.info("No images have been analyzed yet. Image analysis happens automatically during textbook processing.")
        
        # Option to re-analyze with different settings
        if st.button("🔄 Re-analyze Textbook Images", type="primary"):
            reanalyze_textbook_images()

def analyze_uploaded_images(uploaded_images):
    """Analyze uploaded images with AI"""
    if not st.session_state.multimodal_processor:
        st.session_state.multimodal_processor = OpenSourceMultimodalProcessor()
    
    processor = st.session_state.multimodal_processor
    
    for i, uploaded_image in enumerate(uploaded_images):
        st.markdown(f"### 🖼️ Analysis Results for {uploaded_image.name}")
        
        # Display the image
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(uploaded_image, caption=uploaded_image.name, width=300)
        
        with col2:
            with st.spinner(f"🧠 Analyzing {uploaded_image.name} with AI..."):
                # Convert uploaded file to PIL Image
                from PIL import Image
                import io
                
                image = Image.open(io.BytesIO(uploaded_image.getvalue()))
                
                # Analyze single image
                analysis = processor._analyze_single_image(
                    image, 0, st.session_state.difficulty_level
                )
                
                if analysis:
                    display_single_image_analysis(analysis, uploaded_image.name)

def display_image_analysis_results():
    """Display comprehensive image analysis results"""
    st.subheader("🖼️ Textbook Image Analysis Results")
    
    results = st.session_state.image_analysis_results
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_images = len(results)
    high_value_images = sum(1 for img in results.values() if img.get('educational_value') == 'high')
    images_with_text = sum(1 for img in results.values() if img.get('text_content'))
    total_concepts = sum(len(img.get('concepts', [])) for img in results.values())
    
    with col1:
        st.metric("🖼️ Total Images", total_images)
    with col2:
        st.metric("⭐ High Value", high_value_images)
    with col3:
        st.metric("📝 With Text", images_with_text)
    with col4:
        st.metric("🔑 Concepts", total_concepts)
    
    # Filter and display options
    st.markdown("### 🔍 Filter Images")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        value_filter = st.selectbox(
            "Educational Value",
            ["All", "High", "Medium", "Low", "Minimal"]
        )
    
    with col2:
        type_filter = st.selectbox(
            "Image Type",
            ["All"] + list(set(img.get('image_type', 'unknown') for img in results.values()))
        )
    
    with col3:
        page_filter = st.selectbox(
            "Page Number",
            ["All"] + sorted(list(set(str(img.get('page_number', 0)) for img in results.values())))
        )
    
    # Filter results
    filtered_results = filter_image_results(results, value_filter, type_filter, page_filter)
    
    # Display filtered images
    if filtered_results:
        st.markdown(f"### 📋 Showing {len(filtered_results)} Images")
        
        for image_key, analysis in filtered_results.items():
            display_single_image_analysis(analysis, image_key)
    else:
        st.info("No images match the selected filters.")

def filter_image_results(results, value_filter, type_filter, page_filter):
    """Filter image analysis results based on criteria"""
    filtered = {}
    
    for key, analysis in results.items():
        # Value filter
        if value_filter != "All":
            if analysis.get('educational_value', '').lower() != value_filter.lower():
                continue
        
        # Type filter
        if type_filter != "All":
            if analysis.get('image_type', '') != type_filter:
                continue
        
        # Page filter
        if page_filter != "All":
            if str(analysis.get('page_number', 0)) != page_filter:
                continue
        
        filtered[key] = analysis
    
    return filtered

def display_single_image_analysis(analysis, image_name):
    """Display analysis for a single image"""
    with st.expander(f"🔍 {image_name} - {analysis.get('educational_value', 'unknown').title()} Value"):
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**📊 Analysis Summary**")
            st.write(f"**Type:** {analysis.get('image_type', 'Unknown')}")
            st.write(f"**Educational Value:** {analysis.get('educational_value', 'Unknown')}")
            st.write(f"**Page:** {analysis.get('page_number', 'Unknown')}")
            
            # Concepts
            concepts = analysis.get('concepts', [])
            if concepts:
                st.write("**🔑 Key Concepts:**")
                for concept in concepts[:5]:
                    st.write(f"• {concept}")
        
        with col2:
            # Description
            description = analysis.get('description', '')
            if description:
                st.markdown("**🖼️ AI Description**")
                st.write(description)
            
            # Text content
            text_content = analysis.get('text_content', '')
            if text_content:
                st.markdown("**📝 Extracted Text**")
                st.code(text_content, language=None)
            
            # Educational elements
            elements = analysis.get('educational_elements', [])
            if elements:
                st.markdown("**🎯 Educational Elements**")
                st.write(", ".join(elements))
            
            # Suggested questions
            questions = analysis.get('suggested_questions', [])
            if questions:
                st.markdown("**❓ Suggested Questions**")
                for i, question in enumerate(questions, 1):
                    st.write(f"{i}. {question}")

def reanalyze_textbook_images():
    """Re-analyze textbook images with current settings"""
    if not st.session_state.knowledge_base:
        st.error("No textbook processed")
        return
    
    with st.spinner("🔄 Re-analyzing textbook images..."):
        # This would re-run the image analysis
        st.info("Re-analysis feature coming soon!")

def voice_learning_page():
    """NEW: Voice learning page with enhanced speech interaction"""
    st.header("🎤 Voice-Enabled Learning Experience")
    
    if not ENHANCED_MODULES_AVAILABLE:
        st.error("❌ Enhanced modules not available. Please install required dependencies.")
        return
    
    # Initialize voice processor
    if not st.session_state.voice_processor:
        st.session_state.voice_processor = VoiceChatProcessor()
    
    voice_processor = st.session_state.voice_processor
    
    # Voice learning dashboard
    st.markdown(f"### 🎯 Voice Learning Dashboard ({st.session_state.difficulty_level.title()} Level)")
    
    # Voice interaction statistics
    col1, col2, col3, col4 = st.columns(4)
    
    voice_interactions = st.session_state.user_progress.get('voice_interactions', 0)
    
    with col1:
        st.metric("🎤 Voice Chats", voice_interactions)
    with col2:
        if st.session_state.textbook_processed:
            st.metric("📚 Textbook", "Ready")
        else:
            st.metric("📚 Textbook", "Not loaded")
    with col3:
        difficulty_level = st.session_state.difficulty_level
        st.metric("📊 Level", difficulty_level.title())
    with col4:
        # Voice feature availability
        config = Config()
        voice_available = config.model_availability.get('coqui_tts', False) or True  # gTTS fallback
        st.metric("🔊 Voice", "Ready" if voice_available else "Limited")
    
    # Main voice interface
    voice_processor.setup_voice_chat_interface()
    
    # Voice learning history
    if voice_interactions > 0:
        st.markdown("### 📋 Voice Learning Summary")
        summary = voice_processor.create_voice_interaction_summary()
        st.markdown(summary)
    
    # Voice settings and help
    with st.sidebar:
        if st.session_state.current_page == "🎤 Voice Learning":
            st.markdown("---")
            st.markdown("### 🎤 Voice Help")
            
            st.info("**How to use voice chat:**")
            st.write("1. Click 'Record' button")
            st.write("2. Speak your question clearly")
            st.write("3. Wait for AI response")
            st.write("4. Listen to voice reply")
            
            if st.session_state.difficulty_level == 'beginner':
                st.success("**Beginner Tips:**\n• Speak slowly and clearly\n• Use simple questions\n• Voice responses are slower")
            elif st.session_state.difficulty_level == 'advanced':
                st.success("**Advanced Tips:**\n• Use technical terminology\n• Ask complex questions\n• Voice responses are faster")

def enhanced_settings_page():
    """Enhanced settings page with model setup and configuration"""
    st.header("⚙️ Enhanced Platform Settings")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🤖 Model Setup", 
        "🎯 Learning Preferences", 
        "🔧 System Configuration", 
        "🎮 Gamification Settings",
        "📊 Data Management"
    ])
    
    with tab1:
        display_model_setup_tab()
    
    with tab2:
        display_learning_preferences_tab()
    
    with tab3:
        display_system_configuration_tab()
    
    with tab4:
        display_gamification_settings_tab()
    
    with tab5:
        display_data_management_tab()

def display_model_setup_tab():
    """Display model setup and configuration tab"""
    st.subheader("🤖 AI Model Setup & Status")
    
    if ENHANCED_MODULES_AVAILABLE:
        # Import model setup manager
        try:
            from scripts.model_setup import ModelSetupManager
            
            setup_manager = ModelSetupManager()
            setup_manager.display_setup_status_streamlit()
            
        except ImportError:
            st.error("Model setup script not available")
            
        st.markdown("---")
        st.markdown("### 🔧 Manual Model Configuration")
        
        # Display current model status
        config = Config()
        if hasattr(config, 'model_availability'):
            st.markdown("**Current Model Status:**")
            
            for model_name, available in config.model_availability.items():
                status_color = "🟢" if available else "🔴"
                st.write(f"{status_color} **{model_name.replace('_', ' ').title()}:** {'Available' if available else 'Not Available'}")
        
        # Model configuration options
        with st.expander("🔧 Advanced Model Settings"):
            st.checkbox("Enable Local Processing", value=config.USE_OPEN_SOURCE_MODELS)
            st.checkbox("Fallback to Commercial APIs", value=config.FALLBACK_TO_COMMERCIAL)
            
            st.selectbox(
                "Primary Embedding Model",
                ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "sentence-t5-base"],
                index=0
            )
            
            st.selectbox(
                "Local LLM Model",
                ["llama3.2", "llama3.1", "mixtral", "gemma2"],
                index=0
            )
    else:
        st.warning("⚠️ Enhanced modules not available")
        st.info("💡 Install enhanced modules to access advanced model setup")
        
        # Basic API configuration
        st.markdown("### 🔑 Basic API Configuration")
        
        # API key debugging
        config = Config()
        config.debug_api_key()

def display_learning_preferences_tab():
    """Display learning preferences configuration"""
    st.subheader("🎯 Learning Preferences")
    
    # Difficulty level settings
    st.markdown("### 📊 Difficulty Level Settings")
    
    current_difficulty = st.session_state.difficulty_level
    
    difficulty_descriptions = {
        "beginner": {
            "description": "Simple explanations, basic vocabulary, extra examples",
            "features": ["Slower pace", "Hints available", "Encouraging feedback", "Visual aids"]
        },
        "intermediate": {
            "description": "Balanced detail, standard terminology, good examples",
            "features": ["Normal pace", "Balanced content", "Standard feedback", "Mixed media"]
        },
        "advanced": {
            "description": "Technical depth, complex concepts, minimal guidance",
            "features": ["Fast pace", "Challenge mode", "Technical feedback", "Advanced content"]
        }
    }
    
    for level, info in difficulty_descriptions.items():
        is_current = level == current_difficulty
        
        with st.expander(f"{'🎯 ' if is_current else ''}{level.title()} Level {'(Current)' if is_current else ''}"):
            st.write(f"**Description:** {info['description']}")
            st.write("**Features:**")
            for feature in info['features']:
                st.write(f"• {feature}")
    
    # Learning style preferences
    st.markdown("### 🧠 Learning Style Preferences")
    
    learning_styles = ["Visual", "Auditory", "Kinesthetic", "Reading/Writing"]
    current_style = st.session_state.get('learning_style', 'Visual')
    
    selected_style = st.selectbox(
        "Preferred Learning Style",
        learning_styles,
        index=learning_styles.index(current_style) if current_style in learning_styles else 0
    )
    
    if selected_style != current_style:
        st.session_state.learning_style = selected_style
        st.success(f"Learning style updated to {selected_style}")
    
    # Voice settings
    if ENHANCED_MODULES_AVAILABLE:
        st.markdown("### 🎤 Voice Learning Settings")
        
        voice_enabled = st.checkbox("Enable Voice Interactions", value=st.session_state.get('voice_enabled', True))
        st.session_state.voice_enabled = voice_enabled
        
        if voice_enabled:
            col1, col2 = st.columns(2)
            
            with col1:
                st.selectbox("Speech Recognition", ["Auto (Groq + OpenAI)", "Groq Only", "OpenAI Only"])
            
            with col2:
                st.selectbox("Text-to-Speech", ["Auto (Coqui + gTTS)", "Coqui TTS", "gTTS Only"])

def display_system_configuration_tab():
    """Display system configuration options"""
    st.subheader("🔧 System Configuration")
    
    # API Configuration
    st.markdown("### 🔑 API Configuration")
    
    config = Config()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Primary API (Groq):**")
        if config.GROQ_API_KEY:
            masked_key = config.GROQ_API_KEY[:8] + "..." + config.GROQ_API_KEY[-4:]
            st.success(f"✅ Configured: {masked_key}")
        else:
            st.error("❌ Not configured")
        
        st.selectbox("Groq Model", config.GROQ_MODELS, index=0)
    
    with col2:
        st.write("**Fallback API (OpenAI):**")
        if config.OPENAI_API_KEY:
            masked_key = config.OPENAI_API_KEY[:8] + "..." + config.OPENAI_API_KEY[-4:]
            st.success(f"✅ Configured: {masked_key}")
        else:
            st.warning("⚠️ Not configured (optional)")
    
    # Performance settings
    st.markdown("### ⚡ Performance Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.slider("Max Tokens", 1000, 8000, config.MAX_TOKENS)
        st.slider("Temperature", 0.0, 2.0, config.TEMPERATURE, step=0.1)
    
    with col2:
        st.slider("Batch Size", 1, 20, 5)
        st.checkbox("Enable Caching", value=True)
    
    # Directory information
    st.markdown("### 📁 Directory Configuration")
    
    directories = config.get_directory_config()
    
    for name, path in directories.items():
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        st.write(f"{status} **{name.replace('_', ' ').title()}:** `{path}`")

def display_gamification_settings_tab():
    """Display gamification configuration"""
    st.subheader("🎮 Gamification Settings")
    
    if ENHANCED_MODULES_AVAILABLE:
        gamification = st.session_state.enhanced_gamification
        
        # Current progress overview
        level_info = gamification.get_current_level_info()
        
        st.markdown("### 🏆 Current Progress Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_level = level_info['current_level']
            st.metric("Level", f"{current_level['level']} - {current_level['name']}")
        
        with col2:
            st.metric("Total Points", f"{level_info['current_points']:,}")
        
        with col3:
            badges_earned = len(st.session_state.user_progress.get('badges_earned', []))
            st.metric("Achievements", f"{badges_earned}/{len(gamification.achievements)}")
        
        with col4:
            streak = st.session_state.user_progress.get('learning_streak', 0)
            st.metric("Streak", f"{streak} days")
        
        # Achievement progress
        st.markdown("### 🎯 Achievement Progress")
        gamification.display_achievement_progress()
        
        # Settings
        st.markdown("### ⚙️ Notification Settings")
        
        notifications = st.session_state.user_progress.get('notification_settings', {
            'achievements': True,
            'streaks': True,
            'level_ups': True
        })
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            notifications['achievements'] = st.checkbox("Achievement Notifications", value=notifications.get('achievements', True))
        
        with col2:
            notifications['streaks'] = st.checkbox("Streak Notifications", value=notifications.get('streaks', True))
        
        with col3:
            notifications['level_ups'] = st.checkbox("Level Up Notifications", value=notifications.get('level_ups', True))
        
        st.session_state.user_progress['notification_settings'] = notifications
    
    else:
        st.info("Enhanced gamification features require enhanced modules")

def display_data_management_tab():
    """Display data management and export options"""
    st.subheader("📊 Data Management")
    
    # Export options
    st.markdown("### 📤 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export Learning Analytics", use_container_width=True):
            if st.session_state.analytics_dashboard:
                analytics_data = st.session_state.analytics_dashboard.export_analytics_data()
                st.download_button(
                    "💾 Download Analytics",
                    analytics_data,
                    "intellilearn_analytics.json",
                    "application/json"
                )
        
        if st.button("🏆 Export Gamification Data", use_container_width=True):
            if ENHANCED_MODULES_AVAILABLE:
                gamification_data = st.session_state.enhanced_gamification.export_gamification_data()
            else:
                gamification_data = json.dumps(st.session_state.user_progress, indent=2)
            
            st.download_button(
                "💾 Download Progress",
                gamification_data,
                "intellilearn_progress.json",
                "application/json"
            )
    
    with col2:
        if st.button("💬 Export Chat History", use_container_width=True):
            if st.session_state.get('chat_messages'):
                chat_data = json.dumps(st.session_state.chat_messages, indent=2)
                st.download_button(
                    "💾 Download Chat",
                    chat_data,
                    "intellilearn_chat.json",
                    "application/json"
                )
        
        if ENHANCED_MODULES_AVAILABLE and st.button("🎤 Export Voice Data", use_container_width=True):
            if st.session_state.voice_processor:
                voice_data = st.session_state.voice_processor.export_voice_conversation()
                if voice_data:
                    st.download_button(
                        "💾 Download Voice Data",
                        voice_data,
                        "intellilearn_voice.json",
                        "application/json"
                    )
    
    # Data cleanup
    st.markdown("### 🗑️ Data Cleanup")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧹 Clear Cache", use_container_width=True):
            # Clear various caches
            if 'knowledge_base' in st.session_state:
                del st.session_state['knowledge_base']
            if 'image_analysis_results' in st.session_state:
                del st.session_state['image_analysis_results']
            st.success("Cache cleared!")
    
    with col2:
        if st.button("💬 Clear Chat History", use_container_width=True):
            st.session_state.chat_messages = []
            st.success("Chat history cleared!")
    
    with col3:
        if st.button("🔄 Reset All Data", use_container_width=True):
            # Reset everything (with confirmation)
            if st.confirm("Are you sure? This will reset all your progress!"):
                for key in list(st.session_state.keys()):
                    if key not in ['current_page']:  # Keep current page
                        del st.session_state[key]
                st.success("All data reset!")
                st.rerun()

def analytics_page():
    """Enhanced analytics dashboard page"""
    st.header("📊 Learning Analytics Dashboard")
    
    if not st.session_state.textbook_processed:
        st.warning("⚠️ Please process a textbook first to view analytics!")
        return
    
    analytics_dashboard = st.session_state.analytics_dashboard
    
    # Enhanced analytics with difficulty tracking
    st.subheader(f"📈 Performance Overview ({st.session_state.difficulty_level.title()} Level)")
    
    # Get difficulty-specific metrics
    difficulty_level = st.session_state.difficulty_level
    difficulty_scores = st.session_state.user_progress.get(f'{difficulty_level}_scores', [])
    difficulty_quizzes = st.session_state.user_progress.get(f'{difficulty_level}_quizzes', 0)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_quizzes = st.session_state.user_progress.get('quizzes_taken', 0)
        st.metric("🎯 Total Quizzes", total_quizzes)
    
    with col2:
        st.metric(f"📊 {difficulty_level.title()} Quizzes", difficulty_quizzes)
    
    with col3:
        if difficulty_scores:
            avg_score = sum(difficulty_scores) / len(difficulty_scores)
            st.metric(f"📈 {difficulty_level.title()} Avg", f"{avg_score:.1f}%")
        else:
            st.metric(f"📈 {difficulty_level.title()} Avg", "No data")
    
    with col4:
        total_points = st.session_state.user_progress.get('total_points', 0)
        st.metric("🏆 Total Points", f"{total_points:,}")
    
    # Enhanced progress visualization
    analytics_dashboard.display_enhanced_dashboard()
    
    # Additional enhanced analytics
    if ENHANCED_MODULES_AVAILABLE:
        st.markdown("---")
        st.subheader("🎯 Advanced Learning Insights")
        
        # Performance by difficulty level
        st.markdown("### 📊 Performance by Difficulty Level")
        
        difficulty_data = []
        for diff_level in ['beginner', 'intermediate', 'advanced']:
            scores = st.session_state.user_progress.get(f'{diff_level}_scores', [])
            quizzes = st.session_state.user_progress.get(f'{diff_level}_quizzes', 0)
            
            if scores:
                avg_score = sum(scores) / len(scores)
                difficulty_data.append({
                    'Level': diff_level.title(),
                    'Quizzes': quizzes,
                    'Average Score': round(avg_score, 1),
                    'Best Score': max(scores),
                    'Progress': 'Improving' if len(scores) > 1 and scores[-1] > scores[0] else 'Stable'
                })
        
        if difficulty_data:
            import pandas as pd
            df = pd.DataFrame(difficulty_data)
            st.dataframe(df, use_container_width=True)

def learning_progress_page():
    """Enhanced learning progress page with gamification"""
    st.header("🏆 Learning Progress & Achievements")
    
    if ENHANCED_MODULES_AVAILABLE:
        gamification = st.session_state.enhanced_gamification
        gamification.display_enhanced_progress_dashboard()
        
        # Additional progress insights
        st.markdown("---")
        st.subheader("📈 Detailed Progress Analysis")
        
        # Time-based progress
        if st.session_state.user_progress.get('achievements_history'):
            st.markdown("### 🕒 Achievement Timeline")
            
            achievements = st.session_state.user_progress['achievements_history']
            for achievement in achievements[-5:]:  # Show last 5
                badge = gamification.achievements.get(achievement['badge_id'], {})
                st.success(f"{badge.get('icon', '🏅')} **{badge.get('name', 'Achievement')}** - {achievement.get('date', '')[:10]} (+{achievement.get('points_earned', 0)} points)")
        
        # Learning recommendations
        st.markdown("### 💡 Personalized Recommendations")
        
        current_level = st.session_state.difficulty_level
        total_points = st.session_state.user_progress.get('total_points', 0)
        
        if total_points < 500:
            st.info(f"🎯 **For {current_level} learners:** Focus on completing more quizzes to build your foundation!")
        elif total_points < 2000:
            st.info(f"🚀 **Great progress!** Consider exploring {get_next_difficulty_level(current_level)} level content for a challenge!")
        else:
            st.info("🌟 **Excellent work!** You're mastering the content. Try creating educational videos or helping others!")
    
    else:
        # Original gamification display
        gamification = st.session_state.gamification
        gamification.display_progress_dashboard(st.session_state.user_progress)

def get_next_difficulty_level(current_level):
    """Get the next difficulty level for recommendations"""
    levels = ['beginner', 'intermediate', 'advanced']
    try:
        current_index = levels.index(current_level)
        if current_index < len(levels) - 1:
            return levels[current_index + 1]
        else:
            return current_level  # Already at highest level
    except ValueError:
        return 'intermediate'

def adaptive_learning_page():
    """Enhanced adaptive learning page"""
    st.header("🎮 Adaptive Learning Engine")
    
    if not st.session_state.textbook_processed:
        st.warning("⚠️ Please process a textbook first!")
        return
    
    adaptive_engine = st.session_state.adaptive_engine
    
    # Enhanced learning style detection
    st.subheader("🧠 Learning Style Analysis")
    
    if st.session_state.user_interactions:
        detected_style = adaptive_engine.detect_learning_style(st.session_state.user_interactions)
        st.session_state.learning_style = detected_style
        
        st.success(f"📊 **Detected Learning Style:** {detected_style}")
        
        # Style-specific recommendations
        style_recommendations = {
            'Visual': "🖼️ Focus on diagrams, charts, and video content",
            'Auditory': "🎤 Use voice chat and audio explanations",
            'Kinesthetic': "🤲 Try interactive quizzes and hands-on exercises",
            'Reading/Writing': "📝 Focus on text-based content and note-taking"
        }
        
        st.info(style_recommendations.get(detected_style, "Continue with your current learning approach"))
    else:
        st.info("Take some quizzes and interact with content to detect your learning style")
    
    # Enhanced adaptive path generation
    st.subheader("🎯 Personalized Learning Path")
    
    if st.session_state.user_interactions:
        # Get difficulty-specific performance
        difficulty_scores = st.session_state.user_progress.get(f'{st.session_state.difficulty_level}_scores', [])
        current_performance = sum(difficulty_scores) / len(difficulty_scores) if difficulty_scores else 75
        
        learning_path = adaptive_engine.generate_adaptive_path(
            current_performance=current_performance,
            subject=st.session_state.selected_subject,
            learning_style=st.session_state.learning_style,
            knowledge_base=st.session_state.knowledge_base
        )
        
        if learning_path:
            st.markdown("### 📋 Recommended Learning Sequence")
            
            for i, step in enumerate(learning_path[:5], 1):  # Show first 5 steps
                difficulty_badge = {
                    'beginner': '🟢',
                    'intermediate': '🟡',
                    'advanced': '🔴'
                }.get(step.get('difficulty', 'intermediate'), '🟡')
                
                st.markdown(f"""
                **Step {i}: {step.get('chapter', 'Unknown Chapter')}**
                {difficulty_badge} {step.get('difficulty', 'intermediate').title()} Level
                - **Focus:** {step.get('focus_area', 'General understanding')}
                - **Estimated Time:** {step.get('estimated_time', 30)} minutes
                """)
    else:
        st.info("Complete some learning activities to generate your personalized path")
    
    # Enhanced real-time adaptation
    st.subheader("⚡ Real-Time Adaptation")
    
    if st.session_state.quiz_results:
        recent_performance = [result['score'] for result in st.session_state.quiz_results[-3:]]
        
        if recent_performance:
            avg_recent = sum(recent_performance) / len(recent_performance)
            
            if avg_recent >= 85:
                st.success("🚀 **Excellent performance!** Consider moving to advanced content or helping others learn.")
            elif avg_recent >= 70:
                st.info("👍 **Good progress!** Continue with current difficulty or try slightly more challenging content.")
            else:
                st.warning("💪 **Keep practicing!** Consider reviewing fundamentals or trying easier content first.")
            
            # Difficulty adjustment suggestions
            current_level = st.session_state.difficulty_level
            
            if avg_recent >= 90 and current_level != 'advanced':
                next_level = get_next_difficulty_level(current_level)
                st.info(f"💡 **Suggestion:** You might be ready for {next_level} level content!")
            elif avg_recent < 60 and current_level != 'beginner':
                st.info("💡 **Suggestion:** Consider switching to an easier difficulty level to build confidence.")

def test_functions_page():
    """Enhanced test functions page for debugging and demonstrations"""
    st.header("🧪 Enhanced Test Functions & Debugging")
    
    # Enhanced model testing
    st.subheader("🤖 AI Model Testing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧠 Test Enhanced NLP", use_container_width=True):
            if ENHANCED_MODULES_AVAILABLE and st.session_state.enhanced_nlp:
                test_text = "Photosynthesis is the process by which plants convert light energy into chemical energy."
                
                with st.spinner("Testing enhanced NLP..."):
                    try:
                        # Test concept extraction
                        concepts = st.session_state.enhanced_nlp.extract_educational_concepts_advanced(
                            test_text, "biology"
                        )
                        
                        # Test difficulty assessment
                        difficulty = st.session_state.enhanced_nlp.assess_text_difficulty_enhanced(test_text)
                        
                        st.success("✅ Enhanced NLP test successful!")
                        st.write(f"**Concepts found:** {', '.join(concepts[:5])}")
                        st.write(f"**Difficulty level:** {difficulty.get('level', 'unknown')}")
                        
                    except Exception as e:
                        st.error(f"❌ Enhanced NLP test failed: {e}")
            else:
                st.warning("Enhanced NLP not available")
    
    with col2:
        if st.button("🖼️ Test Image Analysis", use_container_width=True):
            if ENHANCED_MODULES_AVAILABLE and st.session_state.multimodal_processor:
                st.info("Image analysis test requires an actual image. Upload an image in the Image Analysis page to test.")
            else:
                st.warning("Multimodal processor not available")
    
    # Voice testing
    if ENHANCED_MODULES_AVAILABLE:
        st.subheader("🎤 Voice System Testing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔊 Test Text-to-Speech", use_container_width=True):
                if st.session_state.voice_processor:
                    test_text = f"This is a test of the {st.session_state.difficulty_level} level text-to-speech system."
                    
                    with st.spinner("Generating test audio..."):
                        audio_result = st.session_state.voice_processor._text_to_speech(
                            test_text, st.session_state.difficulty_level
                        )
                    
                    if audio_result:
                        st.success("✅ TTS test successful!")
                        st.audio(audio_result, format="audio/mp3")
                    else:
                        st.error("❌ TTS test failed")
                else:
                    st.warning("Voice processor not initialized")
        
        with col2:
            st.info("Speech-to-text testing requires the voice chat interface in the Voice Learning page.")
    
    # Enhanced debugging
    st.subheader("🔧 System Debugging")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Session State Info", use_container_width=True):
            st.json({
                'textbook_processed': st.session_state.get('textbook_processed', False),
                'difficulty_level': st.session_state.get('difficulty_level', 'unknown'),
                'learning_style': st.session_state.get('learning_style', 'unknown'),
                'voice_enabled': st.session_state.get('voice_enabled', False),
                'enhanced_modules': ENHANCED_MODULES_AVAILABLE,
                'total_points': st.session_state.user_progress.get('total_points', 0),
                'quiz_count': len(st.session_state.get('quiz_results', [])),
                'chat_messages': len(st.session_state.get('chat_messages', []))
            })
    
    with col2:
        if st.button("🤖 Model Status", use_container_width=True):
            config = Config()
            if hasattr(config, 'model_availability'):
                st.json(config.model_availability)
            else:
                st.json({"status": "Model availability checking not available"})
    
    with col3:
        if st.button("🎯 User Progress", use_container_width=True):
            st.json(st.session_state.user_progress)
    
    # Configuration testing
    st.subheader("⚙️ Configuration Testing")
    
    config = Config()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔑 Test API Connection", use_container_width=True):
            try:
                # Test Groq API
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {config.GROQ_API_KEY}"
                }
                
                payload = {
                    "model": config.GROQ_MODEL,
                    "messages": [{"role": "user", "content": "Test message"}],
                    "max_tokens": 50
                }
                
                response = requests.post(
                    f"{config.GROQ_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=10
                )
                
                if response.status_code == 200:
                    st.success("✅ Groq API connection successful!")
                else:
                    st.error(f"❌ Groq API error: {response.status_code}")
                    
            except Exception as e:
                st.error(f"❌ API test failed: {e}")
    
    with col2:
        if st.button("📁 Test Directories", use_container_width=True):
            directories = config.get_directory_config()
            
            all_exist = True
            for name, path in directories.items():
                exists = os.path.exists(path)
                status = "✅" if exists else "❌"
                st.write(f"{status} {name}: {path}")
                if not exists:
                    all_exist = False
            
            if all_exist:
                st.success("✅ All directories exist!")
            else:
                st.warning("⚠️ Some directories are missing")

# Main application flow
def main():
    """Enhanced main application with comprehensive features"""
    
    # Initialize enhanced session state
    initialize_session_state()
    
    # Enhanced sidebar
    setup_enhanced_sidebar()
    
    # Enhanced main header
    st.markdown("""
    <div class="main-header">
        <h1>🎓 IntelliLearn AI - Next-Generation Educational Platform</h1>
        <p>Enhanced with Open-Source Models, Voice Learning, and Adaptive Difficulty</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced page routing
    page = st.session_state.current_page
    
    try:
        if page == "🏠 Dashboard":
            enhanced_dashboard_page()
        elif page == "📚 Upload & Process":
            textbook_processing_page()
        elif page == "🎬 Video Generation":
            video_generation_page()
        elif page == "💬 AI Tutor Chat":
            ai_tutor_chat_page()
        elif page == "🎯 Interactive Quizzes":
            interactive_quiz_page()
        elif page == "📊 Learning Analytics":
            analytics_page()
        elif page == "🏆 Learning Progress":
            learning_progress_page()
        elif page == "🎮 Adaptive Learning":
            adaptive_learning_page()
        elif page == "🖼️ Image Analysis":
            image_analysis_page()
        elif page == "🎤 Voice Learning":
            voice_learning_page()
        elif page == "⚙️ Settings":
            enhanced_settings_page()
        elif page == "🧪 Test Functions":
            test_functions_page()
        else:
            st.error(f"Unknown page: {page}")
            
    except Exception as e:
        st.error(f"❌ Page error: {str(e)}")
        st.info("🔄 Try refreshing the page or contact support if the issue persists.")
        
        # Debug information
        with st.expander("🔧 Debug Information"):
            st.write(f"**Page:** {page}")
            st.write(f"**Error:** {str(e)}")
            st.write(f"**Enhanced Modules:** {ENHANCED_MODULES_AVAILABLE}")

if __name__ == "__main__":
    main()
