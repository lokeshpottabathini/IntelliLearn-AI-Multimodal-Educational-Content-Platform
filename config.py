import os
import streamlit as st
from pathlib import Path
import json
from typing import Dict, List, Optional

class Config:
    """Enhanced Configuration class for IntelliLearn AI Educational Platform with Open-Source Models"""
    
    # ===============================
    # API Configuration
    # ===============================
    
    # Groq API Configuration (Primary)
    GROQ_API_KEY = st.secrets.get('GROQ_API_KEY', '') or os.getenv('GROQ_API_KEY', '')
    GROQ_BASE_URL = "https://api.groq.com/openai/v1"
    GROQ_MODEL = "llama-3.3-70b-versatile"
    
    # Alternative Models
    GROQ_MODELS = [
        "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768", 
        "llama3-70b-8192",
        "gemma2-9b-it"
    ]
    
    # OpenAI Configuration (Fallback)
    OPENAI_API_KEY = st.secrets.get('OPENAI_API_KEY', '') or os.getenv('OPENAI_API_KEY', '')
    OPENAI_MODEL = "gpt-4"
    
    # ===============================
    # Open Source Model Configuration (NEW)
    # ===============================
    
    # Enable open-source models
    USE_OPEN_SOURCE_MODELS = True
    FALLBACK_TO_COMMERCIAL = True
    MODEL_FALLBACK_ORDER = ["open_source", "groq", "openai"]
    
    # Primary embedding model
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384
    
    # Local LLM configuration
    LOCAL_LLM_MODEL = "llama3.2"
    LOCAL_LLM_ENABLED = True
    OLLAMA_BASE_URL = "http://localhost:11434"
    
    # Enhanced NLP Models
    TOPIC_MODELING_ENABLED = True
    BERTOPIC_MODEL_NAME = "bertopic"
    ADVANCED_NER_ENABLED = True
    SPACY_MODEL = "en_core_web_sm"
    
    # Image understanding models
    IMAGE_CAPTIONING_MODEL = "Salesforce/blip-image-captioning-base"
    VISION_LANGUAGE_MODEL = "ViT-B/32"
    CLIP_MODEL_ENABLED = True
    
    # Text-to-Speech configuration
    TTS_MODEL = "tts_models/en/ljspeech/tacotron2-DDC_ph"
    USE_COQUI_TTS = True
    TTS_FALLBACK_TO_GTTS = True
    
    # Document analysis models
    LAYOUTPARSER_ENABLED = True
    UNSTRUCTURED_ENABLED = True
    ENHANCED_CHAPTER_DETECTION = True
    
    # Performance settings
    MAX_LOCAL_MODEL_MEMORY = 4096  # MB
    MODEL_CACHE_ENABLED = True
    BATCH_PROCESSING_ENABLED = True
    
    # ===============================
    # Token Limits & Model Parameters
    # ===============================
    
    MAX_TOKENS = 4096
    MAX_COMPLETION_TOKENS = 2048
    TEMPERATURE = 0.7
    TOP_P = 0.9
    FREQUENCY_PENALTY = 0.0
    PRESENCE_PENALTY = 0.0
    
    # Context Window Limits (Updated)
    CONTEXT_WINDOW_LIMITS = {
        "llama-3.3-70b-versatile": 32768,
        "mixtral-8x7b-32768": 32768,
        "llama3-70b-8192": 8192,
        "gemma2-9b-it": 8192,
        "llama3.2": 8192,  # Local Ollama model
    }
    
    # ===============================
    # Directory Configuration (Enhanced)
    # ===============================
    
    # Base directories
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "output"
    TEMP_DIR = BASE_DIR / "temp"
    CACHE_DIR = BASE_DIR / "cache"
    
    # Video Generation
    VIDEO_OUTPUT_DIR = str(BASE_DIR / "output" / "videos")
    SLIDES_OUTPUT_DIR = str(BASE_DIR / "output" / "slides")
    AUDIO_OUTPUT_DIR = str(BASE_DIR / "output" / "audio")
    
    # Text Processing
    TEXTBOOK_UPLOAD_DIR = str(BASE_DIR / "temp" / "textbooks")
    PROCESSED_DATA_DIR = str(BASE_DIR / "data" / "processed")
    
    # Model Cache (NEW)
    MODEL_CACHE_DIR = str(BASE_DIR / "cache" / "models")
    EMBEDDING_CACHE_DIR = str(BASE_DIR / "cache" / "embeddings")
    IMAGE_CACHE_DIR = str(BASE_DIR / "cache" / "images")
    
    # Backup and logs
    BACKUP_DIR = str(BASE_DIR / "backup")
    LOG_DIR = str(BASE_DIR / "logs")
    
    # Templates and assets
    TEMPLATES_DIR = str(BASE_DIR / "templates")
    ASSETS_DIR = str(BASE_DIR / "assets")
    FONTS_DIR = str(BASE_DIR / "assets" / "fonts")
    
    # ===============================
    # Video Generation Configuration (Enhanced)
    # ===============================
    
    # Video settings
    VIDEO_RESOLUTION = (1920, 1080)
    VIDEO_FPS = 30
    VIDEO_DURATION_MIN = 3
    VIDEO_DURATION_MAX = 20  # Increased for advanced content
    
    # Quality presets by difficulty
    VIDEO_QUALITY_PRESETS = {
        'beginner': {'resolution': (1280, 720), 'fps': 24, 'bitrate': '2M'},
        'intermediate': {'resolution': (1920, 1080), 'fps': 30, 'bitrate': '4M'},
        'advanced': {'resolution': (1920, 1080), 'fps': 30, 'bitrate': '6M'}
    }
    
    # Audio settings (Enhanced)
    AUDIO_SAMPLE_RATE = 44100
    AUDIO_CHANNELS = 2
    AUDIO_BITRATE = '128k'
    
    # Voice settings by difficulty level
    VOICE_SETTINGS = {
        'beginner': {'speed': 0.8, 'pitch': 'normal', 'clarity': 'high'},
        'intermediate': {'speed': 1.0, 'pitch': 'normal', 'clarity': 'normal'},
        'advanced': {'speed': 1.2, 'pitch': 'normal', 'clarity': 'normal'}
    }
    
    # TTS Voice options
    TTS_VOICES = {
        'coqui': ['tacotron2-DDC_ph', 'glow-tts'],
        'openai': ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'],
        'gtts': ['en']
    }
    
    # Image and slide generation
    SLIDE_TEMPLATE_PATH = str(BASE_DIR / "templates" / "slide_template.png")
    FONT_PATH = str(BASE_DIR / "assets" / "fonts" / "Arial.ttf")
    SLIDE_BACKGROUNDS = {
        'beginner': '#f0f8ff',
        'intermediate': '#f8f8ff',
        'advanced': '#f5f5f5'
    }
    
    # ===============================
    # Educational Platform Settings (Enhanced)
    # ===============================
    
    # Difficulty Level Configuration (NEW)
    DIFFICULTY_LEVELS = ['beginner', 'intermediate', 'advanced']
    DEFAULT_DIFFICULTY = 'intermediate'
    
    # Learning analytics
    DEFAULT_READING_SPEED = 200
    READING_SPEEDS = {
        'beginner': 150,
        'intermediate': 200,
        'advanced': 250
    }
    
    # Quiz configuration by difficulty
    QUIZ_SETTINGS = {
        'beginner': {
            'passing_score': 60,
            'questions_per_quiz': 3,
            'max_options': 3,
            'time_limit': 300,  # 5 minutes
            'hints_enabled': True
        },
        'intermediate': {
            'passing_score': 70,
            'questions_per_quiz': 5,
            'max_options': 4,
            'time_limit': 600,  # 10 minutes
            'hints_enabled': False
        },
        'advanced': {
            'passing_score': 80,
            'questions_per_quiz': 7,
            'max_options': 5,
            'time_limit': 900,  # 15 minutes
            'hints_enabled': False
        }
    }
    
    # Enhanced Achievement System (NEW)
    ACHIEVEMENT_CATEGORIES = {
        'milestone': {'name': 'Milestones', 'color': '#4CAF50', 'icon': '🎯'},
        'excellence': {'name': 'Excellence', 'color': '#FFD700', 'icon': '⭐'},
        'consistency': {'name': 'Consistency', 'color': '#FF6B6B', 'icon': '🔥'},
        'engagement': {'name': 'Engagement', 'color': '#4ECDC4', 'icon': '💬'},
        'mastery': {'name': 'Mastery', 'color': '#A8E6CF', 'icon': '🏆'},
        'difficulty': {'name': 'Difficulty', 'icon': '📊'}
    }
    
    # Points system with difficulty multipliers (Enhanced)
    POINTS_SYSTEM = {
        'textbook_processed': {'base': 100, 'beginner': 1.0, 'intermediate': 1.2, 'advanced': 1.5},
        'quiz_completed': {'base': 50, 'beginner': 1.0, 'intermediate': 1.3, 'advanced': 1.6},
        'perfect_quiz': {'base': 300, 'beginner': 1.0, 'intermediate': 1.5, 'advanced': 2.0},
        'chapter_completed': {'base': 100, 'beginner': 1.0, 'intermediate': 1.2, 'advanced': 1.4},
        'video_generated': {'base': 75, 'beginner': 1.0, 'intermediate': 1.1, 'advanced': 1.3},
        'voice_interaction': {'base': 10, 'beginner': 1.0, 'intermediate': 1.1, 'advanced': 1.2},
        'streak_day': {'base': 20, 'beginner': 1.0, 'intermediate': 1.0, 'advanced': 1.0},
        'ai_conversation': {'base': 5, 'beginner': 1.0, 'intermediate': 1.1, 'advanced': 1.2}
    }
    
    # Level system (Enhanced)
    LEVEL_SYSTEM = [
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
    
    # ===============================
    # Processing Configuration (Enhanced)
    # ===============================
    
    # Enhanced Chapter detection
    MIN_CHAPTER_WORDS = 50
    MAX_CHAPTERS_PER_BOOK = 100  # Increased
    CHAPTER_DETECTION_PATTERNS = 15  # Enhanced patterns
    CHAPTER_DETECTION_METHODS = ['pattern_matching', 'layout_analysis', 'educational_structure']
    
    # Content analysis (Enhanced)
    MAX_TOPICS_PER_CHAPTER = 12  # Increased
    MAX_CONCEPTS_PER_CHAPTER = 30  # Increased
    MAX_EXAMPLES_PER_CHAPTER = 15  # Increased
    
    # Topic modeling settings
    TOPIC_MODELING_CONFIG = {
        'min_topic_size': 3,
        'n_topics': 'auto',
        'embedding_model': 'all-MiniLM-L6-v2',
        'umap_n_components': 5,
        'calculate_probabilities': True
    }
    
    # Quiz generation (Enhanced)
    QUIZ_GENERATION_CONFIG = {
        'bloom_taxonomy_levels': ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create'],
        'question_types': ['multiple_choice', 'true_false', 'short_answer', 'matching', 'fill_blank'],
        'difficulty_adaptation': True,
        'concept_based_generation': True
    }
    
    # ===============================
    # Voice and Audio Configuration (NEW)
    # ===============================
    
    # Voice input settings
    VOICE_INPUT_ENABLED = True
    VOICE_TIMEOUT = 30
    VOICE_SILENCE_THRESHOLD = 0.01
    VOICE_CHUNK_DURATION = 1  # seconds
    
    # Speech-to-text configuration
    STT_CONFIG = {
        'primary': 'groq_whisper',
        'fallback': ['openai_whisper', 'speech_recognition'],
        'language': 'en',
        'model': 'whisper-large-v3'
    }
    
    # Audio recording settings
    AUDIO_RECORDING = {
        'sample_rate': 16000,  # Whisper prefers 16kHz
        'channels': 1,
        'chunk_size': 1024,
        'format': 'wav'
    }
    
    # ===============================
    # Multimodal AI Configuration (NEW)
    # ===============================
    
    # Image processing settings
    IMAGE_PROCESSING = {
        'max_image_size': (1024, 1024),
        'supported_formats': ['png', 'jpg', 'jpeg', 'gif', 'bmp'],
        'enhancement_enabled': True,
        'ocr_enabled': True
    }
    
    # Computer vision models
    VISION_MODELS = {
        'image_captioning': 'Salesforce/blip-image-captioning-base',
        'image_classification': 'ViT-B/32',
        'object_detection': 'facebook/detr-resnet-50',
        'ocr': 'tesseract'
    }
    
    # Educational image analysis
    EDUCATIONAL_IMAGE_ANALYSIS = {
        'diagram_detection': True,
        'chart_analysis': True,
        'text_extraction': True,
        'concept_identification': True
    }
    
    # ===============================
    # Rate Limiting & Performance (Enhanced)
    # ===============================
    
    # API rate limiting
    REQUESTS_PER_MINUTE = 100  # Increased
    TOKENS_PER_MINUTE = 100000  # Increased
    BATCH_SIZE = 10  # Increased
    RETRY_ATTEMPTS = 5  # Increased
    RETRY_DELAY = 60
    
    # Processing limits (Enhanced)
    MAX_FILE_SIZE_MB = 100  # Increased
    MAX_PROCESSING_TIME = 7200  # 2 hours
    CHUNK_SIZE = 2000  # Increased
    
    # Parallel processing
    MAX_WORKERS = 4
    ASYNC_PROCESSING_ENABLED = True
    
    # Caching configuration
    CACHE_SETTINGS = {
        'embeddings_ttl': 86400,  # 24 hours
        'images_ttl': 3600,  # 1 hour
        'models_ttl': 604800,  # 1 week
        'max_cache_size_mb': 1000
    }
    
    # ===============================
    # Logging & Debugging (Enhanced)
    # ===============================
    
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = str(BASE_DIR / "logs" / "intellilearn.log")
    
    # Enhanced logging configuration
    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
            },
            'simple': {
                'format': '%(levelname)s - %(message)s'
            }
        },
        'handlers': {
            'file': {
                'level': 'INFO',
                'class': 'logging.FileHandler',
                'filename': str(BASE_DIR / "logs" / "intellilearn.log"),
                'formatter': 'detailed'
            },
            'console': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'simple'
            }
        },
        'loggers': {
            'intellilearn': {
                'level': 'INFO',
                'handlers': ['file', 'console']
            }
        }
    }
    
    DEBUG_MODE = os.getenv('DEBUG', 'False').lower() == 'true'
    VERBOSE_LOGGING = os.getenv('VERBOSE', 'False').lower() == 'true'
    
    # ===============================
    # Security & Privacy (Enhanced)
    # ===============================
    
    # Session management
    SESSION_TIMEOUT = 7200  # 2 hours
    MAX_SESSIONS = 2000  # Increased
    SESSION_ENCRYPTION_ENABLED = True
    
    # Data retention (Enhanced)
    DATA_RETENTION = {
        'temp_files_hours': 24,
        'cache_days': 7,
        'user_data_days': 365,
        'logs_days': 30,
        'backups_days': 90
    }
    
    # Privacy settings
    PRIVACY_SETTINGS = {
        'anonymize_logs': True,
        'encrypt_sensitive_data': True,
        'auto_cleanup_enabled': True,
        'gdpr_compliance': True
    }
    
    # ===============================
    # Platform Integration (Enhanced)
    # ===============================
    
    # External services
    EXTERNAL_APIS = {
        'wikipedia': {
            'endpoint': 'https://en.wikipedia.org/api/rest_v1/',
            'enabled': True,
            'rate_limit': 100
        },
        'duckduckgo': {
            'enabled': True,
            'max_results': 10
        },
        'youtube': {
            'enabled': False,  # Optional integration
            'api_key': os.getenv('YOUTUBE_API_KEY', '')
        }
    }
    
    # Embedding and NLP models
    NLP_MODELS = {
        'sentence_transformer': 'all-MiniLM-L6-v2',
        'bert': 'bert-base-uncased',
        'roberta': 'roberta-base',
        'educational_bert': 'microsoft/DialoGPT-medium'
    }
    
    # Educational standards integration
    EDUCATIONAL_STANDARDS = {
        'bloom_taxonomy': True,
        'common_core': False,
        'international_baccalaureate': False,
        'custom_standards': True
    }
    
    # ===============================
    # Subject-Specific Configuration (NEW)
    # ===============================
    
    # Subject categories with specialized settings
    SUBJECT_CATEGORIES = {
        'STEM': ['Mathematics', 'Physics', 'Chemistry', 'Biology', 'Computer Science'],
        'Languages': ['English', 'Spanish', 'French', 'German', 'Chinese'],
        'Humanities': ['History', 'Literature', 'Philosophy', 'Art', 'Music'],
        'Social Sciences': ['Psychology', 'Sociology', 'Economics', 'Political Science'],
        'Applied': ['Engineering', 'Medicine', 'Business', 'Law']
    }
    
    # Subject-specific processing configurations
    SUBJECT_CONFIG = {
        'Mathematics': {
            'equation_parsing': True,
            'formula_recognition': True,
            'problem_solving_steps': True
        },
        'Science': {
            'diagram_analysis': True,
            'experiment_detection': True,
            'concept_mapping': True
        },
        'Languages': {
            'grammar_analysis': True,
            'literature_themes': True,
            'writing_assessment': True
        }
    }
    
    def __init__(self):
        """Initialize enhanced configuration and create necessary directories"""
        self.create_directories()
        self.validate_config()
        self.setup_logging()
        self.initialize_models()
    
    def create_directories(self):
        """Create all necessary directories"""
        directories = [
            self.BASE_DIR / "data",
            self.BASE_DIR / "output",
            self.BASE_DIR / "temp",
            self.BASE_DIR / "cache",
            self.BASE_DIR / "logs",
            self.BASE_DIR / "backup",
            self.BASE_DIR / "templates",
            self.BASE_DIR / "assets",
            self.BASE_DIR / "output" / "videos",
            self.BASE_DIR / "output" / "slides", 
            self.BASE_DIR / "output" / "audio",
            self.BASE_DIR / "temp" / "textbooks",
            self.BASE_DIR / "data" / "processed",
            self.BASE_DIR / "cache" / "models",
            self.BASE_DIR / "cache" / "embeddings",
            self.BASE_DIR / "cache" / "images",
            self.BASE_DIR / "assets" / "fonts"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def validate_config(self):
        """Enhanced configuration validation"""
        validation_results = {}
        
        # API key validation
        if not self.GROQ_API_KEY:
            st.warning("⚠️ GROQ_API_KEY not found. Please set it in Streamlit secrets or environment variables.")
            validation_results['groq_api'] = False
        elif not self.GROQ_API_KEY.startswith('gsk_'):
            st.error("❌ Invalid GROQ_API_KEY format. Should start with 'gsk_'")
            validation_results['groq_api'] = False
        else:
            validation_results['groq_api'] = True
        
        # Token limits validation
        if self.MAX_TOKENS > self.CONTEXT_WINDOW_LIMITS.get(self.GROQ_MODEL, 4096):
            st.warning(f"⚠️ MAX_TOKENS ({self.MAX_TOKENS}) exceeds model limit")
            validation_results['token_limits'] = False
        else:
            validation_results['token_limits'] = True
        
        # Directory permissions validation
        try:
            test_file = self.BASE_DIR / "temp" / "test_write.txt"
            test_file.write_text("test")
            test_file.unlink()
            validation_results['file_permissions'] = True
        except Exception:
            st.warning("⚠️ File system permissions issue detected")
            validation_results['file_permissions'] = False
        
        return validation_results
    
    def setup_logging(self):
        """Setup enhanced logging configuration"""
        import logging.config
        
        try:
            logging.config.dictConfig(self.LOGGING_CONFIG)
        except Exception as e:
            # Fallback to basic logging
            logging.basicConfig(
                level=getattr(logging, self.LOG_LEVEL),
                format=self.LOG_FORMAT,
                handlers=[
                    logging.FileHandler(self.LOG_FILE),
                    logging.StreamHandler()
                ]
            )
    
    def initialize_models(self):
        """Initialize model availability checking"""
        self.model_availability = {
            'sentence_transformers': False,
            'spacy': False,
            'transformers': False,
            'ollama': False,
            'coqui_tts': False,
            'clip': False,
            'layoutparser': False
        }
        
        # Check each model availability (non-blocking)
        self._check_model_availability()
    
    def _check_model_availability(self):
        """Check availability of open-source models"""
        try:
            import sentence_transformers
            self.model_availability['sentence_transformers'] = True
        except ImportError:
            pass
        
        try:
            import spacy
            self.model_availability['spacy'] = True
        except ImportError:
            pass
        
        try:
            import transformers
            self.model_availability['transformers'] = True
        except ImportError:
            pass
        
        try:
            import ollama
            self.model_availability['ollama'] = True
        except ImportError:
            pass
        
        try:
            from TTS.api import TTS
            self.model_availability['coqui_tts'] = True
        except ImportError:
            pass
        
        try:
            import clip
            self.model_availability['clip'] = True
        except ImportError:
            pass
    
    def debug_api_key(self):
        """Enhanced API key debugging"""
        st.write("**🔍 API Key Debug Information:**")
        
        # Check Streamlit secrets
        secrets_key = st.secrets.get('GROQ_API_KEY', '')
        env_key = os.getenv('GROQ_API_KEY', '')
        
        st.write(f"- Streamlit Secrets: {'✅ Found' if secrets_key else '❌ Not found'}")
        st.write(f"- Environment Variable: {'✅ Found' if env_key else '❌ Not found'}")
        
        if self.GROQ_API_KEY:
            st.write(f"- Current API Key: {'✅ Valid format' if self.GROQ_API_KEY.startswith('gsk_') else '❌ Invalid format'}")
            st.write(f"- API Key Length: {len(self.GROQ_API_KEY)} characters")
            masked_key = self.GROQ_API_KEY[:8] + "..." + self.GROQ_API_KEY[-4:]
            st.write(f"- Masked Key: {masked_key}")
        else:
            st.write("- Current API Key: ❌ Not found")
        
        # Model availability
        st.write("**🤖 Model Availability:**")
        for model_name, available in self.model_availability.items():
            status = "✅ Available" if available else "❌ Not available"
            st.write(f"- {model_name}: {status}")
    
    def get_model_config(self, model_name=None, difficulty_level='intermediate'):
        """Get enhanced configuration for specific model with difficulty adaptation"""
        model = model_name or self.GROQ_MODEL
        
        # Base configuration
        config = {
            'model': model,
            'max_tokens': min(self.MAX_TOKENS, self.CONTEXT_WINDOW_LIMITS.get(model, 4096)),
            'temperature': self.TEMPERATURE,
            'top_p': self.TOP_P,
            'frequency_penalty': self.FREQUENCY_PENALTY,
            'presence_penalty': self.PRESENCE_PENALTY,
            'difficulty_level': difficulty_level
        }
        
        # Difficulty-specific adjustments
        if difficulty_level == 'beginner':
            config['temperature'] = 0.3  # More consistent responses
            config['max_tokens'] = min(config['max_tokens'], 2048)
        elif difficulty_level == 'advanced':
            config['temperature'] = 0.8  # More creative responses
            config['max_tokens'] = config['max_tokens']  # Full token limit
        
        return config
    
    def get_difficulty_config(self, difficulty_level='intermediate'):
        """Get configuration for specific difficulty level"""
        return {
            'quiz_settings': self.QUIZ_SETTINGS.get(difficulty_level, self.QUIZ_SETTINGS['intermediate']),
            'voice_settings': self.VOICE_SETTINGS.get(difficulty_level, self.VOICE_SETTINGS['intermediate']),
            'video_quality': self.VIDEO_QUALITY_PRESETS.get(difficulty_level, self.VIDEO_QUALITY_PRESETS['intermediate']),
            'reading_speed': self.READING_SPEEDS.get(difficulty_level, self.DEFAULT_READING_SPEED),
            'points_multiplier': self.POINTS_SYSTEM['quiz_completed'].get(difficulty_level, 1.0)
        }
    
    def get_directory_config(self):
        """Get all directory configurations"""
        return {
            'video_output': self.VIDEO_OUTPUT_DIR,
            'slides_output': self.SLIDES_OUTPUT_DIR,
            'audio_output': self.AUDIO_OUTPUT_DIR,
            'textbook_upload': self.TEXTBOOK_UPLOAD_DIR,
            'processed_data': self.PROCESSED_DATA_DIR,
            'cache': self.CACHE_DIR,
            'models': self.MODEL_CACHE_DIR,
            'embeddings': self.EMBEDDING_CACHE_DIR,
            'images': self.IMAGE_CACHE_DIR,
            'backup': self.BACKUP_DIR,
            'templates': self.TEMPLATES_DIR,
            'assets': self.ASSETS_DIR
        }
    
    def get_voice_config(self, difficulty_level='intermediate'):
        """Get voice configuration for specific difficulty level"""
        base_config = self.VOICE_SETTINGS.get(difficulty_level, self.VOICE_SETTINGS['intermediate'])
        
        return {
            **base_config,
            'stt_config': self.STT_CONFIG,
            'audio_recording': self.AUDIO_RECORDING,
            'tts_models': self.TTS_VOICES,
            'enabled': self.VOICE_INPUT_ENABLED
        }
    
    def get_subject_config(self, subject='General'):
        """Get subject-specific configuration"""
        subject_category = None
        for category, subjects in self.SUBJECT_CATEGORIES.items():
            if subject in subjects:
                subject_category = category
                break
        
        return {
            'category': subject_category,
            'specialized_config': self.SUBJECT_CONFIG.get(subject, {}),
            'educational_standards': self.EDUCATIONAL_STANDARDS
        }
    
    @property
    def is_configured(self):
        """Check if basic configuration is valid"""
        return bool(self.GROQ_API_KEY and self.GROQ_API_KEY.startswith('gsk_'))
    
    @property
    def is_enhanced_mode_available(self):
        """Check if enhanced features are available"""
        return any([
            self.model_availability.get('sentence_transformers', False),
            self.model_availability.get('transformers', False),
            self.model_availability.get('ollama', False)
        ])
    
    def export_config(self):
        """Export configuration for backup/debugging"""
        config_export = {
            'version': '2.0.0',
            'timestamp': str(Path(__file__).stat().st_mtime),
            'directories': self.get_directory_config(),
            'model_availability': self.model_availability,
            'difficulty_levels': self.DIFFICULTY_LEVELS,
            'subject_categories': list(self.SUBJECT_CATEGORIES.keys()),
            'features_enabled': {
                'open_source_models': self.USE_OPEN_SOURCE_MODELS,
                'voice_input': self.VOICE_INPUT_ENABLED,
                'multimodal': self.IMAGE_PROCESSING,
                'enhanced_nlp': self.TOPIC_MODELING_ENABLED
            }
        }
        
        return json.dumps(config_export, indent=2, default=str)


# Global configuration instance
config = Config()
