#!/usr/bin/env python3
"""
IntelliLearn AI Educational Assistant - Enhanced Modules Package
Comprehensive AI-powered educational platform with open-source model integration

This package contains the complete suite of educational AI modules:
- Advanced text processing with NLP and educational content analysis
- Multi-modal AI integration (text, voice, video, images)
- Adaptive learning systems with difficulty-aware content
- Comprehensive analytics and progress tracking
- Open-source model integration for cost-effective operation
- Enhanced gamification and engagement systems

Version 2.0.0 - Enhanced with Open Source AI Models
Author: IntelliLearn AI Development Team
License: MIT
"""

import logging
import warnings
from typing import Dict, Any, Optional, List
from pathlib import Path

# Configure package-level logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Version and metadata
__version__ = "2.0.0"
__author__ = "IntelliLearn AI Development Team"
__license__ = "MIT"
__description__ = "Advanced AI Educational Assistant with Open Source Models"

# Enhanced module imports with fallback handling
def safe_import(module_name: str, class_name: str, fallback_class=None):
    """Safely import modules with fallback options"""
    try:
        module = __import__(f"modules.{module_name}", fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError as e:
        logger.warning(f"Could not import {class_name} from {module_name}: {e}")
        if fallback_class:
            logger.info(f"Using fallback class for {class_name}")
            return fallback_class
        return None
    except Exception as e:
        logger.error(f"Error importing {class_name}: {e}")
        return None

# Core modules (always available)
try:
    from .text_processor import AdvancedTextProcessor
    TEXT_PROCESSOR_AVAILABLE = True
except ImportError:
    logger.warning("AdvancedTextProcessor not available, using fallback")
    from .text_processor import TextProcessor as AdvancedTextProcessor
    TEXT_PROCESSOR_AVAILABLE = False

try:
    from .video_generator import ProductionVideoGenerator
    VIDEO_GENERATOR_AVAILABLE = True
except ImportError:
    logger.warning("ProductionVideoGenerator not available, using fallback")
    from .video_generator import VideoGenerator as ProductionVideoGenerator
    VIDEO_GENERATOR_AVAILABLE = False

try:
    from .chatbot import EnhancedEducationalChatbot
    CHATBOT_AVAILABLE = True
except ImportError:
    logger.warning("EnhancedEducationalChatbot not available, using fallback")
    from .chatbot import ChatbotAssistant as EnhancedEducationalChatbot
    CHATBOT_AVAILABLE = False

try:
    from .analytics_dashboard import LearningAnalyticsDashboard
    ANALYTICS_AVAILABLE = True
except ImportError:
    logger.warning("LearningAnalyticsDashboard not available")
    LearningAnalyticsDashboard = None
    ANALYTICS_AVAILABLE = False

# Enhanced modules (optional, with graceful degradation)
try:
    from .enhanced_chapter_detection_v2 import SuperiorChapterDetector
    ENHANCED_CHAPTER_DETECTION_AVAILABLE = True
except ImportError:
    logger.info("SuperiorChapterDetector not available - using standard detection")
    SuperiorChapterDetector = None
    ENHANCED_CHAPTER_DETECTION_AVAILABLE = False

try:
    from .multimodal_processor import OpenSourceMultimodalProcessor
    MULTIMODAL_AVAILABLE = True
except ImportError:
    logger.info("OpenSourceMultimodalProcessor not available")
    OpenSourceMultimodalProcessor = None
    MULTIMODAL_AVAILABLE = False

try:
    from .voice_chat_processor import VoiceChatProcessor
    VOICE_CHAT_AVAILABLE = True
except ImportError:
    logger.info("VoiceChatProcessor not available")
    VoiceChatProcessor = None
    VOICE_CHAT_AVAILABLE = False

try:
    from .enhanced_nlp_processor import EnhancedNLPProcessor
    ENHANCED_NLP_AVAILABLE = True
except ImportError:
    logger.info("EnhancedNLPProcessor not available")
    EnhancedNLPProcessor = None
    ENHANCED_NLP_AVAILABLE = False

try:
    from .opensource_video_generator import OpenSourceVideoGenerator
    OPENSOURCE_VIDEO_AVAILABLE = True
except ImportError:
    logger.info("OpenSourceVideoGenerator not available")
    OpenSourceVideoGenerator = None
    OPENSOURCE_VIDEO_AVAILABLE = False

try:
    from .gamification_enhanced import EnhancedGamificationEngine
    ENHANCED_GAMIFICATION_AVAILABLE = True
except ImportError:
    logger.info("EnhancedGamificationEngine not available")
    from .gamification import GamificationEngine as EnhancedGamificationEngine
    ENHANCED_GAMIFICATION_AVAILABLE = False

try:
    from .adaptive_learning import AdaptiveLearningEngine
    ADAPTIVE_LEARNING_AVAILABLE = True
except ImportError:
    logger.info("AdaptiveLearningEngine not available")
    AdaptiveLearningEngine = None
    ADAPTIVE_LEARNING_AVAILABLE = False

try:
    from .rag_pipeline import AdvancedRAGPipeline
    RAG_PIPELINE_AVAILABLE = True
except ImportError:
    logger.info("AdvancedRAGPipeline not available")
    AdvancedRAGPipeline = None
    RAG_PIPELINE_AVAILABLE = False

# Utility modules
try:
    from .helpers import (
        EnhancedDirectoryManager,
        AdvancedFileValidator,
        IntelligentTextProcessor,
        AdvancedCacheManager,
        AdvancedResponseFormatter,
        KnowledgeBaseValidator
    )
    ENHANCED_HELPERS_AVAILABLE = True
except ImportError:
    logger.info("Enhanced helpers not available")
    EnhancedDirectoryManager = None
    AdvancedFileValidator = None
    IntelligentTextProcessor = None
    AdvancedCacheManager = None
    AdvancedResponseFormatter = None
    KnowledgeBaseValidator = None
    ENHANCED_HELPERS_AVAILABLE = False

# Configuration and feature flags
FEATURE_FLAGS = {
    'core_modules': {
        'text_processor': TEXT_PROCESSOR_AVAILABLE,
        'video_generator': VIDEO_GENERATOR_AVAILABLE,
        'chatbot': CHATBOT_AVAILABLE,
        'analytics_dashboard': ANALYTICS_AVAILABLE
    },
    'enhanced_modules': {
        'superior_chapter_detection': ENHANCED_CHAPTER_DETECTION_AVAILABLE,
        'multimodal_processor': MULTIMODAL_AVAILABLE,
        'voice_chat': VOICE_CHAT_AVAILABLE,
        'enhanced_nlp': ENHANCED_NLP_AVAILABLE,
        'opensource_video': OPENSOURCE_VIDEO_AVAILABLE,
        'enhanced_gamification': ENHANCED_GAMIFICATION_AVAILABLE,
        'adaptive_learning': ADAPTIVE_LEARNING_AVAILABLE,
        'rag_pipeline': RAG_PIPELINE_AVAILABLE
    },
    'utility_modules': {
        'enhanced_helpers': ENHANCED_HELPERS_AVAILABLE
    }
}

# Module compatibility matrix
COMPATIBILITY_MATRIX = {
    'minimum_required': ['AdvancedTextProcessor', 'EnhancedEducationalChatbot'],
    'recommended': ['ProductionVideoGenerator', 'LearningAnalyticsDashboard'],
    'advanced_features': [
        'SuperiorChapterDetector', 'OpenSourceMultimodalProcessor',
        'VoiceChatProcessor', 'EnhancedNLPProcessor'
    ],
    'optional_enhancements': [
        'OpenSourceVideoGenerator', 'EnhancedGamificationEngine',
        'AdaptiveLearningEngine', 'AdvancedRAGPipeline'
    ]
}

# Main exports - Core modules (always included)
__all__ = [
    # Core modules
    'AdvancedTextProcessor',
    'ProductionVideoGenerator',
    'EnhancedEducationalChatbot',
    'LearningAnalyticsDashboard',
    
    # Enhanced modules (conditional)
    'SuperiorChapterDetector',
    'OpenSourceMultimodalProcessor',
    'VoiceChatProcessor',
    'EnhancedNLPProcessor',
    'OpenSourceVideoGenerator',
    'EnhancedGamificationEngine',
    'AdaptiveLearningEngine',
    'AdvancedRAGPipeline',
    
    # Utility modules
    'EnhancedDirectoryManager',
    'AdvancedFileValidator',
    'IntelligentTextProcessor',
    'AdvancedCacheManager',
    'AdvancedResponseFormatter',
    'KnowledgeBaseValidator',
    
    # Package functions
    'get_available_modules',
    'get_module_info',
    'check_dependencies',
    'initialize_platform',
    'get_feature_flags'
]

# Filter out None values from __all__
__all__ = [name for name in __all__ if globals().get(name) is not None]

def get_available_modules() -> Dict[str, bool]:
    """Get list of available modules and their status"""
    return {
        'core_modules': {
            'AdvancedTextProcessor': TEXT_PROCESSOR_AVAILABLE,
            'ProductionVideoGenerator': VIDEO_GENERATOR_AVAILABLE,
            'EnhancedEducationalChatbot': CHATBOT_AVAILABLE,
            'LearningAnalyticsDashboard': ANALYTICS_AVAILABLE
        },
        'enhanced_modules': {
            'SuperiorChapterDetector': ENHANCED_CHAPTER_DETECTION_AVAILABLE,
            'OpenSourceMultimodalProcessor': MULTIMODAL_AVAILABLE,
            'VoiceChatProcessor': VOICE_CHAT_AVAILABLE,
            'EnhancedNLPProcessor': ENHANCED_NLP_AVAILABLE,
            'OpenSourceVideoGenerator': OPENSOURCE_VIDEO_AVAILABLE,
            'EnhancedGamificationEngine': ENHANCED_GAMIFICATION_AVAILABLE,
            'AdaptiveLearningEngine': ADAPTIVE_LEARNING_AVAILABLE,
            'AdvancedRAGPipeline': RAG_PIPELINE_AVAILABLE
        },
        'utility_modules': {
            'EnhancedHelpers': ENHANCED_HELPERS_AVAILABLE
        }
    }

def get_module_info(module_name: str) -> Dict[str, Any]:
    """Get detailed information about a specific module"""
    
    module_descriptions = {
        'AdvancedTextProcessor': {
            'description': 'Advanced text processing with NLP and educational content analysis',
            'capabilities': ['PDF processing', 'Chapter detection', 'Concept extraction', 'Difficulty analysis'],
            'dependencies': ['PyMuPDF', 'spaCy', 'transformers'],
            'version': '2.0.0'
        },
        'ProductionVideoGenerator': {
            'description': 'High-quality educational video generation with AI narration',
            'capabilities': ['Script generation', 'Voice synthesis', 'Slide creation', 'Video rendering'],
            'dependencies': ['MoviePy', 'gTTS', 'Pillow', 'matplotlib'],
            'version': '2.0.0'
        },
        'EnhancedEducationalChatbot': {
            'description': 'Intelligent chatbot with educational focus and adaptive responses',
            'capabilities': ['Context-aware responses', 'Difficulty adaptation', 'Learning tracking', 'Multi-modal input'],
            'dependencies': ['streamlit', 'requests', 'sentence-transformers'],
            'version': '2.0.0'
        },
        'LearningAnalyticsDashboard': {
            'description': 'Comprehensive analytics and progress tracking system',
            'capabilities': ['Progress visualization', 'Performance metrics', 'Learning insights', 'Predictive analytics'],
            'dependencies': ['plotly', 'pandas', 'numpy', 'scikit-learn'],
            'version': '2.0.0'
        },
        'SuperiorChapterDetector': {
            'description': 'Advanced chapter detection with 90%+ accuracy for educational content',
            'capabilities': ['15+ detection patterns', 'Educational content analysis', 'Multi-pass detection'],
            'dependencies': ['spaCy', 'regex', 'textstat'],
            'version': '2.0.0'
        },
        'OpenSourceMultimodalProcessor': {
            'description': 'Multimodal AI processing for images, text, and educational diagrams',
            'capabilities': ['Image understanding', 'OCR', 'Diagram analysis', 'Visual question answering'],
            'dependencies': ['transformers', 'Pillow', 'opencv-python', 'clip-by-openai'],
            'version': '2.0.0'
        },
        'VoiceChatProcessor': {
            'description': 'Voice-enabled learning with speech recognition and synthesis',
            'capabilities': ['Speech-to-text', 'Text-to-speech', 'Voice chat', 'Difficulty-adjusted speech'],
            'dependencies': ['whisper', 'TTS', 'pydub', 'audio-recorder-streamlit'],
            'version': '2.0.0'
        },
        'EnhancedNLPProcessor': {
            'description': 'Advanced NLP with topic modeling and educational concept extraction',
            'capabilities': ['BERTopic modeling', 'Concept extraction', 'Sentiment analysis', 'Text classification'],
            'dependencies': ['bertopic', 'sentence-transformers', 'spaCy', 'textstat'],
            'version': '2.0.0'
        },
        'OpenSourceVideoGenerator': {
            'description': 'Local video generation with open-source models',
            'capabilities': ['Local AI narration', 'Custom voice models', 'Educational slide generation'],
            'dependencies': ['TTS', 'ollama', 'moviepy', 'matplotlib'],
            'version': '2.0.0'
        },
        'EnhancedGamificationEngine': {
            'description': 'Advanced gamification with difficulty-based achievements',
            'capabilities': ['20+ achievement badges', 'Progress tracking', 'Adaptive rewards', 'Learning streaks'],
            'dependencies': ['streamlit', 'pandas', 'datetime'],
            'version': '2.0.0'
        },
        'AdaptiveLearningEngine': {
            'description': 'Personalized learning paths based on user performance and style',
            'capabilities': ['Learning style detection', 'Adaptive paths', 'Performance prediction', 'Content recommendation'],
            'dependencies': ['scikit-learn', 'numpy', 'pandas'],
            'version': '2.0.0'
        },
        'AdvancedRAGPipeline': {
            'description': 'Advanced retrieval-augmented generation for educational content',
            'capabilities': ['FAISS indexing', 'Hybrid search', 'Context generation', 'Educational optimization'],
            'dependencies': ['sentence-transformers', 'faiss-cpu', 'chromadb', 'langchain'],
            'version': '2.0.0'
        }
    }
    
    return module_descriptions.get(module_name, {
        'description': 'Module information not available',
        'capabilities': [],
        'dependencies': [],
        'version': 'Unknown'
    })

def check_dependencies() -> Dict[str, Dict[str, bool]]:
    """Check which dependencies are available for each module"""
    
    dependencies_status = {}
    
    # Core dependencies
    core_deps = {
        'streamlit': False,
        'requests': False,
        'pandas': False,
        'numpy': False
    }
    
    # Enhanced dependencies
    enhanced_deps = {
        'transformers': False,
        'sentence_transformers': False,
        'spacy': False,
        'moviepy': False,
        'plotly': False,
        'faiss': False,
        'whisper': False,
        'TTS': False,
        'bertopic': False,
        'ollama': False
    }
    
    # Check core dependencies
    for dep in core_deps:
        try:
            __import__(dep)
            core_deps[dep] = True
        except ImportError:
            pass
    
    # Check enhanced dependencies
    for dep in enhanced_deps:
        try:
            if dep == 'faiss':
                __import__('faiss')
            elif dep == 'sentence_transformers':
                __import__('sentence_transformers')
            else:
                __import__(dep)
            enhanced_deps[dep] = True
        except ImportError:
            pass
    
    return {
        'core_dependencies': core_deps,
        'enhanced_dependencies': enhanced_deps,
        'core_satisfied': all(core_deps.values()),
        'enhanced_satisfied': all(enhanced_deps.values())
    }

def initialize_platform(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Initialize the IntelliLearn AI platform with available modules"""
    
    if config is None:
        config = {}
    
    initialization_result = {
        'success': False,
        'core_modules_loaded': 0,
        'enhanced_modules_loaded': 0,
        'total_modules_available': len(__all__),
        'errors': [],
        'warnings': [],
        'initialized_modules': {},
        'platform_capabilities': []
    }
    
    try:
        # Initialize core modules
        core_modules = ['AdvancedTextProcessor', 'EnhancedEducationalChatbot']
        
        for module_name in core_modules:
            if globals().get(module_name) is not None:
                initialization_result['core_modules_loaded'] += 1
                initialization_result['initialized_modules'][module_name] = True
                logger.info(f"✅ Core module loaded: {module_name}")
            else:
                initialization_result['errors'].append(f"Core module missing: {module_name}")
                logger.error(f"❌ Core module missing: {module_name}")
        
        # Initialize enhanced modules
        enhanced_modules = [
            'ProductionVideoGenerator', 'LearningAnalyticsDashboard',
            'SuperiorChapterDetector', 'OpenSourceMultimodalProcessor',
            'VoiceChatProcessor', 'EnhancedNLPProcessor'
        ]
        
        for module_name in enhanced_modules:
            if globals().get(module_name) is not None:
                initialization_result['enhanced_modules_loaded'] += 1
                initialization_result['initialized_modules'][module_name] = True
                logger.info(f"✅ Enhanced module loaded: {module_name}")
            else:
                initialization_result['warnings'].append(f"Enhanced module not available: {module_name}")
                logger.warning(f"⚠️ Enhanced module not available: {module_name}")
        
        # Determine platform capabilities
        capabilities = []
        
        if TEXT_PROCESSOR_AVAILABLE:
            capabilities.extend(['PDF Processing', 'Text Analysis', 'Chapter Detection'])
        
        if CHATBOT_AVAILABLE:
            capabilities.extend(['Interactive Q&A', 'Educational Assistance'])
        
        if VIDEO_GENERATOR_AVAILABLE:
            capabilities.extend(['Video Generation', 'Educational Content Creation'])
        
        if ANALYTICS_AVAILABLE:
            capabilities.extend(['Learning Analytics', 'Progress Tracking'])
        
        if MULTIMODAL_AVAILABLE:
            capabilities.extend(['Image Processing', 'Multimodal AI'])
        
        if VOICE_CHAT_AVAILABLE:
            capabilities.extend(['Voice Interaction', 'Speech Processing'])
        
        if ENHANCED_GAMIFICATION_AVAILABLE:
            capabilities.extend(['Gamification', 'Achievement System'])
        
        if ADAPTIVE_LEARNING_AVAILABLE:
            capabilities.extend(['Adaptive Learning', 'Personalized Paths'])
        
        if RAG_PIPELINE_AVAILABLE:
            capabilities.extend(['Advanced Search', 'Context Generation'])
        
        initialization_result['platform_capabilities'] = capabilities
        
        # Success criteria: at least core modules loaded
        initialization_result['success'] = initialization_result['core_modules_loaded'] >= 2
        
        if initialization_result['success']:
            logger.info(f"🎉 IntelliLearn AI Platform initialized successfully!")
            logger.info(f"📊 Core modules: {initialization_result['core_modules_loaded']}")
            logger.info(f"🚀 Enhanced modules: {initialization_result['enhanced_modules_loaded']}")
            logger.info(f"🎯 Capabilities: {', '.join(capabilities)}")
        else:
            logger.error("❌ Platform initialization failed - insufficient core modules")
        
    except Exception as e:
        initialization_result['errors'].append(f"Initialization error: {str(e)}")
        logger.error(f"💥 Platform initialization error: {e}")
    
    return initialization_result

def get_feature_flags() -> Dict[str, Dict[str, bool]]:
    """Get current feature flags for the platform"""
    return FEATURE_FLAGS.copy()

def get_platform_info() -> Dict[str, Any]:
    """Get comprehensive platform information"""
    
    available_modules = get_available_modules()
    dependencies = check_dependencies()
    
    # Count available modules
    core_count = sum(available_modules['core_modules'].values())
    enhanced_count = sum(available_modules['enhanced_modules'].values())
    total_possible = len(available_modules['core_modules']) + len(available_modules['enhanced_modules'])
    
    return {
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'modules': {
            'core_available': core_count,
            'enhanced_available': enhanced_count,
            'total_available': core_count + enhanced_count,
            'total_possible': total_possible,
            'availability_percentage': ((core_count + enhanced_count) / total_possible) * 100
        },
        'dependencies': dependencies,
        'feature_flags': FEATURE_FLAGS,
        'compatibility_matrix': COMPATIBILITY_MATRIX,
        'platform_ready': core_count >= 2  # Minimum viable platform
    }

def display_platform_status():
    """Display platform status (for debugging/info purposes)"""
    
    info = get_platform_info()
    
    print("🎓 IntelliLearn AI Educational Platform")
    print("=" * 50)
    print(f"Version: {info['version']}")
    print(f"Author: {info['author']}")
    print(f"Description: {info['description']}")
    print()
    
    print("📊 Module Status:")
    print(f"  Core modules: {info['modules']['core_available']}/4")
    print(f"  Enhanced modules: {info['modules']['enhanced_available']}/8")
    print(f"  Overall availability: {info['modules']['availability_percentage']:.1f}%")
    print(f"  Platform ready: {'✅ Yes' if info['platform_ready'] else '❌ No'}")
    print()
    
    print("🔧 Dependencies:")
    core_deps = info['dependencies']['core_dependencies']
    enhanced_deps = info['dependencies']['enhanced_dependencies']
    
    print("  Core dependencies:")
    for dep, available in core_deps.items():
        status = "✅" if available else "❌"
        print(f"    {status} {dep}")
    
    print("  Enhanced dependencies:")
    for dep, available in list(enhanced_deps.items())[:5]:  # Show first 5
        status = "✅" if available else "❌"
        print(f"    {status} {dep}")
    
    if len(enhanced_deps) > 5:
        remaining = len(enhanced_deps) - 5
        available_remaining = sum(list(enhanced_deps.values())[5:])
        print(f"    ... and {remaining} more ({available_remaining}/{remaining} available)")

# Module initialization logging
logger.info(f"🎓 IntelliLearn AI Modules Package v{__version__} loaded")
logger.info(f"📦 Available modules: {len(__all__)}")

# Display warnings for missing critical modules
missing_core = []
for module in COMPATIBILITY_MATRIX['minimum_required']:
    if globals().get(module) is None:
        missing_core.append(module)

if missing_core:
    logger.warning(f"⚠️ Missing core modules: {', '.join(missing_core)}")
    logger.warning("Platform functionality may be limited")

# Success message
available_count = sum(1 for name in __all__ if globals().get(name) is not None)
logger.info(f"✅ Successfully loaded {available_count}/{len(__all__)} modules")

# Optional: Display full status on import (useful for debugging)
# Uncomment the next line to see detailed status on import
# display_platform_status()
