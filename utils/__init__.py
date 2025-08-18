#!/usr/bin/env python3
"""
IntelliLearn AI Educational Assistant - Enhanced Utilities Package
Comprehensive utility modules for advanced educational AI platform

This package contains enhanced utility functions and classes that provide
the foundation for the AI Educational Assistant:

Core Utilities:
- helpers: Enhanced utility functions with NLP and educational optimization
- rag_pipeline: Advanced Retrieval-Augmented Generation with FAISS indexing
- cache_manager: Intelligent caching system with TTL and cleanup
- text_processing: Advanced text analysis with educational content awareness
- file_handling: Comprehensive file validation and processing
- learning_analytics: Statistical analysis and progress tracking utilities

Enhanced Features:
- Open-source model integration
- Difficulty-aware content processing
- Educational content optimization
- Advanced caching and performance optimization
- Comprehensive validation and error handling
- Multi-modal content support

Version 2.0.0 - Enhanced with Open Source AI Models
Author: IntelliLearn AI Development Team
License: MIT
"""

import logging
import warnings
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import sys

# Configure package-level logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Package metadata
__version__ = "2.0.0"
__author__ = "IntelliLearn AI Development Team"
__license__ = "MIT"
__description__ = "Enhanced Educational AI Utilities with Open Source Integration"

# Enhanced imports with fallback handling
def safe_import_utility(module_name: str, item_names: List[str]) -> Dict[str, Any]:
    """Safely import utility functions with fallback options"""
    imported_items = {}
    
    try:
        module = __import__(f"utils.{module_name}", fromlist=item_names)
        
        for item_name in item_names:
            try:
                imported_items[item_name] = getattr(module, item_name)
                logger.debug(f"✅ Imported {item_name} from {module_name}")
            except AttributeError:
                logger.warning(f"⚠️ {item_name} not found in {module_name}")
                imported_items[item_name] = None
        
    except ImportError as e:
        logger.warning(f"❌ Could not import from {module_name}: {e}")
        for item_name in item_names:
            imported_items[item_name] = None
    
    return imported_items

# Core helper functions (enhanced versions)
helper_imports = safe_import_utility('helpers', [
    # Core functions (backward compatible)
    'setup_directories',
    'validate_file_type',
    'calculate_file_hash',
    'clean_text',
    'chunk_text',
    'extract_key_terms',
    'format_response_with_citations',
    'safe_json_parse',
    'estimate_reading_time',
    'create_progress_tracker',
    'validate_knowledge_base',
    'log_user_interaction',
    'format_duration',
    'truncate_text',
    'create_topic_summary',
    'ensure_session_state_keys',
    'TextAnalyzer',
    
    # Enhanced classes and functions
    'EnhancedDirectoryManager',
    'AdvancedFileValidator',
    'IntelligentTextProcessor',
    'AdvancedCacheManager',
    'AdvancedResponseFormatter',
    'KnowledgeBaseValidator',
    'create_learning_progress_tracker',
    'display_knowledge_base_statistics',
    'estimate_reading_time_enhanced',
    'create_download_button_enhanced',
    'display_processing_summary',
    'ensure_enhanced_session_state'
])

# Extract imported items to module level
for name, item in helper_imports.items():
    if item is not None:
        globals()[name] = item

# RAG Pipeline components
rag_imports = safe_import_utility('rag_pipeline', [
    # Core RAG classes
    'RAGPipeline',
    'EnhancedRAGPipeline',
    'AdvancedRAGPipeline',
    'Document',
    'EnhancedDocument',
    
    # Utility functions
    'create_rag_pipeline',
    'create_advanced_rag_pipeline',
    'get_cached_rag_pipeline',
    'calculate_knowledge_base_hash',
    'display_rag_interface',
    'test_advanced_rag_pipeline'
])

# Extract RAG imports to module level
for name, item in rag_imports.items():
    if item is not None:
        globals()[name] = item

# Feature availability flags
FEATURE_AVAILABILITY = {
    'core_helpers': helper_imports['setup_directories'] is not None,
    'enhanced_helpers': helper_imports['EnhancedDirectoryManager'] is not None,
    'basic_rag': rag_imports['RAGPipeline'] is not None,
    'advanced_rag': rag_imports['AdvancedRAGPipeline'] is not None,
    'text_analysis': helper_imports['TextAnalyzer'] is not None,
    'cache_management': helper_imports['AdvancedCacheManager'] is not None,
    'file_validation': helper_imports['AdvancedFileValidator'] is not None
}

# Backward compatibility aliases
def create_progress_tracker_legacy(*args, **kwargs):
    """Legacy progress tracker (backward compatibility)"""
    if helper_imports['create_learning_progress_tracker']:
        return helper_imports['create_learning_progress_tracker'](*args, **kwargs)
    elif helper_imports['create_progress_tracker']:
        return helper_imports['create_progress_tracker'](*args, **kwargs)
    else:
        logger.error("No progress tracker available")
        return None

# Enhanced __all__ list with conditional inclusion
_base_exports = [
    # Core helper functions (always included if available)
    "setup_directories",
    "validate_file_type", 
    "calculate_file_hash",
    "clean_text",
    "chunk_text",
    "extract_key_terms",
    "format_response_with_citations",
    "safe_json_parse",
    "estimate_reading_time",
    "create_progress_tracker",
    "validate_knowledge_base",
    "log_user_interaction",
    "format_duration",
    "truncate_text",
    "create_topic_summary",
    "ensure_session_state_keys",
    "TextAnalyzer",
    
    # RAG Pipeline (core)
    "RAGPipeline",
    "Document",
    "create_rag_pipeline",
    "get_cached_rag_pipeline"
]

_enhanced_exports = [
    # Enhanced helper classes
    "EnhancedDirectoryManager",
    "AdvancedFileValidator",
    "IntelligentTextProcessor",
    "AdvancedCacheManager",
    "AdvancedResponseFormatter",
    "KnowledgeBaseValidator",
    
    # Enhanced helper functions
    "create_learning_progress_tracker",
    "display_knowledge_base_statistics",
    "estimate_reading_time_enhanced",
    "create_download_button_enhanced",
    "display_processing_summary",
    "ensure_enhanced_session_state",
    
    # Enhanced RAG Pipeline
    "EnhancedRAGPipeline",
    "AdvancedRAGPipeline",
    "EnhancedDocument",
    "create_advanced_rag_pipeline",
    "calculate_knowledge_base_hash",
    "display_rag_interface",
    
    # Utility functions
    "get_utility_info",
    "check_utility_dependencies",
    "initialize_utilities",
    "get_feature_availability"
]

# Build __all__ list based on what's actually available
__all__ = []

# Add base exports if available
for export in _base_exports:
    if globals().get(export) is not None:
        __all__.append(export)

# Add enhanced exports if available
for export in _enhanced_exports:
    if globals().get(export) is not None:
        __all__.append(export)

# Add utility management functions
__all__.extend([
    "get_utility_info",
    "check_utility_dependencies", 
    "initialize_utilities",
    "get_feature_availability",
    "create_progress_tracker_legacy"
])

def get_utility_info() -> Dict[str, Any]:
    """Get comprehensive information about available utilities"""
    
    return {
        'version': __version__,
        'description': __description__,
        'feature_availability': FEATURE_AVAILABILITY.copy(),
        'available_functions': {
            'core_helpers': [name for name in _base_exports if globals().get(name) is not None],
            'enhanced_helpers': [name for name in _enhanced_exports if globals().get(name) is not None],
            'total_available': len(__all__)
        },
        'capabilities': {
            'text_processing': FEATURE_AVAILABILITY['core_helpers'],
            'advanced_text_analysis': FEATURE_AVAILABILITY['text_analysis'],
            'file_handling': FEATURE_AVAILABILITY['file_validation'],
            'caching': FEATURE_AVAILABILITY['cache_management'],
            'basic_rag': FEATURE_AVAILABILITY['basic_rag'],
            'advanced_rag': FEATURE_AVAILABILITY['advanced_rag'],
            'educational_optimization': FEATURE_AVAILABILITY['enhanced_helpers']
        },
        'backward_compatibility': True,
        'open_source_integration': FEATURE_AVAILABILITY['advanced_rag']
    }

def check_utility_dependencies() -> Dict[str, Dict[str, bool]]:
    """Check dependencies for utility modules"""
    
    dependencies = {
        'core_dependencies': {
            'pathlib': True,  # Built-in
            'hashlib': True,  # Built-in
            'json': True,     # Built-in
            're': True,       # Built-in
            'datetime': True, # Built-in
            'typing': True,   # Built-in
        },
        'enhanced_dependencies': {
            'numpy': False,
            'pandas': False,
            'sentence_transformers': False,
            'spacy': False,
            'textstat': False,
            'faiss': False,
            'chromadb': False,
            'plotly': False,
            'scikit_learn': False
        },
        'optional_dependencies': {
            'transformers': False,
            'torch': False,
            'langchain': False,
            'bertopic': False,
            'opencv': False,
            'pillow': False
        }
    }
    
    # Check enhanced dependencies
    for dep_category in ['enhanced_dependencies', 'optional_dependencies']:
        for dep in dependencies[dep_category]:
            try:
                if dep == 'scikit_learn':
                    import sklearn
                elif dep == 'opencv':
                    import cv2
                elif dep == 'pillow':
                    import PIL
                else:
                    __import__(dep)
                dependencies[dep_category][dep] = True
            except ImportError:
                pass
    
    # Calculate satisfaction rates
    core_satisfied = all(dependencies['core_dependencies'].values())
    enhanced_satisfied = sum(dependencies['enhanced_dependencies'].values()) / len(dependencies['enhanced_dependencies'])
    optional_satisfied = sum(dependencies['optional_dependencies'].values()) / len(dependencies['optional_dependencies'])
    
    dependencies['satisfaction_rates'] = {
        'core': 100.0 if core_satisfied else 0.0,
        'enhanced': enhanced_satisfied * 100,
        'optional': optional_satisfied * 100
    }
    
    return dependencies

def initialize_utilities(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Initialize utility modules with configuration"""
    
    if config is None:
        config = {}
    
    initialization_result = {
        'success': False,
        'utilities_loaded': 0,
        'total_utilities': len(__all__),
        'errors': [],
        'warnings': [],
        'initialized_components': {},
        'performance_optimizations': []
    }
    
    try:
        # Initialize directory structure
        if globals().get('setup_directories'):
            try:
                setup_directories()
                initialization_result['initialized_components']['directories'] = True
                initialization_result['utilities_loaded'] += 1
                logger.info("✅ Directory structure initialized")
            except Exception as e:
                initialization_result['errors'].append(f"Directory setup failed: {e}")
        
        # Initialize enhanced directory manager if available
        if globals().get('EnhancedDirectoryManager'):
            try:
                dir_manager = EnhancedDirectoryManager()
                created_dirs = dir_manager.setup_directories()
                initialization_result['initialized_components']['enhanced_directories'] = True
                initialization_result['utilities_loaded'] += 1
                initialization_result['performance_optimizations'].append(f"Created {len(created_dirs)} directories")
                logger.info("✅ Enhanced directory manager initialized")
            except Exception as e:
                initialization_result['warnings'].append(f"Enhanced directories not available: {e}")
        
        # Initialize cache manager if available
        if globals().get('AdvancedCacheManager'):
            try:
                cache_manager = AdvancedCacheManager()
                # Perform initial cache cleanup
                cleanup_stats = cache_manager.cleanup_cache(max_age_days=7)
                initialization_result['initialized_components']['cache_manager'] = True
                initialization_result['utilities_loaded'] += 1
                initialization_result['performance_optimizations'].append(
                    f"Cache cleaned: {cleanup_stats.get('files_removed', 0)} files removed"
                )
                logger.info("✅ Advanced cache manager initialized")
            except Exception as e:
                initialization_result['warnings'].append(f"Cache manager not available: {e}")
        
        # Initialize text processor if available
        if globals().get('IntelligentTextProcessor'):
            try:
                text_processor = IntelligentTextProcessor()
                initialization_result['initialized_components']['text_processor'] = True
                initialization_result['utilities_loaded'] += 1
                initialization_result['performance_optimizations'].append("NLP models loaded for enhanced text processing")
                logger.info("✅ Intelligent text processor initialized")
            except Exception as e:
                initialization_result['warnings'].append(f"Enhanced text processor not available: {e}")
        
        # Initialize session state
        if globals().get('ensure_enhanced_session_state'):
            try:
                ensure_enhanced_session_state()
                initialization_result['initialized_components']['session_state'] = True
                initialization_result['utilities_loaded'] += 1
                logger.info("✅ Enhanced session state initialized")
            except Exception as e:
                if globals().get('ensure_session_state_keys'):
                    ensure_session_state_keys()
                    initialization_result['initialized_components']['basic_session_state'] = True
                    logger.info("✅ Basic session state initialized")
                else:
                    initialization_result['warnings'].append(f"Session state initialization failed: {e}")
        
        # Success criteria
        initialization_result['success'] = initialization_result['utilities_loaded'] > 0
        
        if initialization_result['success']:
            logger.info(f"🎉 Utilities initialized successfully!")
            logger.info(f"📊 Loaded: {initialization_result['utilities_loaded']} utilities")
            logger.info(f"⚡ Optimizations: {len(initialization_result['performance_optimizations'])}")
        
    except Exception as e:
        initialization_result['errors'].append(f"Critical initialization error: {e}")
        logger.error(f"💥 Utility initialization failed: {e}")
    
    return initialization_result

def get_feature_availability() -> Dict[str, bool]:
    """Get current feature availability status"""
    return FEATURE_AVAILABILITY.copy()

def create_utility_usage_report() -> Dict[str, Any]:
    """Create a comprehensive usage report for utilities"""
    
    available_utilities = [name for name in __all__ if globals().get(name) is not None]
    missing_utilities = [name for name in __all__ if globals().get(name) is None]
    
    # Categorize utilities
    categories = {
        'core_functions': [
            'setup_directories', 'validate_file_type', 'calculate_file_hash',
            'clean_text', 'chunk_text', 'extract_key_terms'
        ],
        'rag_pipeline': [
            'RAGPipeline', 'EnhancedRAGPipeline', 'AdvancedRAGPipeline',
            'create_rag_pipeline', 'create_advanced_rag_pipeline'
        ],
        'enhanced_classes': [
            'EnhancedDirectoryManager', 'AdvancedFileValidator',
            'IntelligentTextProcessor', 'AdvancedCacheManager'
        ],
        'utility_functions': [
            'format_response_with_citations', 'validate_knowledge_base',
            'create_progress_tracker', 'display_processing_summary'
        ]
    }
    
    category_availability = {}
    for category, utilities in categories.items():
        available_in_category = [util for util in utilities if util in available_utilities]
        category_availability[category] = {
            'available': len(available_in_category),
            'total': len(utilities),
            'percentage': (len(available_in_category) / len(utilities)) * 100 if utilities else 0,
            'missing': [util for util in utilities if util not in available_utilities]
        }
    
    return {
        'total_utilities': len(__all__),
        'available_utilities': len(available_utilities),
        'missing_utilities': len(missing_utilities),
        'availability_percentage': (len(available_utilities) / len(__all__)) * 100,
        'category_breakdown': category_availability,
        'feature_flags': FEATURE_AVAILABILITY,
        'missing_list': missing_utilities,
        'recommendations': _generate_usage_recommendations(category_availability)
    }

def _generate_usage_recommendations(category_availability: Dict[str, Dict]) -> List[str]:
    """Generate recommendations based on utility availability"""
    
    recommendations = []
    
    # Check core functions
    core_availability = category_availability.get('core_functions', {}).get('percentage', 0)
    if core_availability < 80:
        recommendations.append("🔧 Install missing core dependencies to enable basic functionality")
    
    # Check RAG pipeline
    rag_availability = category_availability.get('rag_pipeline', {}).get('percentage', 0)
    if rag_availability < 50:
        recommendations.append("🔍 Install sentence-transformers and faiss-cpu for advanced search capabilities")
    
    # Check enhanced classes
    enhanced_availability = category_availability.get('enhanced_classes', {}).get('percentage', 0)
    if enhanced_availability < 70:
        recommendations.append("🚀 Install enhanced dependencies (spaCy, textstat) for advanced text processing")
    
    # General recommendations
    if not FEATURE_AVAILABILITY['advanced_rag']:
        recommendations.append("📚 Consider upgrading to AdvancedRAGPipeline for better educational content retrieval")
    
    if not FEATURE_AVAILABILITY['enhanced_helpers']:
        recommendations.append("⚡ Enable enhanced helpers for improved performance and educational features")
    
    return recommendations

def display_utility_status():
    """Display comprehensive utility status"""
    
    info = get_utility_info()
    deps = check_utility_dependencies()
    report = create_utility_usage_report()
    
    print("🛠️ IntelliLearn AI Utilities Status")
    print("=" * 50)
    print(f"Version: {info['version']}")
    print(f"Available utilities: {info['available_functions']['total_available']}")
    print(f"Availability: {report['availability_percentage']:.1f}%")
    print()
    
    print("🎯 Feature Capabilities:")
    for feature, available in info['capabilities'].items():
        status = "✅" if available else "❌"
        print(f"  {status} {feature.replace('_', ' ').title()}")
    print()
    
    print("📊 Category Breakdown:")
    for category, stats in report['category_breakdown'].items():
        print(f"  {category.replace('_', ' ').title()}: {stats['available']}/{stats['total']} ({stats['percentage']:.0f}%)")
    print()
    
    print("🔧 Dependencies:")
    print(f"  Core: {deps['satisfaction_rates']['core']:.0f}%")
    print(f"  Enhanced: {deps['satisfaction_rates']['enhanced']:.0f}%")
    print(f"  Optional: {deps['satisfaction_rates']['optional']:.0f}%")
    print()
    
    if report['recommendations']:
        print("💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"  {rec}")

# Backward compatibility warnings
def _check_backward_compatibility():
    """Check and warn about backward compatibility issues"""
    
    # Check for renamed functions
    renames = {
        'create_progress_tracker': 'create_learning_progress_tracker',
        'estimate_reading_time': 'estimate_reading_time_enhanced'
    }
    
    for old_name, new_name in renames.items():
        if globals().get(old_name) is None and globals().get(new_name) is not None:
            logger.info(f"📝 {old_name} is available as {new_name} (enhanced version)")

# Package initialization
try:
    _check_backward_compatibility()
    
    available_count = len([name for name in __all__ if globals().get(name) is not None])
    total_count = len(__all__)
    
    logger.info(f"🛠️ IntelliLearn AI Utilities v{__version__} loaded")
    logger.info(f"📦 Available utilities: {available_count}/{total_count} ({(available_count/total_count)*100:.1f}%)")
    
    if available_count == total_count:
        logger.info("✅ All utilities available - full functionality enabled")
    elif available_count >= total_count * 0.8:
        logger.info("🎯 Most utilities available - enhanced functionality enabled")
    elif available_count >= total_count * 0.5:
        logger.warning("⚠️ Some utilities missing - basic functionality available")
    else:
        logger.error("❌ Many utilities missing - limited functionality")
    
    # Log feature availability
    available_features = [k for k, v in FEATURE_AVAILABILITY.items() if v]
    if available_features:
        logger.info(f"🚀 Available features: {', '.join(available_features)}")
    
except Exception as e:
    logger.error(f"💥 Utility package initialization error: {e}")

# Optional: Display full status on import (useful for debugging)
# Uncomment the next line to see detailed status on import
# display_utility_status()
