import re
import numpy as np
from collections import Counter
import streamlit as st
import fitz
from typing import List, Dict, Optional
import logging

# Try to import advanced document analysis libraries
try:
    from unstructured.partition.pdf import partition_pdf
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False

try:
    import layoutparser as lp
    LAYOUTPARSER_AVAILABLE = True
except ImportError:
    LAYOUTPARSER_AVAILABLE = False

class SuperiorChapterDetector:
    def __init__(self):
        """Initialize enhanced chapter detector with comprehensive patterns and models"""
        
        # Enhanced educational keywords
        self.educational_keywords = {
            'chapter', 'lesson', 'unit', 'section', 'part', 'introduction',
            'overview', 'summary', 'objectives', 'goals', 'learning', 'exercise',
            'practice', 'review', 'assessment', 'quiz', 'test', 'assignment',
            'conclusion', 'activity', 'workshop', 'tutorial', 'module', 'topic',
            'discussion', 'practical', 'experiment', 'project', 'case study'
        }
        
        # Enhanced chapter detection patterns (15+ patterns)
        self.advanced_patterns = [
            # Standard patterns
            r'^(chapter|lesson|unit|section|part)\s+\d+',
            r'^\d+\.\s+[A-Z][a-zA-Z\s]+',
            r'^[A-Z][A-Z\s]{5,}$',
            r'^\d+\s*[:-]\s*[A-Z][a-zA-Z\s]+',
            r'^(lesson|chapter|unit)\s*\d*\s*[:-]\s*[A-Z]',
            r'^\d+\.\d+\s+[A-Z]',
            
            # Educational-specific patterns (NEW)
            r'^MODULE\s*[-–]\s*\d+',           # MODULE - 2
            r'^LESSON\s+\d+',                  # LESSON 11
            r'^UNIT\s+\d+',                    # UNIT 5
            r'^(BIOLOGY|CHEMISTRY|PHYSICS|MATHEMATICS|HISTORY|SCIENCE)\s*\d*',  # Subject headers
            r'^\d+\.\d+\s+[A-Z\s]{5,}',       # 11.1 PHOTOSYNTHESIS
            r'^[A-Z\s]{8,}$',                 # All caps educational titles
            r'^part\s+[IVX]+',                # Part I, Part II
            r'^appendix\s+[A-Z]',             # Appendix A
            r'^exercise\s+\d+',               # Exercise 1
            r'^activity\s+\d+',               # Activity 1
            r'^Topic\s+\d+',                  # Topic 1
            r'^Section\s+[A-Z]\d*',           # Section A1
        ]
        
        # Skip patterns for non-content elements
        self.skip_patterns = [
            r'^page\s+\d+',
            r'^\d+\s*$',
            r'^©.*copyright',
            r'^all rights reserved',
            r'^table of contents',
            r'^index$',
            r'^bibliography$',
            r'^references$',
            r'^footnote',
            r'^figure\s+\d+',
            r'^table\s+\d+',
            r'^notes?$',
            r'^\d{1,3}$',  # Page numbers
        ]
        
        # Enhanced chapter types
        self.chapter_types = {
            'introduction': ['introduction', 'overview', 'getting started', 'preface', 'foreword', 'prologue'],
            'content': ['chapter', 'lesson', 'unit', 'section', 'topic', 'module', 'part'],
            'practice': ['exercise', 'practice', 'problem', 'quiz', 'test', 'assignment', 'activity', 'workshop'],
            'summary': ['summary', 'conclusion', 'review', 'recap', 'wrap-up', 'key points'],
            'reference': ['appendix', 'reference', 'glossary', 'index', 'bibliography', 'further reading'],
            'objectives': ['objective', 'goal', 'outcome', 'learning outcomes', 'aims', 'targets']
        }
        
        # Initialize layout parser if available
        self.layout_model = None
        if LAYOUTPARSER_AVAILABLE:
            try:
                self.layout_model = lp.Detectron2LayoutModel(
                    'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
                )
                st.info("✅ Advanced layout analysis enabled")
            except Exception as e:
                st.warning(f"Layout analysis not available: {e}")
                self.layout_model = None

    def detect_chapters_enhanced(self, doc):
        """Enhanced chapter detection with multiple methods"""
        try:
            chapters = {}
            
            # Method 1: Advanced document structure analysis
            if UNSTRUCTURED_AVAILABLE:
                chapters_method1 = self._detect_with_unstructured(doc)
                if chapters_method1:
                    chapters.update(chapters_method1)
                    st.success(f"🔍 Method 1: Found {len(chapters_method1)} chapters with advanced parsing")
            
            # Method 2: Enhanced pattern matching
            chapters_method2 = self._detect_with_enhanced_patterns(doc)
            if chapters_method2:
                chapters = self._merge_chapter_results(chapters, chapters_method2)
                st.success(f"🔍 Method 2: Found {len(chapters_method2)} chapters with pattern matching")
            
            # Method 3: Layout-based detection (if available)
            if self.layout_model:
                chapters_method3 = self._detect_with_layout_analysis(doc)
                if chapters_method3:
                    chapters = self._merge_chapter_results(chapters, chapters_method3)
                    st.success(f"🔍 Method 3: Found {len(chapters_method3)} chapters with layout analysis")
            
            # Method 4: Educational content analysis
            chapters_method4 = self._detect_educational_structure(doc)
            if chapters_method4:
                chapters = self._merge_chapter_results(chapters, chapters_method4)
                st.success(f"🔍 Method 4: Found {len(chapters_method4)} chapters with educational analysis")
            
            # Validate and enhance final results
            validated_chapters = self._validate_and_enhance_chapters(chapters)
            
            if validated_chapters:
                st.success(f"🎉 Total chapters detected: {len(validated_chapters)}")
                return validated_chapters
            else:
                st.warning("⚠️ No chapters detected, using fallback method")
                return self._fallback_chapter_detection(doc)
                
        except Exception as e:
            st.error(f"Enhanced chapter detection failed: {str(e)}")
            return self._fallback_chapter_detection(doc)

    def _detect_with_unstructured(self, doc):
        """Use Unstructured library for intelligent document parsing"""
        if not UNSTRUCTURED_AVAILABLE:
            return {}
        
        try:
            # Save document temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                doc.save(tmp_file.name)
                
                # Use Unstructured for advanced parsing
                elements = partition_pdf(
                    filename=tmp_file.name,
                    strategy="hi_res",
                    include_page_breaks=True,
                    infer_table_structure=True
                )
                
                chapters = {}
                current_chapter = None
                current_content = []
                
                for element in elements:
                    text = str(element).strip()
                    
                    if hasattr(element, 'category') and element.category == 'Title':
                        # This is likely a title/heading
                        if self._is_likely_chapter_title(text):
                            # Save previous chapter
                            if current_chapter and current_content:
                                chapters[current_chapter] = self._create_enhanced_chapter_data(
                                    current_chapter, current_content
                                )
                            
                            # Start new chapter
                            current_chapter = self._clean_chapter_name(text)
                            current_content = []
                    else:
                        # Regular content
                        if self._is_meaningful_content(text):
                            current_content.append(text)
                
                # Add final chapter
                if current_chapter and current_content:
                    chapters[current_chapter] = self._create_enhanced_chapter_data(
                        current_chapter, current_content
                    )
                
                # Cleanup
                import os
                os.unlink(tmp_file.name)
                
                return chapters
                
        except Exception as e:
            st.warning(f"Unstructured parsing failed: {e}")
            return {}

    def _detect_with_enhanced_patterns(self, doc):
        """Enhanced pattern-based detection"""
        chapters = {}
        current_chapter = None
        current_content = []
        
        # Analyze document structure first
        font_analysis = self._analyze_document_structure_enhanced(doc)
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if not text or len(text) < 3:
                                continue
                            
                            # Enhanced chapter title detection
                            is_chapter = self._is_chapter_title_enhanced(
                                text, span, font_analysis, page_num
                            )
                            
                            if is_chapter:
                                # Save previous chapter
                                if current_chapter and current_content:
                                    chapters[current_chapter] = self._create_enhanced_chapter_data(
                                        current_chapter, current_content
                                    )
                                
                                # Start new chapter
                                current_chapter = self._clean_chapter_name(text)
                                current_content = []
                                
                            else:
                                # Add meaningful content
                                if self._is_meaningful_content(text):
                                    current_content.append(text)
        
        # Add final chapter
        if current_chapter and current_content:
            chapters[current_chapter] = self._create_enhanced_chapter_data(
                current_chapter, current_content
            )
        
        return chapters

    def _detect_educational_structure(self, doc):
        """Detect educational document structure patterns"""
        chapters = {}
        
        # Look for educational patterns in first few pages
        structure_info = self._analyze_educational_patterns(doc)
        
        if structure_info['has_educational_structure']:
            # Use educational-specific detection
            chapters = self._extract_educational_chapters(doc, structure_info)
        
        return chapters

    def _analyze_educational_patterns(self, doc):
        """Analyze if document follows educational patterns"""
        structure_info = {
            'has_educational_structure': False,
            'subject': None,
            'module_number': None,
            'lesson_format': None,
            'numbering_system': None
        }
        
        # Analyze first 3 pages for educational indicators
        combined_text = ""
        for page_num in range(min(3, doc.page_count)):
            page = doc[page_num]
            combined_text += page.get_text() + " "
        
        text_upper = combined_text.upper()
        
        # Check for subject indicators
        subjects = ['BIOLOGY', 'CHEMISTRY', 'PHYSICS', 'MATHEMATICS', 'HISTORY', 'SCIENCE', 'ENGLISH']
        for subject in subjects:
            if subject in text_upper:
                structure_info['subject'] = subject
                structure_info['has_educational_structure'] = True
                break
        
        # Check for module patterns
        module_match = re.search(r'MODULE\s*[-–]?\s*(\d+)', text_upper)
        if module_match:
            structure_info['module_number'] = module_match.group(1)
            structure_info['has_educational_structure'] = True
        
        # Check for lesson patterns
        if 'LESSON' in text_upper:
            structure_info['lesson_format'] = 'lesson'
            structure_info['has_educational_structure'] = True
        
        # Check numbering system (11.1, 11.2 format)
        if re.search(r'\d+\.\d+\s+[A-Z]', combined_text):
            structure_info['numbering_system'] = 'decimal'
            structure_info['has_educational_structure'] = True
        
        return structure_info

    def _extract_educational_chapters(self, doc, structure_info):
        """Extract chapters using educational structure information"""
        chapters = {}
        current_chapter = None
        current_content = []
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for educational chapter patterns
                is_chapter = False
                
                # Subject-specific chapter detection
                if structure_info['subject']:
                    if structure_info['subject'] in line.upper() and len(line) < 100:
                        is_chapter = True
                
                # Module-specific patterns
                if structure_info['module_number']:
                    module_pattern = f"MODULE\\s*[-–]?\\s*{structure_info['module_number']}"
                    if re.search(module_pattern, line, re.IGNORECASE):
                        is_chapter = True
                
                # Lesson patterns
                if structure_info['lesson_format'] == 'lesson':
                    if re.match(r'^LESSON\s+\d+', line.upper()):
                        is_chapter = True
                
                # Decimal numbering system (11.1, 11.2)
                if structure_info['numbering_system'] == 'decimal':
                    if re.match(r'^\d+\.\d+\s+[A-Z]', line):
                        is_chapter = True
                
                # All caps educational titles
                if line.isupper() and len(line) > 8 and len(line) < 80:
                    # Check if it contains educational keywords
                    if any(keyword.upper() in line for keyword in self.educational_keywords):
                        is_chapter = True
                
                if is_chapter:
                    # Save previous chapter
                    if current_chapter and current_content:
                        chapters[current_chapter] = self._create_enhanced_chapter_data(
                            current_chapter, current_content
                        )
                    
                    # Start new chapter
                    current_chapter = self._clean_chapter_name(line)
                    current_content = []
                else:
                    # Add meaningful content
                    if self._is_meaningful_content(line):
                        current_content.append(line)
        
        # Add final chapter
        if current_chapter and current_content:
            chapters[current_chapter] = self._create_enhanced_chapter_data(
                current_chapter, current_content
            )
        
        return chapters

    def _analyze_document_structure_enhanced(self, doc):
        """Enhanced document structure analysis"""
        font_sizes = []
        font_families = []
        font_flags = []
        page_positions = []
        text_lengths = []
        
        # Analyze more pages for better statistics
        sample_pages = min(20, doc.page_count)
        
        for page_num in range(sample_pages):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_sizes.append(span.get("size", 12))
                            font_families.append(span.get("font", "default"))
                            font_flags.append(span.get("flags", 0))
                            text_lengths.append(len(span.get("text", "")))
                            
                            bbox = span.get("bbox", [0, 0, 0, 0])
                            if len(bbox) >= 4:
                                page_positions.append(bbox[1])
        
        if font_sizes:
            analysis = {
                'chapter_threshold': np.percentile(font_sizes, 90),  # Higher threshold
                'large_font_threshold': np.percentile(font_sizes, 80),
                'average_font_size': np.mean(font_sizes),
                'font_variance': np.var(font_sizes),
                'max_font_size': np.max(font_sizes),
                'min_font_size': np.min(font_sizes),
                'common_fonts': Counter(font_families).most_common(5),
                'bold_usage': sum(1 for flag in font_flags if flag & 2**4) / len(font_flags),
                'italic_usage': sum(1 for flag in font_flags if flag & 2**3) / len(font_flags),
                'page_top_threshold': np.percentile(page_positions, 25) if page_positions else 0,
                'average_text_length': np.mean(text_lengths)
            }
        else:
            # Fallback values
            analysis = {
                'chapter_threshold': 16,
                'large_font_threshold': 14,
                'average_font_size': 12,
                'font_variance': 4,
                'max_font_size': 20,
                'min_font_size': 8,
                'common_fonts': [('default', 100)],
                'bold_usage': 0.1,
                'italic_usage': 0.05,
                'page_top_threshold': 50,
                'average_text_length': 30
            }
        
        return analysis

    def _is_chapter_title_enhanced(self, text, span, font_analysis, page_num):
        """Enhanced multi-criteria chapter title detection"""
        
        # Skip obvious non-chapter elements
        if self._should_skip_text(text):
            return False
        
        # Get span properties
        font_size = span.get("size", 12)
        font_flags = span.get("flags", 0)
        bbox = span.get("bbox", [0, 0, 0, 0])
        
        # Calculate comprehensive detection scores
        scores = self._calculate_enhanced_detection_scores(
            text, font_size, font_flags, bbox, font_analysis, page_num
        )
        
        # Enhanced weighted scoring system
        total_score = (
            scores['pattern'] * 0.35 +        # Pattern matching (increased weight)
            scores['formatting'] * 0.25 +     # Font/formatting
            scores['position'] * 0.15 +       # Page position
            scores['structure'] * 0.15 +      # Text structure
            scores['educational'] * 0.10      # Educational context
        )
        
        # Dynamic threshold based on document characteristics
        threshold = 0.6
        
        # Adjust threshold based on document analysis
        if font_analysis['font_variance'] > 15:  # High variance document
            threshold = 0.5
        elif font_analysis['bold_usage'] < 0.03:  # Minimal formatting
            threshold = 0.4
        elif font_analysis['max_font_size'] - font_analysis['min_font_size'] > 10:  # Large font range
            threshold = 0.5
        
        return total_score >= threshold

    def _calculate_enhanced_detection_scores(self, text, font_size, font_flags, bbox, font_analysis, page_num):
        """Calculate comprehensive detection scores with more factors"""
        
        text_lower = text.lower().strip()
        
        # 1. Enhanced pattern matching score
        pattern_score = 0.0
        for i, pattern in enumerate(self.advanced_patterns):
            if re.match(pattern, text_lower):
                # Give higher scores to more specific educational patterns
                if i >= 6:  # Educational-specific patterns
                    pattern_score = 1.0
                else:
                    pattern_score = 0.8
                break
        
        # 2. Enhanced formatting score
        is_very_large = font_size >= font_analysis['chapter_threshold']
        is_large = font_size >= font_analysis['large_font_threshold']
        is_bold = font_flags & 2**4
        is_italic = font_flags & 2**3
        
        formatting_score = 0.0
        if is_very_large:
            formatting_score += 0.7
        elif is_large:
            formatting_score += 0.4
        
        if is_bold:
            formatting_score += 0.4
        
        if is_italic:
            formatting_score += 0.1
            
        # Font size relative to document average
        if font_size > font_analysis['average_font_size'] * 1.5:
            formatting_score += 0.3
        
        formatting_score = min(formatting_score, 1.0)
        
        # 3. Enhanced position score
        position_score = 0.0
        if len(bbox) >= 4:
            y_position = bbox[1]
            if y_position <= font_analysis['page_top_threshold']:
                position_score = 0.9
            elif y_position <= font_analysis['page_top_threshold'] * 1.5:
                position_score = 0.6
            elif y_position <= font_analysis['page_top_threshold'] * 2:
                position_score = 0.3
        
        # 4. Enhanced structure score
        word_count = len(text.split())
        char_count = len(text)
        
        is_appropriate_length = 2 <= word_count <= 15
        is_short = word_count <= 8
        starts_with_capital = text[0].isupper() if text else False
        has_number = bool(re.search(r'\d+', text))
        is_all_caps = text.isupper() and len(text) > 5
        ends_properly = not text.endswith(('.', ',', ';', ':', '!', '?'))
        has_educational_structure = bool(re.search(r'(chapter|lesson|unit|section|part|module)', text_lower))
        
        structure_score = 0.0
        if is_appropriate_length:
            structure_score += 0.3
        if is_short:
            structure_score += 0.2
        if starts_with_capital:
            structure_score += 0.2
        if has_number:
            structure_score += 0.3
        if is_all_caps and word_count > 1:
            structure_score += 0.4
        if ends_properly:
            structure_score += 0.2
        if has_educational_structure:
            structure_score += 0.3
        
        structure_score = min(structure_score, 1.0)
        
        # 5. Enhanced educational context score
        educational_score = 0.0
        
        # Educational keywords
        keyword_matches = sum(1 for keyword in self.educational_keywords if keyword in text_lower)
        educational_score += min(keyword_matches * 0.3, 0.6)
        
        # Subject-specific terms
        subject_terms = ['biology', 'chemistry', 'physics', 'mathematics', 'history', 'science']
        if any(term in text_lower for term in subject_terms):
            educational_score += 0.4
        
        # Educational numbering patterns
        if re.search(r'\d+\.\d+|\d+\s*[:-]\s*', text):
            educational_score += 0.3
        
        educational_score = min(educational_score, 1.0)
        
        return {
            'pattern': pattern_score,
            'formatting': formatting_score,
            'position': position_score,
            'structure': structure_score,
            'educational': educational_score
        }

    def _is_likely_chapter_title(self, text):
        """Check if text is likely a chapter title"""
        text_lower = text.lower().strip()
        
        # Check against all patterns
        for pattern in self.advanced_patterns:
            if re.match(pattern, text_lower):
                return True
        
        # Additional checks for educational content
        if len(text.split()) <= 10 and text[0].isupper():
            # Check for educational keywords
            if any(keyword in text_lower for keyword in self.educational_keywords):
                return True
        
        return False

    def _create_enhanced_chapter_data(self, chapter_name, content_list):
        """Create enhanced chapter data structure"""
        content = '\n'.join(content_list)
        word_count = len(content.split())
        
        return {
            'content': content,
            'type': self._categorize_chapter_type_enhanced(chapter_name, content),
            'word_count': word_count,
            'estimated_reading_time': max(1, word_count // 200),
            'paragraph_count': len([p for p in content_list if len(p.strip()) > 50]),
            'content_quality': self._assess_content_quality_enhanced(content),
            'educational_level': self._estimate_educational_level_enhanced(content, chapter_name),
            'readability': self._calculate_readability_enhanced(content),
            'key_concepts': self._extract_key_concepts_basic(content),
            'difficulty_indicators': self._analyze_difficulty_indicators(content),
            'learning_objectives_count': self._count_learning_objectives(content)
        }

    def _categorize_chapter_type_enhanced(self, chapter_name, content):
        """Enhanced chapter type categorization"""
        name_lower = chapter_name.lower().strip()
        content_lower = content.lower()
        
        # Check name patterns first
        for chapter_type, keywords in self.chapter_types.items():
            if any(keyword in name_lower for keyword in keywords):
                return chapter_type
        
        # Check content patterns
        if any(word in content_lower for word in ['objective', 'goal', 'aim', 'outcome']):
            return 'objectives'
        elif any(word in content_lower for word in ['exercise', 'problem', 'question', 'practice']):
            return 'practice'
        elif any(word in content_lower for word in ['summary', 'conclusion', 'key points', 'review']):
            return 'summary'
        elif any(word in content_lower for word in ['introduction', 'overview', 'background']):
            return 'introduction'
        else:
            return 'content'

    def _assess_content_quality_enhanced(self, content):
        """Enhanced content quality assessment"""
        if not content:
            return 'empty'
        
        word_count = len(content.split())
        sentence_count = len([s for s in content.split('.') if s.strip()])
        paragraph_indicators = content.count('\n\n') + content.count('\n')
        
        # Enhanced quality metrics
        has_structure = bool(re.search(r'(first|second|third|next|finally|conclusion)', content.lower()))
        has_examples = bool(re.search(r'(example|instance|such as|for example)', content.lower()))
        has_definitions = bool(re.search(r'(definition|define|means|refers to)', content.lower()))
        
        quality_score = 0
        
        if word_count < 50:
            quality_score = 1  # minimal
        elif word_count < 200:
            quality_score = 2  # brief
        elif word_count < 1000:
            quality_score = 3  # moderate
        else:
            quality_score = 4  # comprehensive
        
        # Bonus for educational structure
        if has_structure:
            quality_score += 0.5
        if has_examples:
            quality_score += 0.5
        if has_definitions:
            quality_score += 0.5
        
        quality_levels = ['empty', 'minimal', 'brief', 'moderate', 'good', 'comprehensive', 'excellent']
        return quality_levels[min(int(quality_score), len(quality_levels) - 1)]

    def _estimate_educational_level_enhanced(self, content, chapter_name):
        """Enhanced educational level estimation"""
        name_lower = chapter_name.lower()
        content_lower = content.lower()
        
        # Advanced level indicators
        advanced_indicators = [
            'analysis', 'synthesis', 'evaluation', 'hypothesis', 'methodology',
            'theoretical', 'empirical', 'correlation', 'regression', 'statistical',
            'complex', 'advanced', 'sophisticated', 'comprehensive'
        ]
        
        # High school indicators
        high_school_indicators = [
            'algebra', 'geometry', 'calculus', 'biology', 'chemistry', 'physics',
            'equation', 'formula', 'theorem', 'principle', 'law'
        ]
        
        # Middle school indicators
        middle_school_indicators = [
            'basic', 'introduction', 'simple', 'fundamental', 'elementary',
            'overview', 'getting started'
        ]
        
        advanced_count = sum(1 for indicator in advanced_indicators if indicator in content_lower)
        high_school_count = sum(1 for indicator in high_school_indicators if indicator in content_lower)
        middle_school_count = sum(1 for indicator in middle_school_indicators if indicator in content_lower)
        
        word_count = len(content.split())
        sentences = content.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()]) if sentences else 10
        
        # Scoring system
        level_score = 0
        
        if advanced_count > 3 or avg_sentence_length > 25:
            level_score += 3
        if high_school_count > 2:
            level_score += 2
        if middle_school_count > 2:
            level_score -= 1
        
        if word_count > 1000:
            level_score += 1
        elif word_count < 300:
            level_score -= 1
        
        if level_score >= 3:
            return 'advanced'
        elif level_score >= 1:
            return 'high_school'
        else:
            return 'middle_school'

    def _calculate_readability_enhanced(self, content):
        """Enhanced readability calculation"""
        if not content:
            return 'unknown'
        
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        words = content.split()
        syllable_count = 0
        
        if not sentences or not words:
            return 'unknown'
        
        # Estimate syllables (rough approximation)
        for word in words:
            syllable_count += max(1, len(re.findall(r'[aeiouyAEIOUY]', word)))
        
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllable_count / len(words)
        
        # Flesch Reading Ease approximation
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        if flesch_score >= 90:
            return 'very_easy'
        elif flesch_score >= 80:
            return 'easy'
        elif flesch_score >= 70:
            return 'fairly_easy'
        elif flesch_score >= 60:
            return 'standard'
        elif flesch_score >= 50:
            return 'fairly_difficult'
        elif flesch_score >= 30:
            return 'difficult'
        else:
            return 'very_difficult'

    def _extract_key_concepts_basic(self, content):
        """Extract key concepts from content (basic NLP)"""
        concepts = set()
        
        # Pattern-based concept extraction
        concept_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns
            r'\b\w+(?:tion|sion|ment|ness|ity|ism|ogy|graphy)\b',  # Academic suffixes
            r'\b(?:principle|theory|law|rule|concept|method|process|system|model)\s+of\s+\w+',
        ]
        
        for pattern in concept_patterns:
            matches = re.findall(pattern, content)
            concepts.update(match for match in matches if len(match) > 3 and len(match) < 30)
        
        # Remove common stop words and filter
        stop_words = {'this', 'that', 'with', 'from', 'they', 'have', 'were', 'been'}
        filtered_concepts = [concept for concept in concepts 
                           if concept.lower() not in stop_words and len(concept.split()) <= 3]
        
        return filtered_concepts[:10]

    def _analyze_difficulty_indicators(self, content):
        """Analyze difficulty indicators in content"""
        content_lower = content.lower()
        
        difficulty_indicators = {
            'mathematical_notation': len(re.findall(r'[∑∏∫∂αβγδεφπλμσω]|\\[a-zA-Z]+', content)),
            'complex_sentences': len([s for s in content.split('.') if len(s.split()) > 30]),
            'technical_terms': len(re.findall(r'\b\w{12,}\b', content)),
            'passive_voice': len(re.findall(r'\bis\s+\w+ed\b|\bare\s+\w+ed\b|\bwas\s+\w+ed\b|\bwere\s+\w+ed\b', content_lower)),
            'subordinate_clauses': len(re.findall(r'\b(although|because|since|while|whereas|if|unless|until)\b', content_lower))
        }
        
        return difficulty_indicators

    def _count_learning_objectives(self, content):
        """Count learning objectives in content"""
        objective_patterns = [
            r'(?i)objective[s]?\s*\d*\s*[:\-]',
            r'(?i)learning\s+outcome[s]?\s*\d*\s*[:\-]',
            r'(?i)students?\s+will\s+(be\s+able\s+to|understand|learn|know)',
            r'(?i)after\s+this\s+(lesson|chapter|unit)',
            r'(?i)by\s+the\s+end\s+of\s+this'
        ]
        
        count = 0
        for pattern in objective_patterns:
            count += len(re.findall(pattern, content))
        
        return count

    def _merge_chapter_results(self, chapters1, chapters2):
        """Merge results from different detection methods"""
        merged = chapters1.copy()
        
        for chapter_name, chapter_data in chapters2.items():
            if chapter_name in merged:
                # Merge content if chapters overlap
                existing_content = merged[chapter_name]['content']
                new_content = chapter_data['content']
                
                # Only merge if new content adds significant value
                if len(new_content) > len(existing_content) * 0.5:
                    merged[chapter_name]['content'] = existing_content + '\n\n' + new_content
                    merged[chapter_name]['word_count'] = len(merged[chapter_name]['content'].split())
            else:
                merged[chapter_name] = chapter_data
        
        return merged

    def _validate_and_enhance_chapters(self, chapters):
        """Enhanced validation and chapter enhancement"""
        validated_chapters = {}
        
        for chapter_name, chapter_data in chapters.items():
            # Enhanced validation criteria
            word_count = chapter_data.get('word_count', 0)
            content_quality = chapter_data.get('content_quality', 'minimal')
            
            # Skip chapters that are too short or low quality
            if word_count < 20 or content_quality == 'empty':
                continue
            
            # Skip duplicate or very similar chapters
            if not self._is_duplicate_chapter_enhanced(chapter_name, validated_chapters):
                validated_chapters[chapter_name] = chapter_data
        
        return validated_chapters

    def _is_duplicate_chapter_enhanced(self, chapter_name, existing_chapters):
        """Enhanced duplicate chapter detection"""
        for existing_name in existing_chapters.keys():
            # Exact match
            if chapter_name.lower() == existing_name.lower():
                return True
            
            # High similarity match
            similarity = self._calculate_name_similarity_enhanced(chapter_name, existing_name)
            if similarity > 0.85:
                return True
            
            # Substring match for very similar names
            if len(chapter_name) > 10 and len(existing_name) > 10:
                if chapter_name.lower() in existing_name.lower() or existing_name.lower() in chapter_name.lower():
                    return True
        
        return False

    def _calculate_name_similarity_enhanced(self, name1, name2):
        """Enhanced name similarity calculation"""
        words1 = set(name1.lower().split())
        words2 = set(name2.lower().split())
        
        if not words1 or not words2:
            return 0
        
        # Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        jaccard = len(intersection) / len(union)
        
        # Character-level similarity
        char_similarity = len(set(name1.lower()).intersection(set(name2.lower()))) / len(set(name1.lower()).union(set(name2.lower())))
        
        # Combined similarity
        return (jaccard * 0.7) + (char_similarity * 0.3)

    def _should_skip_text(self, text):
        """Enhanced text skipping logic"""
        text_lower = text.lower().strip()
        
        # Skip based on patterns
        for pattern in self.skip_patterns:
            if re.match(pattern, text_lower):
                return True
        
        # Skip very long text (likely paragraph content)
        if len(text.split()) > 25:
            return True
        
        # Skip very short text
        if len(text.strip()) < 3:
            return True
        
        # Skip text that's mostly numbers or symbols
        if len(re.sub(r'[^a-zA-Z\s]', '', text)) < len(text) * 0.4:
            return True
        
        # Skip common header/footer patterns
        common_skips = ['notes', 'biology', 'chemistry', 'physics', 'mathematics', 'page']
        if text_lower in common_skips:
            return True
        
        return False

    def _is_meaningful_content(self, text):
        """Enhanced meaningful content detection"""
        if len(text.strip()) < 15:  # Minimum content length
            return False
        
        text_lower = text.lower().strip()
        
        # Skip obvious non-content
        skip_indicators = [
            r'^page\s+\d+',
            r'^\d+\s*$',
            r'^chapter\s+\d+\s*$',
            r'^©.*',
            r'^all rights reserved',
            r'^figure\s+\d+',
            r'^table\s+\d+',
            r'^\d+\s*\.\s*\d+\s*$',
            r'^[a-z]\)\s*$',
            r'^\([a-z0-9]+\)\s*$',
            r'^notes?\s*$',
            r'^\d+\s*$'
        ]
        
        for pattern in skip_indicators:
            if re.match(pattern, text_lower):
                return False
        
        # Check for meaningful content indicators
        meaningful_indicators = [
            r'[.!?]',  # Sentence endings
            r'\b(the|and|is|are|was|were|in|on|at|to|for|of|with|this|that)\b',  # Common words
            r'[a-zA-Z]{4,}',  # Longer words
            r'(because|therefore|however|although|since|while)',  # Connecting words
        ]
        
        meaningful_count = sum(1 for pattern in meaningful_indicators if re.search(pattern, text_lower))
        
        # Must have at least 3 meaningful indicators
        return meaningful_count >= 3

    def _clean_chapter_name(self, raw_name):
        """Enhanced chapter name cleaning"""
        # Remove extra whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', raw_name.strip())
        
        # Remove common prefixes that might be formatting artifacts
        cleaned = re.sub(r'^[^\w\s]*', '', cleaned).strip()
        
        # Handle all caps titles
        if cleaned.isupper() and len(cleaned) > 10:
            # Convert to title case but preserve acronyms
            words = cleaned.split()
            title_words = []
            for word in words:
                if len(word) <= 3 and word.isupper():
                    title_words.append(word)  # Keep short acronyms
                else:
                    title_words.append(word.title())
            cleaned = ' '.join(title_words)
        
        # Remove trailing page numbers, dots, or other artifacts
        cleaned = re.sub(r'\s*\d+\s*$', '', cleaned).strip()
        cleaned = re.sub(r'\s*[\.\-_]+\s*$', '', cleaned).strip()
        
        # Remove module/lesson prefixes if they're standalone
        if re.match(r'^(module|lesson|unit|section|part)\s*[-–]?\s*\d*\s*$', cleaned.lower()):
            return cleaned  # Keep if it's just the identifier
        
        # Ensure reasonable length
        if not cleaned or len(cleaned) < 3:
            cleaned = "Untitled Chapter"
        elif len(cleaned) > 150:
            cleaned = cleaned[:150] + "..."
        
        return cleaned

    def _fallback_chapter_detection(self, doc):
        """Enhanced fallback detection when advanced methods fail"""
        st.warning("Using enhanced fallback chapter detection method")
        
        chapters = {}
        current_chapter = None
        current_content = []
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Enhanced fallback detection
                if self._is_enhanced_fallback_chapter(line):
                    if current_chapter and current_content:
                        chapters[current_chapter] = self._create_enhanced_chapter_data(
                            current_chapter, current_content
                        )
                    
                    current_chapter = self._clean_chapter_name(line)
                    current_content = []
                else:
                    if self._is_meaningful_content(line):
                        current_content.append(line)
        
        # Add final chapter
        if current_chapter and current_content:
            chapters[current_chapter] = self._create_enhanced_chapter_data(
                current_chapter, current_content
            )
        
        return chapters

    def _is_enhanced_fallback_chapter(self, line):
        """Enhanced fallback chapter detection"""
        line_lower = line.lower().strip()
        
        # Basic patterns with higher threshold
        fallback_patterns = [
            r'^chapter\s+\d+',
            r'^lesson\s+\d+',
            r'^unit\s+\d+',
            r'^section\s+\d+',
            r'^module\s*[-–]?\s*\d+',
            r'^\d+\.\s+[A-Z]',
            r'^\d+\.\d+\s+[A-Z]'
        ]
        
        for pattern in fallback_patterns:
            if re.match(pattern, line_lower):
                return True
        
        # All caps educational titles (more restrictive)
        if (line.isupper() and 
            8 <= len(line) <= 60 and 
            len(line.split()) >= 2 and
            not re.search(r'\d{3,}', line)):  # No long numbers
            return True
        
        # Mixed case with educational keywords
        if (len(line.split()) <= 10 and 
            line[0].isupper() and
            any(keyword in line_lower for keyword in ['introduction', 'overview', 'summary', 'conclusion'])):
            return True
        
        return False

    def get_detection_statistics(self, chapters):
        """Get comprehensive statistics about the detection process"""
        if not chapters:
            return {'total_chapters': 0}
        
        stats = {
            'total_chapters': len(chapters),
            'chapter_types': {},
            'total_words': 0,
            'average_chapter_length': 0,
            'content_quality_distribution': {},
            'educational_levels': {},
            'readability_distribution': {},
            'total_concepts': 0,
            'average_reading_time': 0
        }
        
        for chapter_data in chapters.values():
            # Count chapter types
            chapter_type = chapter_data.get('type', 'unknown')
            stats['chapter_types'][chapter_type] = stats['chapter_types'].get(chapter_type, 0) + 1
            
            # Sum metrics
            stats['total_words'] += chapter_data.get('word_count', 0)
            
            # Count distributions
            quality = chapter_data.get('content_quality', 'unknown')
            stats['content_quality_distribution'][quality] = stats['content_quality_distribution'].get(quality, 0) + 1
            
            level = chapter_data.get('educational_level', 'unknown')
            stats['educational_levels'][level] = stats['educational_levels'].get(level, 0) + 1
            
            readability = chapter_data.get('readability', 'unknown')
            stats['readability_distribution'][readability] = stats['readability_distribution'].get(readability, 0) + 1
            
            # Count concepts
            concepts = chapter_data.get('key_concepts', [])
            stats['total_concepts'] += len(concepts)
            
            # Sum reading time
            stats['average_reading_time'] += chapter_data.get('estimated_reading_time', 0)
        
        # Calculate averages
        if chapters:
            stats['average_chapter_length'] = stats['total_words'] // len(chapters)
            stats['average_concepts_per_chapter'] = stats['total_concepts'] / len(chapters)
        
        return stats
