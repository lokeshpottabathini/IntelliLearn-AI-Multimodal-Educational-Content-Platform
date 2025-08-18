import re
import numpy as np
from collections import Counter
import streamlit as st

class EnhancedChapterDetector:
    def __init__(self):
        """Initialize enhanced chapter detector with comprehensive patterns"""
        self.educational_keywords = {
            'chapter', 'lesson', 'unit', 'section', 'part', 'introduction',
            'overview', 'summary', 'objectives', 'goals', 'learning', 'exercise',
            'practice', 'review', 'assessment', 'quiz', 'test', 'assignment',
            'conclusion', 'activity', 'workshop', 'tutorial', 'module', 'topic'
        }
        
        # Enhanced chapter detection patterns
        self.chapter_patterns = [
            r'^(chapter|lesson|unit|section|part)\s+\d+',  # Chapter 1, Lesson 2
            r'^\d+\.\s+[A-Z][a-zA-Z\s]+',                 # 1. Introduction
            r'^[A-Z][A-Z\s]{5,}$',                        # ALL CAPS titles
            r'^\d+\s*[:-]\s*[A-Z][a-zA-Z\s]+',           # 1: Introduction
            r'^(lesson|chapter|unit)\s*\d*\s*[:-]\s*[A-Z]', # Chapter: Title
            r'^\d+\.\d+\s+[A-Z]',                         # 1.1 Subtopic
            r'^part\s+[IVX]+',                            # Part I, Part II
            r'^appendix\s+[A-Z]',                         # Appendix A
            r'^exercise\s+\d+',                           # Exercise 1
            r'^activity\s+\d+',                           # Activity 1
        ]
        
        # Patterns for non-content elements to skip
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
            r'^table\s+\d+'
        ]
        
        # Chapter type classification
        self.chapter_types = {
            'introduction': ['introduction', 'overview', 'getting started', 'preface', 'foreword'],
            'content': ['chapter', 'lesson', 'unit', 'section', 'topic', 'module'],
            'practice': ['exercise', 'practice', 'problem', 'quiz', 'test', 'assignment', 'activity'],
            'summary': ['summary', 'conclusion', 'review', 'recap', 'wrap-up'],
            'reference': ['appendix', 'reference', 'glossary', 'index', 'bibliography'],
            'objectives': ['objective', 'goal', 'outcome', 'learning outcomes', 'aims']
        }

    def detect_chapters_enhanced(self, doc):
        """Enhanced chapter detection with comprehensive analysis"""
        try:
            chapters = {}
            current_chapter = None
            current_content = []
            
            # Analyze document structure for better detection
            font_analysis = self._analyze_document_structure(doc)
            
            # Process each page
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
                                        chapters[current_chapter] = self._create_chapter_data(
                                            current_chapter, current_content
                                        )
                                    
                                    # Start new chapter
                                    current_chapter = self._clean_chapter_name(text)
                                    current_content = []
                                    
                                    # Add debug info
                                    st.write(f"📖 Detected chapter: {current_chapter}")
                                    
                                else:
                                    # Add meaningful content only
                                    if self._is_meaningful_content(text):
                                        current_content.append(text)
            
            # Add final chapter
            if current_chapter and current_content:
                chapters[current_chapter] = self._create_chapter_data(
                    current_chapter, current_content
                )
            
            # Validate and enhance detected chapters
            validated_chapters = self._validate_and_enhance_chapters(chapters)
            
            return validated_chapters
            
        except Exception as e:
            st.error(f"Enhanced chapter detection failed: {str(e)}")
            return self._fallback_chapter_detection(doc)

    def _analyze_document_structure(self, doc):
        """Comprehensive document structure analysis"""
        font_sizes = []
        font_families = []
        font_flags = []
        page_positions = []
        
        # Sample multiple pages for better analysis
        sample_pages = min(15, doc.page_count)
        
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
                            
                            # Track position on page
                            bbox = span.get("bbox", [0, 0, 0, 0])
                            if len(bbox) >= 4:
                                page_positions.append(bbox[1])  # y-coordinate
        
        if font_sizes:
            analysis = {
                'chapter_threshold': np.percentile(font_sizes, 85),
                'large_font_threshold': np.percentile(font_sizes, 75),
                'average_font_size': np.mean(font_sizes),
                'font_variance': np.var(font_sizes),
                'common_fonts': Counter(font_families).most_common(5),
                'bold_usage': sum(1 for flag in font_flags if flag & 2**4) / len(font_flags),
                'page_top_threshold': np.percentile(page_positions, 20) if page_positions else 0
            }
        else:
            analysis = {
                'chapter_threshold': 16,
                'large_font_threshold': 14,
                'average_font_size': 12,
                'font_variance': 4,
                'common_fonts': [('default', 100)],
                'bold_usage': 0.1,
                'page_top_threshold': 50
            }
        
        return analysis

    def _is_chapter_title_enhanced(self, text, span, font_analysis, page_num):
        """Multi-criteria enhanced chapter title detection"""
        # Skip obvious non-chapter elements
        if self._should_skip_text(text):
            return False
        
        # Get span properties
        font_size = span.get("size", 12)
        font_flags = span.get("flags", 0)
        bbox = span.get("bbox", [0, 0, 0, 0])
        
        # Calculate detection scores
        scores = self._calculate_detection_scores(
            text, font_size, font_flags, bbox, font_analysis, page_num
        )
        
        # Weighted scoring system
        total_score = (
            scores['pattern'] * 0.30 +      # Pattern matching
            scores['formatting'] * 0.25 +   # Font/formatting
            scores['position'] * 0.15 +     # Page position
            scores['structure'] * 0.15 +    # Text structure
            scores['keywords'] * 0.10 +     # Educational keywords
            scores['context'] * 0.05        # Context clues
        )
        
        # Dynamic threshold based on document characteristics
        threshold = 0.6
        if font_analysis['font_variance'] > 10:  # Variable formatting document
            threshold = 0.5
        elif font_analysis['bold_usage'] < 0.05:  # Minimal bold usage
            threshold = 0.4
        
        return total_score >= threshold

    def _calculate_detection_scores(self, text, font_size, font_flags, bbox, font_analysis, page_num):
        """Calculate comprehensive detection scores"""
        text_lower = text.lower().strip()
        
        # 1. Pattern matching score
        pattern_score = 0.0
        for pattern in self.chapter_patterns:
            if re.match(pattern, text_lower):
                pattern_score = 1.0
                break
        
        # 2. Formatting score
        is_large_font = font_size >= font_analysis['chapter_threshold']
        is_medium_font = font_size >= font_analysis['large_font_threshold']
        is_bold = font_flags & 2**4
        is_italic = font_flags & 2**3
        
        formatting_score = 0.0
        if is_large_font:
            formatting_score += 0.6
        elif is_medium_font:
            formatting_score += 0.3
        
        if is_bold:
            formatting_score += 0.4
        
        formatting_score = min(formatting_score, 1.0)
        
        # 3. Position score (chapters often at top of page)
        position_score = 0.0
        if len(bbox) >= 4:
            y_position = bbox[1]
            if y_position <= font_analysis['page_top_threshold']:
                position_score = 0.8
            elif y_position <= font_analysis['page_top_threshold'] * 2:
                position_score = 0.4
        
        # 4. Structure score
        word_count = len(text.split())
        is_short = word_count <= 12
        is_very_short = word_count <= 6
        starts_with_capital = text[0].isupper() if text else False
        has_number = bool(re.search(r'\d+', text))
        is_all_caps = text.isupper() and len(text) > 5
        ends_properly = not text.endswith(('.', ',', ';'))
        
        structure_score = 0.0
        if is_short:
            structure_score += 0.3
        if is_very_short:
            structure_score += 0.2
        if starts_with_capital:
            structure_score += 0.2
        if has_number:
            structure_score += 0.2
        if is_all_caps:
            structure_score += 0.3
        if ends_properly:
            structure_score += 0.1
        
        structure_score = min(structure_score, 1.0)
        
        # 5. Keywords score
        keyword_score = 0.0
        keyword_matches = sum(1 for keyword in self.educational_keywords if keyword in text_lower)
        keyword_score = min(keyword_matches / 2, 1.0)
        
        # 6. Context score (based on page number and surrounding content)
        context_score = 0.0
        if page_num == 0:  # First page more likely to have chapter
            context_score += 0.5
        elif page_num < 5:  # Early pages
            context_score += 0.3
        
        return {
            'pattern': pattern_score,
            'formatting': formatting_score,
            'position': position_score,
            'structure': structure_score,
            'keywords': keyword_score,
            'context': context_score
        }

    def _should_skip_text(self, text):
        """Check if text should be skipped (not a chapter title)"""
        text_lower = text.lower().strip()
        
        # Skip based on patterns
        for pattern in self.skip_patterns:
            if re.match(pattern, text_lower):
                return True
        
        # Skip very long text (likely paragraph content)
        if len(text.split()) > 20:
            return True
        
        # Skip text that's mostly numbers or symbols
        if len(re.sub(r'[^a-zA-Z\s]', '', text)) < len(text) * 0.3:
            return True
        
        return False

    def _clean_chapter_name(self, raw_name):
        """Clean and standardize chapter names"""
        # Remove extra whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', raw_name.strip())
        
        # Remove common prefixes that might be formatting artifacts
        cleaned = re.sub(r'^[^\w\s]*', '', cleaned).strip()
        
        # Capitalize properly if all caps
        if cleaned.isupper() and len(cleaned) > 10:
            cleaned = cleaned.title()
        
        # Remove trailing page numbers or dots
        cleaned = re.sub(r'\s*\d+\s*$', '', cleaned).strip()
        cleaned = re.sub(r'\s*\.\s*$', '', cleaned).strip()
        
        # Ensure it's not empty
        if not cleaned:
            cleaned = "Untitled Chapter"
        
        return cleaned[:150]  # Reasonable length limit

    def _is_meaningful_content(self, text):
        """Enhanced check for meaningful educational content"""
        # Skip very short text
        if len(text.strip()) < 10:
            return False
        
        # Skip obvious non-content
        text_lower = text.lower().strip()
        
        # Skip headers, footers, page numbers
        skip_indicators = [
            r'^page\s+\d+',
            r'^\d+\s*$',
            r'^chapter\s+\d+\s*$',
            r'^©.*',
            r'^all rights reserved',
            r'^figure\s+\d+',
            r'^table\s+\d+',
            r'^\d+\s*\.\s*\d+\s*$',  # Page numbers like "1.1"
            r'^[a-z]\)\s*$',          # List markers like "a)"
            r'^\([a-z0-9]+\)\s*$'     # Parenthetical markers
        ]
        
        for pattern in skip_indicators:
            if re.match(pattern, text_lower):
                return False
        
        # Check for meaningful content indicators
        meaningful_indicators = [
            r'[.!?]',  # Contains sentence-ending punctuation
            r'\b(the|and|is|are|was|were|in|on|at|to|for|of|with)\b',  # Common words
            r'[a-zA-Z]{3,}',  # Contains words of reasonable length
        ]
        
        meaningful_count = sum(1 for pattern in meaningful_indicators if re.search(pattern, text))
        
        return meaningful_count >= 2

    def _create_chapter_data(self, chapter_name, content_list):
        """Create structured chapter data"""
        content = '\n'.join(content_list)
        word_count = len(content.split())
        
        return {
            'content': content,
            'type': self._categorize_chapter_type(chapter_name),
            'word_count': word_count,
            'estimated_reading_time': max(1, word_count // 200),  # ~200 words per minute
            'paragraph_count': len([p for p in content_list if len(p.strip()) > 50]),
            'content_quality': self._assess_content_quality(content)
        }

    def _categorize_chapter_type(self, chapter_name):
        """Enhanced chapter type categorization"""
        name_lower = chapter_name.lower().strip()
        
        for chapter_type, keywords in self.chapter_types.items():
            if any(keyword in name_lower for keyword in keywords):
                return chapter_type
        
        # Default classification based on patterns
        if re.match(r'^\d+', name_lower):
            return 'content'
        elif any(word in name_lower for word in ['conclusion', 'final', 'end']):
            return 'summary'
        else:
            return 'content'

    def _assess_content_quality(self, content):
        """Assess the quality/completeness of chapter content"""
        if not content:
            return 'empty'
        
        word_count = len(content.split())
        sentence_count = len([s for s in content.split('.') if s.strip()])
        paragraph_indicators = content.count('\n\n') + content.count('\n ')
        
        if word_count < 50:
            return 'minimal'
        elif word_count < 200:
            return 'brief'
        elif word_count < 1000:
            return 'moderate'
        else:
            return 'comprehensive'

    def _validate_and_enhance_chapters(self, chapters):
        """Validate and enhance detected chapters"""
        validated_chapters = {}
        
        for chapter_name, chapter_data in chapters.items():
            # Skip chapters that are too short (likely false positives)
            if chapter_data['word_count'] < 30:
                continue
            
            # Skip duplicate or very similar chapter names
            if self._is_duplicate_chapter(chapter_name, validated_chapters):
                # Merge with existing similar chapter
                similar_chapter = self._find_similar_chapter(chapter_name, validated_chapters)
                if similar_chapter:
                    self._merge_chapters(validated_chapters[similar_chapter], chapter_data)
                continue
            
            # Enhance chapter data
            chapter_data = self._enhance_chapter_data(chapter_data, chapter_name)
            
            validated_chapters[chapter_name] = chapter_data
        
        return validated_chapters

    def _is_duplicate_chapter(self, chapter_name, existing_chapters):
        """Check if chapter name is too similar to existing ones"""
        for existing_name in existing_chapters.keys():
            similarity = self._calculate_name_similarity(chapter_name, existing_name)
            if similarity > 0.8:  # 80% similarity threshold
                return True
        return False

    def _find_similar_chapter(self, chapter_name, existing_chapters):
        """Find the most similar existing chapter"""
        best_match = None
        best_similarity = 0
        
        for existing_name in existing_chapters.keys():
            similarity = self._calculate_name_similarity(chapter_name, existing_name)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = existing_name
        
        return best_match if best_similarity > 0.8 else None

    def _calculate_name_similarity(self, name1, name2):
        """Calculate similarity between two chapter names"""
        # Simple similarity based on common words
        words1 = set(name1.lower().split())
        words2 = set(name2.lower().split())
        
        if not words1 or not words2:
            return 0
        
        common_words = words1.intersection(words2)
        total_words = words1.union(words2)
        
        return len(common_words) / len(total_words)

    def _merge_chapters(self, chapter1_data, chapter2_data):
        """Merge two similar chapters"""
        # Combine content
        chapter1_data['content'] += '\n\n' + chapter2_data['content']
        
        # Update metrics
        chapter1_data['word_count'] += chapter2_data['word_count']
        chapter1_data['estimated_reading_time'] = max(1, chapter1_data['word_count'] // 200)
        chapter1_data['paragraph_count'] += chapter2_data['paragraph_count']

    def _enhance_chapter_data(self, chapter_data, chapter_name):
        """Add enhanced metadata to chapter data"""
        content = chapter_data['content']
        
        # Add readability metrics
        chapter_data['readability'] = self._calculate_readability(content)
        
        # Add content characteristics
        chapter_data['characteristics'] = self._analyze_content_characteristics(content)
        
        # Add educational level estimate
        chapter_data['educational_level'] = self._estimate_educational_level(content, chapter_name)
        
        return chapter_data

    def _calculate_readability(self, content):
        """Calculate readability metrics"""
        if not content:
            return 'unknown'
        
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        words = content.split()
        
        if not sentences or not words:
            return 'unknown'
        
        avg_sentence_length = len(words) / len(sentences)
        
        # Simple readability assessment
        if avg_sentence_length < 10:
            return 'easy'
        elif avg_sentence_length < 20:
            return 'moderate'
        else:
            return 'difficult'

    def _analyze_content_characteristics(self, content):
        """Analyze characteristics of the content"""
        characteristics = []
        content_lower = content.lower()
        
        # Check for various content types
        if re.search(r'\b(example|for instance|such as)\b', content_lower):
            characteristics.append('examples')
        
        if re.search(r'\b(exercise|problem|question)\b', content_lower):
            characteristics.append('exercises')
        
        if re.search(r'\b(definition|define|means|refers to)\b', content_lower):
            characteristics.append('definitions')
        
        if re.search(r'\b(figure|diagram|chart|graph)\b', content_lower):
            characteristics.append('visual_references')
        
        if re.search(r'\b(formula|equation|calculate)\b', content_lower):
            characteristics.append('mathematical')
        
        return characteristics

    def _estimate_educational_level(self, content, chapter_name):
        """Estimate the educational level of the content"""
        # Simple heuristics for educational level
        name_lower = chapter_name.lower()
        content_lower = content.lower()
        
        # Check for elementary indicators
        elementary_indicators = ['basic', 'introduction', 'simple', 'easy', 'fundamental']
        if any(indicator in name_lower for indicator in elementary_indicators):
            return 'elementary'
        
        # Check for advanced indicators
        advanced_indicators = ['advanced', 'complex', 'analysis', 'theory', 'research']
        if any(indicator in name_lower or indicator in content_lower for indicator in advanced_indicators):
            return 'advanced'
        
        # Check for high school indicators
        high_school_indicators = ['algebra', 'geometry', 'biology', 'chemistry', 'physics']
        if any(indicator in name_lower or indicator in content_lower for indicator in high_school_indicators):
            return 'high_school'
        
        return 'middle_school'  # Default

    def _fallback_chapter_detection(self, doc):
        """Fallback chapter detection using basic methods"""
        st.warning("Using fallback chapter detection method")
        
        chapters = {}
        current_chapter = None
        current_content = []
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()
            
            # Simple line-by-line processing
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Basic chapter detection
                if self._is_basic_chapter_title(line):
                    if current_chapter and current_content:
                        chapters[current_chapter] = {
                            'content': '\n'.join(current_content),
                            'type': 'content',
                            'word_count': len('\n'.join(current_content).split()),
                            'estimated_reading_time': max(1, len('\n'.join(current_content).split()) // 200)
                        }
                    
                    current_chapter = self._clean_chapter_name(line)
                    current_content = []
                else:
                    if len(line) > 10:  # Meaningful content
                        current_content.append(line)
        
        # Add final chapter
        if current_chapter and current_content:
            chapters[current_chapter] = {
                'content': '\n'.join(current_content),
                'type': 'content',
                'word_count': len('\n'.join(current_content).split()),
                'estimated_reading_time': max(1, len('\n'.join(current_content).split()) // 200)
            }
        
        return chapters

    def _is_basic_chapter_title(self, line):
        """Basic chapter title detection for fallback"""
        line_lower = line.lower().strip()
        
        # Simple patterns
        basic_patterns = [
            r'^chapter\s+\d+',
            r'^lesson\s+\d+',
            r'^unit\s+\d+',
            r'^\d+\.\s+[A-Z]'
        ]
        
        for pattern in basic_patterns:
            if re.match(pattern, line_lower):
                return True
        
        # Check if line is short and starts with capital
        return (len(line.split()) <= 8 and 
                line[0].isupper() if line else False and
                not line.endswith('.'))

    def get_detection_statistics(self, chapters):
        """Get statistics about the chapter detection process"""
        if not chapters:
            return {}
        
        stats = {
            'total_chapters': len(chapters),
            'chapter_types': {},
            'total_words': 0,
            'average_chapter_length': 0,
            'content_quality_distribution': {},
            'educational_levels': {}
        }
        
        for chapter_data in chapters.values():
            # Count chapter types
            chapter_type = chapter_data.get('type', 'unknown')
            stats['chapter_types'][chapter_type] = stats['chapter_types'].get(chapter_type, 0) + 1
            
            # Sum total words
            stats['total_words'] += chapter_data.get('word_count', 0)
            
            # Count content quality
            quality = chapter_data.get('content_quality', 'unknown')
            stats['content_quality_distribution'][quality] = stats['content_quality_distribution'].get(quality, 0) + 1
            
            # Count educational levels
            level = chapter_data.get('educational_level', 'unknown')
            stats['educational_levels'][level] = stats['educational_levels'].get(level, 0) + 1
        
        stats['average_chapter_length'] = stats['total_words'] // len(chapters) if chapters else 0
        
        return stats
