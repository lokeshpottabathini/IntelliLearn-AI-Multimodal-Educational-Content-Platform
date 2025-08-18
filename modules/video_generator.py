import re
import os
import time
import requests
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import streamlit as st
from gtts import gTTS
from config import Config
import torch 

# Enhanced imports for open-source models
try:
    from .opensource_video_generator import OpenSourceVideoGenerator
    ENHANCED_MODULES_AVAILABLE = True
except ImportError:
    ENHANCED_MODULES_AVAILABLE = False

class ProductionVideoGenerator:
    def __init__(self):
        """Initialize enhanced video generator with difficulty level support"""
        self.config = Config()
        self.output_dir = self.config.VIDEO_OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs("temp", exist_ok=True)
        
        # Initialize enhanced video generator if available
        self.enhanced_generator = None
        if ENHANCED_MODULES_AVAILABLE:
            try:
                self.enhanced_generator = OpenSourceVideoGenerator()
                st.success("✅ Enhanced video generation available")
            except Exception as e:
                st.warning(f"Enhanced video generation not available: {e}")
    
    def create_professional_video_with_difficulty(self, topic_data, chapter_name="", persona="Professional Instructor", 
                                                length=5, include_examples=True, difficulty_level='intermediate'):
        """NEW: Create professional educational video with difficulty adaptation"""
        try:
            st.info(f"🎬 Creating {difficulty_level} level video for: {topic_data.get('summary', 'Educational Topic')}")
            
            # Use enhanced generator if available
            if self.enhanced_generator:
                return self.enhanced_generator.create_professional_video_with_difficulty(
                    topic_data, chapter_name, persona, length, include_examples, difficulty_level
                )
            
            # Fallback to enhanced standard generation
            return self.create_professional_video_enhanced(
                topic_data, chapter_name, persona, length, include_examples, difficulty_level
            )
            
        except Exception as e:
            st.error(f"Enhanced video generation error: {str(e)}")
            # Final fallback to original method
            return self.create_professional_video(topic_data, chapter_name, persona, length, include_examples)

    def create_professional_video_enhanced(self, topic_data, chapter_name="", persona="Professional Instructor", 
                                         length=5, include_examples=True, difficulty_level='intermediate'):
        """Enhanced professional video creation with difficulty adaptation"""
        try:
            st.info(f"🎬 Creating enhanced {difficulty_level} level video...")
            
            # Generate difficulty-adapted script
            script_data = self._generate_professional_script_enhanced(
                topic_data, persona, length, include_examples, difficulty_level
            )
            
            # Create difficulty-adapted slides
            slides_info = self._create_professional_slides_enhanced(
                topic_data, script_data, chapter_name, difficulty_level
            )
            
            # Generate difficulty-adapted voiceover
            audio_info = self._create_professional_audio_enhanced(
                script_data['clean_script'], difficulty_level
            )
            
            # Return enhanced video information
            return {
                'success': True,
                'script': script_data['full_script'],
                'slides': slides_info,
                'audio': audio_info,
                'video_concept': self._generate_video_concept_enhanced(topic_data, difficulty_level),
                'duration': length,
                'difficulty_level': difficulty_level,
                'enhanced_features': {
                    'difficulty_adapted': True,
                    'persona_customized': True,
                    'examples_included': include_examples
                }
            }
            
        except Exception as e:
            st.error(f"Enhanced video generation error: {str(e)}")
            return {'success': False, 'error': str(e)}

    def create_professional_video(self, topic_data, chapter_name="", persona="Friendly Tutor", 
                                length=5, include_examples=True):
        """Create professional educational video with slides and voiceover (original method)"""
        try:
            st.info(f"🎬 Creating video for: {topic_data.get('summary', 'Educational Topic')}")
            
            # Generate professional script with timing
            script_data = self._generate_professional_script(topic_data, persona, length, include_examples)
            
            # Create high-quality slides
            slides_info = self._create_professional_slides(topic_data, script_data, chapter_name)
            
            # Generate clear voiceover
            audio_info = self._create_professional_audio(script_data['clean_script'])
            
            # Return comprehensive video information
            return {
                'success': True,
                'script': script_data['full_script'],
                'slides': slides_info,
                'audio': audio_info,
                'video_concept': self._generate_video_concept(topic_data),
                'duration': length
            }
            
        except Exception as e:
            st.error(f"Video generation error: {str(e)}")
            return {'success': False, 'error': str(e)}

    def _generate_professional_script_enhanced(self, topic_data, persona, length, include_examples, difficulty_level):
        """NEW: Generate enhanced script with difficulty adaptation"""
        
        # Difficulty-adapted persona descriptions
        persona_adaptations = {
            'beginner': {
                'Professional Instructor': 'patient, encouraging teacher who explains concepts very clearly',
                'Friendly Tutor': 'supportive, friendly guide who makes learning fun and easy',
                'Expert Academic': 'knowledgeable but accessible expert who simplifies complex ideas'
            },
            'intermediate': {
                'Professional Instructor': 'clear, structured educator with balanced explanations',
                'Friendly Tutor': 'engaging teacher who connects concepts to real applications',
                'Expert Academic': 'scholarly instructor with practical insights'
            },
            'advanced': {
                'Professional Instructor': 'sophisticated educator who challenges thinking',
                'Friendly Tutor': 'intellectually stimulating guide for deep learning',
                'Expert Academic': 'research-oriented scholar with cutting-edge perspectives'
            }
        }

        adapted_persona = persona_adaptations.get(difficulty_level, {}).get(persona, persona)

        # Difficulty-specific system prompts
        difficulty_prompts = {
            'beginner': f"""You are a {adapted_persona} creating educational content for BEGINNER learners.

SCRIPT REQUIREMENTS FOR BEGINNERS:
- Use simple, clear language and short sentences
- Define all technical terms immediately
- Include lots of encouragement ("Great job!", "You're doing well!")
- Repeat key concepts multiple times in different ways
- Use everyday analogies and simple examples
- Speak slowly with clear pronunciation cues
- Include more time for comprehension

STRUCTURE FOR {length} MINUTES:
1. Warm welcome and encouragement (20 seconds)
2. Simple introduction with preview (45 seconds)
3. Main content broken into small, digestible parts
4. Frequent summaries and check-ins
5. Encouraging conclusion with next steps (30 seconds)

PERSONA: {adapted_persona}
DIFFICULTY: BEGINNER - Keep it simple and encouraging!""",

            'intermediate': f"""You are a {adapted_persona} creating educational content for INTERMEDIATE learners.

SCRIPT REQUIREMENTS FOR INTERMEDIATE:
- Use clear, moderately technical language
- Balance detail with accessibility
- Include practical applications and examples
- Build concepts progressively
- Engage with thoughtful questions
- Standard pacing and complexity

STRUCTURE FOR {length} MINUTES:
1. Professional introduction (15 seconds)
2. Clear overview of objectives (30 seconds)
3. Main content with examples and applications
4. Connections between concepts
5. Summary and practical takeaways (30 seconds)

PERSONA: {adapted_persona}
DIFFICULTY: INTERMEDIATE - Balance accessibility with depth""",

            'advanced': f"""You are a {adapted_persona} creating educational content for ADVANCED learners.

SCRIPT REQUIREMENTS FOR ADVANCED:
- Use precise technical terminology
- Focus on complex relationships and implications
- Include analytical perspectives and critical thinking
- Challenge assumptions and present multiple viewpoints
- Fast-paced with dense information
- Assume strong foundational knowledge

STRUCTURE FOR {length} MINUTES:
1. Direct, scholarly introduction (10 seconds)
2. Complex framework presentation (30 seconds)
3. In-depth analysis with technical details
4. Critical evaluation and synthesis
5. Advanced applications and future directions (20 seconds)

PERSONA: {adapted_persona}
DIFFICULTY: ADVANCED - Provide sophisticated, challenging content"""
        }

        system_prompt = difficulty_prompts.get(difficulty_level, difficulty_prompts['intermediate'])

        # Enhanced content prompt with difficulty context
        content_prompt = f"""Create an engaging {length}-minute educational video script for {difficulty_level} level learners about:

TOPIC: {topic_data.get('summary', 'Topic')}

KEY CONCEPTS TO COVER (adapted for {difficulty_level} level):
{chr(10).join(f"• {point}" for point in topic_data.get('key_points', [])[:5])}

CONTENT DETAILS:
{topic_data.get('content', '')[:1200]}

DIFFICULTY LEVEL: {difficulty_level.upper()}
- Adapt language complexity accordingly
- Adjust pacing for target audience
- Include appropriate examples for skill level

Include timing markers [0:30], slide markers [SLIDE X: Title], and visual cues [VISUAL: description].
Make it educational, accurate, and perfectly matched to {difficulty_level} learners while staying true to the textbook content."""

        try:
            full_script = self._call_groq_api_with_difficulty(content_prompt, system_prompt, difficulty_level)
            
            # Clean script for TTS with difficulty-specific adjustments
            clean_script = self._clean_script_for_audio_enhanced(full_script, difficulty_level)
            
            return {
                'full_script': full_script,
                'clean_script': clean_script,
                'slide_markers': self._extract_slide_markers(full_script),
                'difficulty_level': difficulty_level,
                'adapted_for_level': True
            }
            
        except Exception as e:
            # Fallback script generation with difficulty awareness
            return self._generate_fallback_script_enhanced(topic_data, persona, length, difficulty_level)

    def _create_professional_slides_enhanced(self, topic_data, script_data, chapter_name, difficulty_level):
        """NEW: Create enhanced slides with difficulty-specific design"""
        slides_created = []
        
        try:
            # Extract topics and content from script with difficulty consideration
            slide_contents = self._parse_slide_content_enhanced(script_data, topic_data, difficulty_level)
            
            for i, slide_content in enumerate(slide_contents):
                slide_path = f"{self.output_dir}/slide_{difficulty_level}_{i+1}_{int(time.time())}.png"
                
                self._create_slide_image_enhanced(
                    slide_content, 
                    slide_path, 
                    i+1, 
                    len(slide_contents),
                    chapter_name,
                    difficulty_level
                )
                
                slides_created.append({
                    'path': slide_path,
                    'title': slide_content.get('title', f'Slide {i+1}'),
                    'content': slide_content.get('points', []),
                    'difficulty_level': difficulty_level,
                    'design_adapted': True
                })
            
            return slides_created
            
        except Exception as e:
            st.warning(f"Enhanced slide creation warning: {e}")
            return self._create_basic_slides_enhanced(topic_data, difficulty_level)

    def _create_slide_image_enhanced(self, slide_content, output_path, slide_num, total_slides, chapter_name, difficulty_level):
        """NEW: Create enhanced slide with difficulty-specific design"""
        
        # Canvas setup
        width, height = 1920, 1080
        
        # Difficulty-specific color schemes
        color_schemes = {
            'beginner': {
                'primary': '#4CAF50',      # Friendly green
                'secondary': '#81C784',    # Light green
                'accent': '#FF9800',       # Orange
                'background': '#F1F8E9',   # Very light green
                'text': '#2E7D32',         # Dark green
                'white': '#ffffff'
            },
            'intermediate': {
                'primary': '#2196F3',      # Professional blue
                'secondary': '#64B5F6',    # Light blue
                'accent': '#FF5722',       # Orange-red
                'background': '#E3F2FD',   # Light blue
                'text': '#1565C0',         # Dark blue
                'white': '#ffffff'
            },
            'advanced': {
                'primary': '#9C27B0',      # Sophisticated purple
                'secondary': '#BA68C8',    # Light purple
                'accent': '#E91E63',       # Pink
                'background': '#F3E5F5',   # Very light purple
                'text': '#6A1B9A',         # Dark purple
                'white': '#ffffff'
            }
        }
        
        colors = color_schemes.get(difficulty_level, color_schemes['intermediate'])
        
        # Create image
        img = Image.new('RGB', (width, height), color=colors['background'])
        draw = ImageDraw.Draw(img)
        
        # Difficulty-specific font sizes
        font_sizes = {
            'beginner': {'title': 80, 'subtitle': 52, 'body': 40, 'small': 32},
            'intermediate': {'title': 72, 'subtitle': 48, 'body': 36, 'small': 28},
            'advanced': {'title': 68, 'subtitle': 44, 'body': 34, 'small': 26}
        }
        
        sizes = font_sizes.get(difficulty_level, font_sizes['intermediate'])
        
        # Load fonts with fallbacks
        try:
            title_font = ImageFont.truetype("arial.ttf", sizes['title'])
            subtitle_font = ImageFont.truetype("arial.ttf", sizes['subtitle'])
            body_font = ImageFont.truetype("arial.ttf", sizes['body'])
            small_font = ImageFont.truetype("arial.ttf", sizes['small'])
        except:
            # Fallback to default font
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
            body_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Header section with difficulty indicator
        draw.rectangle([(0, 0), (width, 150)], fill=colors['primary'])
        
        # Difficulty level indicator
        difficulty_badge_color = colors['accent']
        draw.rectangle([(width-300, 20), (width-50, 80)], fill=difficulty_badge_color)
        draw.text((width-275, 30), f"{difficulty_level.upper()}", fill=colors['white'], font=subtitle_font)
        
        # Chapter name
        if chapter_name:
            draw.text((60, 30), chapter_name, fill=colors['white'], font=subtitle_font)
        
        # Slide title with difficulty-appropriate positioning
        title = slide_content.get('title', f'Slide {slide_num}')
        title_y = 180 if difficulty_level == 'beginner' else 200
        draw.text((60, title_y), title, fill=colors['primary'], font=title_font)
        
        # Content area with difficulty-specific spacing
        y_position = 300 if difficulty_level == 'beginner' else 320
        bullet_points = slide_content.get('points', [])
        
        # Adjust content based on difficulty
        max_points = {'beginner': 4, 'intermediate': 5, 'advanced': 6}
        points_to_show = bullet_points[:max_points.get(difficulty_level, 5)]
        
        for i, point in enumerate(points_to_show):
            # Bullet point with difficulty-specific design
            bullet_size = 25 if difficulty_level == 'beginner' else 20
            draw.ellipse([(80, y_position + 10), (80 + bullet_size, y_position + 30)], fill=colors['secondary'])
            
            # Text wrapping based on difficulty
            wrap_length = {'beginner': 60, 'intermediate': 70, 'advanced': 80}
            wrapped_text = self._wrap_text(point, wrap_length.get(difficulty_level, 70))
            
            for line in wrapped_text:
                draw.text((130, y_position), line, fill=colors['text'], font=body_font)
                y_position += 45 if difficulty_level == 'beginner' else 42
            
            y_position += 25 if difficulty_level == 'beginner' else 20  # Space between points
        
        # Footer with enhanced information
        draw.rectangle([(0, height-80), (width, height)], fill=colors['primary'])
        draw.text((60, height-60), f"Slide {slide_num} of {total_slides} • {difficulty_level.title()} Level", 
                 fill=colors['white'], font=small_font)
        
        # Enhanced branding
        draw.text((width-400, height-60), "IntelliLearn AI Enhanced", 
                 fill=colors['white'], font=small_font)
        
        # Save image
        img.save(output_path)
        st.success(f"✅ Created {difficulty_level} level slide {slide_num}: {title}")

    def _create_professional_audio_enhanced(self, clean_script, difficulty_level):
        """NEW: Create difficulty-adapted professional voiceover"""
        try:
            # Prepare script for TTS with difficulty adjustments
            audio_script = self._prepare_script_for_tts_enhanced(clean_script, difficulty_level)
            
            # Difficulty-specific TTS settings
            tts_settings = {
                'beginner': {'slow': True, 'lang': 'en'},
                'intermediate': {'slow': False, 'lang': 'en'},
                'advanced': {'slow': False, 'lang': 'en'}
            }
            
            settings = tts_settings.get(difficulty_level, tts_settings['intermediate'])
            
            # Generate TTS
            tts = gTTS(text=audio_script, **settings)
            audio_path = f"{self.output_dir}/voiceover_{difficulty_level}_{int(time.time())}.mp3"
            tts.save(audio_path)
            
            # Calculate duration based on difficulty (beginners need more time)
            word_count = len(audio_script.split())
            words_per_second = {'beginner': 2.0, 'intermediate': 2.5, 'advanced': 3.0}
            duration = word_count / words_per_second.get(difficulty_level, 2.5)
            
            return {
                'path': audio_path,
                'duration': duration,
                'script': audio_script,
                'difficulty_level': difficulty_level,
                'speed_adapted': True
            }
            
        except Exception as e:
            st.warning(f"Enhanced audio generation warning: {e}")
            return {'path': None, 'error': str(e)}

    def _parse_slide_content_enhanced(self, script_data, topic_data, difficulty_level):
        """NEW: Parse slide content with difficulty adaptation"""
        full_script = script_data.get('full_script', '')
        
        # Look for slide markers
        slide_pattern = r'\[SLIDE \d+:?\s*([^\]]+)\]'
        slide_matches = re.findall(slide_pattern, full_script)
        
        if slide_matches:
            slides = []
            for i, title in enumerate(slide_matches):
                slides.append({
                    'title': title.strip(),
                    'points': self._extract_slide_points_enhanced(full_script, title, i, difficulty_level)
                })
            return slides
        else:
            # Generate slides from topic data with difficulty awareness
            return self._generate_slides_from_topic_enhanced(topic_data, difficulty_level)

    def _extract_slide_points_enhanced(self, script, title, slide_index, difficulty_level):
        """NEW: Extract slide points with difficulty adaptation"""
        # Extract content between slide markers
        lines = script.split('\n')
        slide_content = []
        capturing = False
        
        for line in lines:
            if f'[SLIDE {slide_index + 1}' in line or title in line:
                capturing = True
                continue
            elif '[SLIDE' in line and capturing:
                break
            elif capturing and line.strip():
                # Clean and add meaningful content
                clean_line = re.sub(r'\[.*?\]', '', line).strip()
                if clean_line and len(clean_line) > 10:
                    # Adapt content length based on difficulty
                    if difficulty_level == 'beginner' and len(clean_line) > 80:
                        clean_line = clean_line[:80] + "..."
                    elif difficulty_level == 'advanced' and len(clean_line) < 30:
                        continue  # Skip very short content for advanced
                    
                    slide_content.append(clean_line)
        
        # If no specific content found, return difficulty-adapted generic points
        if not slide_content:
            return self._get_default_slide_points(title, difficulty_level)
        
        # Limit points based on difficulty
        max_points = {'beginner': 3, 'intermediate': 4, 'advanced': 5}
        return slide_content[:max_points.get(difficulty_level, 4)]

    def _get_default_slide_points(self, title, difficulty_level):
        """Get default slide points based on difficulty level"""
        defaults = {
            'beginner': [
                f"Simple introduction to {title}",
                "Easy-to-understand explanation",
                "Basic example to help you learn"
            ],
            'intermediate': [
                f"Understanding {title}",
                "Key concepts and applications", 
                "Practical examples",
                "Important connections"
            ],
            'advanced': [
                f"Comprehensive analysis of {title}",
                "Complex theoretical framework",
                "Advanced applications and implications",
                "Critical evaluation perspectives",
                "Research and future directions"
            ]
        }
        
        return defaults.get(difficulty_level, defaults['intermediate'])

    def _generate_slides_from_topic_enhanced(self, topic_data, difficulty_level):
        """NEW: Generate slides with difficulty-specific adaptations"""
        slides = []
        
        # Difficulty-adapted introduction slide
        intro_content = {
            'beginner': {
                'title': 'Welcome to Your Learning Journey!',
                'points': [
                    f"Today we'll explore: {topic_data.get('summary', 'an exciting topic')}",
                    f"Perfect for beginners - we'll go step by step",
                    "Don't worry, we'll make it easy and fun!",
                    "Ready to start learning together?"
                ]
            },
            'intermediate': {
                'title': 'Learning Objectives',
                'points': [
                    f"Topic: {topic_data.get('summary', 'Educational Content')}",
                    f"Difficulty Level: {difficulty_level.title()}",
                    "We'll explore key concepts and applications",
                    "Let's build on your existing knowledge"
                ]
            },
            'advanced': {
                'title': 'Advanced Analysis Framework',
                'points': [
                    f"Complex examination of: {topic_data.get('summary', 'Advanced Topic')}",
                    "Sophisticated theoretical perspectives",
                    "Critical analysis and synthesis",
                    "Challenging intellectual engagement ahead"
                ]
            }
        }
        
        slides.append(intro_content.get(difficulty_level, intro_content['intermediate']))
        
        # Key points slides with difficulty adaptation
        key_points = topic_data.get('key_points', [])
        if key_points:
            if difficulty_level == 'beginner':
                # Split into smaller chunks for beginners
                for i in range(0, len(key_points), 2):
                    chunk = key_points[i:i+2]
                    slides.append({
                        'title': f'Key Learning Point {i//2 + 1}',
                        'points': [f"🎯 {point}" for point in chunk] + ["Let's take our time to understand this!"]
                    })
            else:
                # Standard grouping for intermediate/advanced
                max_points = {'intermediate': 4, 'advanced': 5}
                slides.append({
                    'title': 'Core Concepts' if difficulty_level == 'intermediate' else 'Advanced Theoretical Framework',
                    'points': key_points[:max_points.get(difficulty_level, 4)]
                })
        
        # Examples slide with difficulty adaptation
        examples = topic_data.get('examples', [])
        if examples and len(examples) > 0:
            example_content = {
                'beginner': {
                    'title': 'Simple Examples to Help You Learn',
                    'points': [f"📚 {ex[:60]}..." if len(ex) > 60 else f"📚 {ex}" for ex in examples[:2]]
                },
                'intermediate': {
                    'title': 'Practical Applications',
                    'points': [f"• {ex[:80]}..." if len(ex) > 80 else f"• {ex}" for ex in examples[:3]]
                },
                'advanced': {
                    'title': 'Complex Case Studies',
                    'points': [f"→ {ex[:100]}..." if len(ex) > 100 else f"→ {ex}" for ex in examples[:4]]
                }
            }
            
            slides.append(example_content.get(difficulty_level, example_content['intermediate']))
        
        # Summary slide with difficulty adaptation
        summary_content = {
            'beginner': {
                'title': 'Great Job! Let\'s Review',
                'points': [
                    "🌟 You learned some amazing new concepts!",
                    "🎯 Remember the key points we covered",
                    "💪 You're ready for the next step",
                    "🚀 Keep up the excellent work!"
                ]
            },
            'intermediate': {
                'title': 'Key Takeaways',
                'points': [
                    "✅ Covered essential concepts and applications",
                    "🔗 Connected theory to practice",
                    "📈 Ready to advance your understanding",
                    "🎯 Apply these concepts in your studies"
                ]
            },
            'advanced': {
                'title': 'Synthesis and Future Directions',
                'points': [
                    "🔬 Analyzed complex theoretical frameworks",
                    "⚡ Explored cutting-edge perspectives",
                    "🎯 Ready for independent research",
                    "🌐 Consider broader implications"
                ]
            }
        }
        
        slides.append(summary_content.get(difficulty_level, summary_content['intermediate']))
        
        return slides

    def _prepare_script_for_tts_enhanced(self, script, difficulty_level):
        """NEW: Enhanced script preparation for TTS with difficulty adaptation"""
        
        # Remove timing markers and visual cues
        clean_script = re.sub(r'\[\d+:\d+\]', '', script)
        clean_script = re.sub(r'\[SLIDE.*?\]', '', clean_script)
        clean_script = re.sub(r'\[VISUAL:.*?\]', '', clean_script)
        clean_script = re.sub(r'\[.*?\]', '', clean_script)
        
        # Clean up extra spaces
        clean_script = re.sub(r'\s+', ' ', clean_script).strip()
        
        # Difficulty-specific text adaptations
        if difficulty_level == 'beginner':
            # Add pauses for better comprehension
            clean_script = clean_script.replace('.', '... ')
            clean_script = clean_script.replace(',', ', ')
            # Add encouraging phrases
            clean_script = clean_script.replace('understand', 'clearly understand')
            clean_script = clean_script.replace('learn', 'easily learn')
        
        elif difficulty_level == 'advanced':
            # More efficient, faster-paced speech
            clean_script = clean_script.replace(' that ', ' ')
            clean_script = clean_script.replace(' which ', ' ')
            # Remove excessive explanatory words
            clean_script = re.sub(r'\b(basically|simply|just)\b', '', clean_script)
        
        return clean_script

    def _clean_script_for_audio_enhanced(self, script, difficulty_level):
        """NEW: Enhanced script cleaning with difficulty awareness"""
        
        # Remove all markers and visual cues
        patterns_to_remove = [
            r'\[\d+:\d+\]',          # Time markers
            r'\[SLIDE.*?\]',         # Slide markers
            r'\[VISUAL:.*?\]',       # Visual cues
            r'\[ANIMATE:.*?\]',      # Animation cues
            r'\[TEXT:.*?\]',         # Text overlays
        ]
        
        clean = script
        for pattern in patterns_to_remove:
            clean = re.sub(pattern, '', clean)
        
        # Clean up spacing
        clean = re.sub(r'\s+', ' ', clean).strip()
        
        # Difficulty-specific enhancements
        if difficulty_level == 'beginner':
            # Add natural pauses and clearer pronunciation
            clean = clean.replace('. ', '. ... ')  # Longer pauses between sentences
            clean = clean.replace('!', '! ')       # Enthusiastic pauses
        
        return clean

    def _generate_video_concept_enhanced(self, topic_data, difficulty_level):
        """NEW: Generate enhanced video concept with difficulty awareness"""
        
        concept_templates = {
            'beginner': {
                'title': f"Beginner's Guide: {topic_data.get('summary', 'Educational Video')}",
                'description': f"A friendly, step-by-step introduction covering {len(topic_data.get('key_points', []))} key concepts for new learners",
                'target_audience': f"Beginner level learners new to the subject",
                'approach': "Encouraging, simple explanations with lots of examples"
            },
            'intermediate': {
                'title': f"Understanding {topic_data.get('summary', 'Educational Video')}",
                'description': f"A comprehensive exploration of {len(topic_data.get('key_points', []))} important concepts with practical applications",
                'target_audience': f"Intermediate learners building on existing knowledge",
                'approach': "Balanced depth with clear explanations and real-world connections"
            },
            'advanced': {
                'title': f"Advanced Analysis: {topic_data.get('summary', 'Educational Video')}",
                'description': f"Sophisticated examination of {len(topic_data.get('key_points', []))} complex concepts with critical analysis",
                'target_audience': f"Advanced learners ready for challenging content",
                'approach': "Technical depth with analytical perspectives and research insights"
            }
        }
        
        template = concept_templates.get(difficulty_level, concept_templates['intermediate'])
        
        return {
            'title': template['title'],
            'description': template['description'],
            'target_audience': template['target_audience'],
            'learning_objectives': topic_data.get('key_points', [])[:3],
            'duration': "5 minutes",
            'format': "Enhanced slide-based presentation with professional voiceover",
            'difficulty_level': difficulty_level,
            'teaching_approach': template['approach'],
            'enhanced_features': [
                f"{difficulty_level.title()}-adapted content",
                "Professional slide design",
                "Optimized voice pacing",
                "Educational best practices"
            ]
        }

    def _call_groq_api_with_difficulty(self, content, system_prompt, difficulty_level):
        """NEW: Enhanced API call with difficulty context"""
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.GROQ_API_KEY}"
        }
        
        # Adjust token limits based on difficulty
        token_limits = {'beginner': 1200, 'intermediate': 1500, 'advanced': 1800}
        max_tokens = token_limits.get(difficulty_level, 1500)
        
        # Adjust temperature based on difficulty (more creativity for advanced)
        temperatures = {'beginner': 0.5, 'intermediate': 0.7, 'advanced': 0.8}
        temperature = temperatures.get(difficulty_level, 0.7)
        
        payload = {
            "model": self.config.GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = requests.post(
            f"{self.config.GROQ_BASE_URL}/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            raise Exception(f"API call failed: {response.status_code}")

    def _generate_fallback_script_enhanced(self, topic_data, persona, length, difficulty_level):
        """NEW: Enhanced fallback script with difficulty adaptation"""
        
        # Difficulty-specific script templates
        script_templates = {
            'beginner': f"""[0:00] Warm Welcome
Welcome, wonderful learners! I'm so excited to explore {topic_data.get('summary', 'our topic')} with you today.
Don't worry if this is new to you - we'll take it step by step together!

[0:30] Gentle Introduction  
Let's start with the basics and build your understanding slowly.

[1:00] Key Concepts - Made Simple
""",
            'intermediate': f"""[0:00] Introduction
Welcome to today's lesson on {topic_data.get('summary', 'our topic')}.
We'll build on what you already know and explore new applications.

[0:30] Main Content
Let's examine the key concepts and their practical implications:

[1:00] Core Learning Points
""",
            'advanced': f"""[0:00] Opening
Today we'll conduct a sophisticated analysis of {topic_data.get('summary', 'our topic')}.
Prepare for challenging concepts and complex relationships.

[0:20] Theoretical Framework
Let's examine the advanced principles:

[0:40] Critical Analysis
"""
        }
        
        script = script_templates.get(difficulty_level, script_templates['intermediate'])
        
        # Add key points with difficulty-appropriate explanations
        for i, point in enumerate(topic_data.get('key_points', [])[:4]):
            time_marker = f"[{i+1}:{30 if difficulty_level == 'beginner' else 15}]"
            
            if difficulty_level == 'beginner':
                script += f"\n{time_marker} Let's understand: {point}\nTake your time with this - it's important!\n"
            elif difficulty_level == 'advanced':
                script += f"\n{time_marker} Advanced concept: {point}\n"
            else:
                script += f"\n{time_marker} {point}\n"
        
        # Difficulty-appropriate conclusions
        conclusions = {
            'beginner': f"\n[{length-1}:00] You Did Great!\nCongratulations! You've learned so much today. I'm proud of your progress!",
            'intermediate': f"\n[{length-1}:00] Summary\nWe've covered the essential concepts. You're ready to apply this knowledge.",
            'advanced': f"\n[{length-1}:00] Synthesis\nWe've analyzed complex frameworks. Continue your research and critical thinking."
        }
        
        script += conclusions.get(difficulty_level, conclusions['intermediate'])
        
        clean_script = self._clean_script_for_audio_enhanced(script, difficulty_level)
        
        return {
            'full_script': script,
            'clean_script': clean_script,
            'slide_markers': [],
            'difficulty_level': difficulty_level,
            'fallback_generated': True
        }

    def _create_basic_slides_enhanced(self, topic_data, difficulty_level):
        """NEW: Enhanced basic slides with difficulty adaptation"""
        
        basic_slide_content = {
            'beginner': {
                'path': None,
                'title': 'Welcome to Learning!',
                'content': [
                    f"Today's topic: {topic_data.get('summary', 'Educational Content')}",
                    "Perfect for beginners",
                    "We'll learn together step by step"
                ]
            },
            'intermediate': {
                'path': None,
                'title': 'Educational Content Overview',
                'content': topic_data.get('key_points', ['Educational concepts to explore'])[:3]
            },
            'advanced': {
                'path': None,
                'title': 'Advanced Theoretical Analysis',
                'content': [
                    f"Complex examination: {topic_data.get('summary', 'Advanced Topic')}",
                    "Sophisticated theoretical framework",
                    "Critical analytical perspectives"
                ]
            }
        }
        
        return [basic_slide_content.get(difficulty_level, basic_slide_content['intermediate'])]

    # Keep all original methods for backward compatibility
    def _generate_professional_script(self, topic_data, persona, length, include_examples):
        """Generate professional educational script with timing markers (original method)"""
        
        system_prompt = f"""You are a professional educational video script writer. Create a {length}-minute script.

PERSONA: {persona}
CONTENT: Must be 100% based on provided textbook content
STRUCTURE: Include timing markers [0:30], [1:15] etc.
SLIDES: Mark slide changes with [SLIDE X: Title]
VISUALS: Include [VISUAL: description] for slide content
EXAMPLES: Include real-world examples that connect to the concept

SCRIPT REQUIREMENTS:
1. Start with engaging hook (15 seconds)
2. Clear introduction (30 seconds) 
3. Main content with 3-4 key points
4. Real-world examples throughout
5. Summary and conclusion (30 seconds)
6. Include slide transition markers
7. Stay strictly within textbook content

TOPIC INFORMATION:
- Main Topic: {topic_data.get('summary', 'Educational Topic')}
- Key Points: {', '.join(topic_data.get('key_points', [])[:4])}
- Content: {topic_data.get('content', '')[:800]}
- Difficulty: {topic_data.get('difficulty', 'Intermediate')}"""

        content_prompt = f"""Create an engaging {length}-minute educational video script about:

TOPIC: {topic_data.get('summary', 'Topic')}

KEY CONCEPTS TO COVER:
{chr(10).join(f"• {point}" for point in topic_data.get('key_points', [])[:5])}

CONTENT DETAILS:
{topic_data.get('content', '')[:1000]}

Make it educational, accurate, and engaging while staying true to the textbook content."""

        try:
            full_script = self._call_groq_api(content_prompt, system_prompt)
            
            # Clean script for TTS (remove markers)
            clean_script = self._clean_script_for_audio(full_script)
            
            return {
                'full_script': full_script,
                'clean_script': clean_script,
                'slide_markers': self._extract_slide_markers(full_script)
            }
            
        except Exception as e:
            # Fallback script generation
            return self._generate_fallback_script(topic_data, persona, length)
    
    def _create_professional_slides(self, topic_data, script_data, chapter_name):
        """Create professional educational slides (original method)"""
        slides_created = []
        
        try:
            # Extract topics and content from script
            slide_contents = self._parse_slide_content(script_data, topic_data)
            
            for i, slide_content in enumerate(slide_contents):
                slide_path = f"{self.output_dir}/slide_{i+1}_{int(time.time())}.png"
                
                self._create_slide_image(
                    slide_content, 
                    slide_path, 
                    i+1, 
                    len(slide_contents),
                    chapter_name
                )
                
                slides_created.append({
                    'path': slide_path,
                    'title': slide_content.get('title', f'Slide {i+1}'),
                    'content': slide_content.get('points', [])
                })
            
            return slides_created
            
        except Exception as e:
            st.warning(f"Slide creation warning: {e}")
            return self._create_basic_slides(topic_data)
    
    def _create_slide_image(self, slide_content, output_path, slide_num, total_slides, chapter_name):
        """Create a professional educational slide image (original method)"""
        # Canvas setup
        width, height = 1920, 1080
        
        # Color scheme
        colors = {
            'primary': '#2c3e50',      # Dark blue-gray
            'secondary': '#3498db',     # Blue
            'accent': '#e74c3c',       # Red
            'background': '#ecf0f1',   # Light gray
            'text': '#2c3e50',         # Dark text
            'white': '#ffffff'
        }
        
        # Create image
        img = Image.new('RGB', (width, height), color=colors['background'])
        draw = ImageDraw.Draw(img)
        
        # Load fonts (with fallbacks)
        try:
            title_font = ImageFont.truetype("arial.ttf", 72)
            subtitle_font = ImageFont.truetype("arial.ttf", 48)
            body_font = ImageFont.truetype("arial.ttf", 36)
            small_font = ImageFont.truetype("arial.ttf", 28)
        except:
            # Fallback to default font
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
            body_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Header section
        draw.rectangle([(0, 0), (width, 150)], fill=colors['primary'])
        
        # Chapter name
        if chapter_name:
            draw.text((60, 30), chapter_name, fill=colors['white'], font=subtitle_font)
        
        # Slide title
        title = slide_content.get('title', f'Slide {slide_num}')
        draw.text((60, 200), title, fill=colors['primary'], font=title_font)
        
        # Content area
        y_position = 320
        bullet_points = slide_content.get('points', [])
        
        for i, point in enumerate(bullet_points[:5]):  # Max 5 points per slide
            # Bullet point
            draw.ellipse([(80, y_position + 10), (100, y_position + 30)], fill=colors['secondary'])
            
            # Text (wrap long text)
            wrapped_text = self._wrap_text(point, 80)  # 80 chars per line
            for line in wrapped_text:
                draw.text((130, y_position), line, fill=colors['text'], font=body_font)
                y_position += 45
            
            y_position += 20  # Space between points
        
        # Footer
        draw.rectangle([(0, height-80), (width, height)], fill=colors['primary'])
        draw.text((60, height-60), f"Slide {slide_num} of {total_slides}", 
                 fill=colors['white'], font=small_font)
        
        # Logo/branding area (right side)
        draw.text((width-300, height-60), "AI Educational Assistant", 
                 fill=colors['white'], font=small_font)
        
        # Save image
        img.save(output_path)
        st.success(f"✅ Created slide {slide_num}: {title}")
    
    def _parse_slide_content(self, script_data, topic_data):
        """Parse script to extract slide content (original method)"""
        full_script = script_data.get('full_script', '')
        
        # Look for slide markers
        slide_pattern = r'\[SLIDE \d+:?\s*([^\]]+)\]'
        slide_matches = re.findall(slide_pattern, full_script)
        
        if slide_matches:
            slides = []
            for i, title in enumerate(slide_matches):
                slides.append({
                    'title': title.strip(),
                    'points': self._extract_slide_points(full_script, title, i)
                })
            return slides
        else:
            # Generate slides from topic data
            return self._generate_slides_from_topic(topic_data)
    
    def _extract_slide_markers(self, script):
        """Extract slide markers from script"""
        markers = re.findall(r'\[SLIDE \d+:?\s*([^\]]+)\]', script)
        return markers

    def _extract_slide_points(self, script, title, slide_index):
        """Extract points for a specific slide (original method)"""
        # Extract content between slide markers
        lines = script.split('\n')
        slide_content = []
        capturing = False
        
        for line in lines:
            if f'[SLIDE {slide_index + 1}' in line or title in line:
                capturing = True
                continue
            elif '[SLIDE' in line and capturing:
                break
            elif capturing and line.strip():
                # Clean and add meaningful content
                clean_line = re.sub(r'\[.*?\]', '', line).strip()
                if clean_line and len(clean_line) > 10:
                    slide_content.append(clean_line)
        
        # If no specific content found, return generic points
        if not slide_content:
            return [
                f"Key concept: {title}",
                "Important learning objectives",
                "Practical applications"
            ]
        
        return slide_content[:5]  # Limit to 5 points

    def _create_basic_slides(self, topic_data):
        """Create basic slides as fallback (original method)"""
        return [{
            'path': None,
            'title': 'Educational Content',
            'content': topic_data.get('key_points', ['No content available'])[:3]
        }]
    
    def _generate_slides_from_topic(self, topic_data):
        """Generate slides directly from topic data (original method)"""
        slides = []
        
        # Title slide
        slides.append({
            'title': 'Introduction',
            'points': [
                topic_data.get('summary', 'Welcome to this educational topic'),
                f"Difficulty Level: {topic_data.get('difficulty', 'Intermediate')}",
                "Let's explore the key concepts together"
            ]
        })
        
        # Key points slides
        key_points = topic_data.get('key_points', [])
        if key_points:
            slides.append({
                'title': 'Key Learning Points',
                'points': key_points[:5]
            })
        
        # Examples slide (if available)
        examples = topic_data.get('examples', [])
        if examples:
            slides.append({
                'title': 'Real-World Examples',
                'points': [ex[:100] + "..." if len(ex) > 100 else ex for ex in examples[:3]]
            })
        
        # Summary slide
        slides.append({
            'title': 'Summary',
            'points': [
                "We covered the main concepts",
                "Reviewed practical examples", 
                "Ready for the next topic!"
            ]
        })
        
        return slides
    
    def _create_professional_audio(self, clean_script):
        """Create professional voiceover (original method)"""
        try:
            # Prepare script for TTS
            audio_script = self._prepare_script_for_tts(clean_script)
            
            # Generate TTS
            tts = gTTS(text=audio_script, lang='en', slow=False)
            audio_path = f"{self.output_dir}/voiceover_{int(time.time())}.mp3"
            tts.save(audio_path)
            
            return {
                'path': audio_path,
                'duration': len(audio_script.split()) * 0.5,  # Approximate duration
                'script': audio_script
            }
            
        except Exception as e:
            st.warning(f"Audio generation warning: {e}")
            return {'path': None, 'error': str(e)}
    
    def _prepare_script_for_tts(self, script):
        """Prepare script for text-to-speech (original method)"""
        # Remove timing markers and visual cues
        clean_script = re.sub(r'\[\d+:\d+\]', '', script)
        clean_script = re.sub(r'\[SLIDE.*?\]', '', clean_script)
        clean_script = re.sub(r'\[VISUAL:.*?\]', '', clean_script)
        clean_script = re.sub(r'\[.*?\]', '', clean_script)
        
        # Clean up extra spaces
        clean_script = re.sub(r'\s+', ' ', clean_script).strip()
        
        return clean_script
    
    def _generate_video_concept(self, topic_data):
        """Generate video concept description (original method)"""
        return {
            'title': topic_data.get('summary', 'Educational Video'),
            'description': f"An educational video covering {len(topic_data.get('key_points', []))} key concepts",
            'target_audience': f"{topic_data.get('difficulty', 'Intermediate')} level learners",
            'learning_objectives': topic_data.get('key_points', [])[:3],
            'duration': "5 minutes",
            'format': "Slide-based presentation with professional voiceover"
        }
    
    def _wrap_text(self, text, max_chars):
        """Wrap text for slide display"""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= max_chars:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def _clean_script_for_audio(self, script):
        """Clean script for audio generation (original method)"""
        # Remove all markers and visual cues
        patterns_to_remove = [
            r'\[\d+:\d+\]',          # Time markers
            r'\[SLIDE.*?\]',         # Slide markers
            r'\[VISUAL:.*?\]',       # Visual cues
            r'\[ANIMATE:.*?\]',      # Animation cues
            r'\[TEXT:.*?\]',         # Text overlays
        ]
        
        clean = script
        for pattern in patterns_to_remove:
            clean = re.sub(pattern, '', clean)
        
        # Clean up spacing
        clean = re.sub(r'\s+', ' ', clean).strip()
        
        return clean
    
    def _call_groq_api(self, content, system_prompt):
        """Call Groq API for script generation (original method)"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.GROQ_API_KEY}"
        }
        
        payload = {
            "model": self.config.GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            "max_tokens": 1500,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{self.config.GROQ_BASE_URL}/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            raise Exception(f"API call failed: {response.status_code}")
    
    def _generate_fallback_script(self, topic_data, persona, length):
        """Generate fallback script if AI fails (original method)"""
        script = f"""[0:00] Introduction
        Welcome to today's lesson on {topic_data.get('summary', 'our topic')}.
        
        [0:30] Main Content
        Let's explore the key concepts:
        """
        
        for i, point in enumerate(topic_data.get('key_points', [])[:4]):
            script += f"\n[{i+1}:00] {point}\n"
        
        script += f"\n[{length-1}:00] Summary\nThat concludes our lesson. Thank you for learning with us!"
        
        clean_script = self._clean_script_for_audio(script)
        
        return {
            'full_script': script,
            'clean_script': clean_script,
            'slide_markers': []
        }
