import streamlit as st
import os
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Optional
import time

# Try to import video/audio libraries
try:
    from moviepy.editor import *
    from moviepy import VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

try:
    from TTS.api import TTS
    COQUI_TTS_AVAILABLE = True
except ImportError:
    COQUI_TTS_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import requests
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from config import Config

class OpenSourceVideoGenerator:
    def __init__(self):
        """Initialize open-source video generator"""
        
        self.config = Config()
        
        # Initialize TTS
        self.tts_engine = None
        if COQUI_TTS_AVAILABLE:
            try:
                # Initialize Coqui TTS with a fast model
                self.tts_engine = TTS("tts_models/en/ljspeech/tacotron2-DDC_ph")
                st.success("✅ Coqui TTS initialized")
            except Exception as e:
                st.warning(f"Coqui TTS failed: {e}")
        
        # Fallback to gTTS
        self.use_gtts = GTTS_AVAILABLE and not self.tts_engine
        
        # Video settings
        self.video_settings = {
            'resolution': (1920, 1080),
            'fps': 24,
            'duration_per_slide': 8,  # seconds
            'transition_duration': 1,
            'background_color': (240, 240, 255),
            'text_color': (20, 20, 60),
            'accent_color': (70, 130, 200)
        }
        
        # Teaching persona configurations
        self.persona_configs = {
            "Simple Tutor": {
                "pace": "slow",
                "language_level": "basic",
                "examples": "many",
                "explanations": "detailed",
                "voice_speed": 0.8
            },
            "Professional Instructor": {
                "pace": "moderate",
                "language_level": "standard",
                "examples": "balanced",
                "explanations": "clear",
                "voice_speed": 1.0
            },
            "Expert Academic": {
                "pace": "fast",
                "language_level": "advanced",
                "examples": "technical",
                "explanations": "comprehensive",
                "voice_speed": 1.2
            }
        }
    
    def create_professional_video_with_difficulty(self, topic_data, chapter_name, persona, length, include_examples, difficulty_level, script_guidance):
        """Create professional educational video with difficulty adaptation"""
        
        try:
            st.info(f"🎬 Creating {difficulty_level} level educational video...")
            
            # Step 1: Generate enhanced script
            with st.spinner("📝 Generating AI script..."):
                script = self._generate_enhanced_script(
                    topic_data, chapter_name, persona, length, 
                    include_examples, difficulty_level, script_guidance
                )
            
            if not script:
                return {'success': False, 'error': 'Script generation failed'}
            
            # Step 2: Create slides
            with st.spinner("🖼️ Generating educational slides..."):
                slides = self._create_educational_slides(
                    script, topic_data, difficulty_level
                )
            
            # Step 3: Generate narration
            with st.spinner("🎵 Generating voice narration..."):
                audio_path = self._generate_narration_with_difficulty(
                    script, persona, difficulty_level
                )
            
            # Step 4: Assemble video
            with st.spinner("🎬 Assembling professional video..."):
                video_path = self._assemble_educational_video(
                    slides, audio_path, script, difficulty_level
                )
            
            return {
                'success': True,
                'video_path': video_path,
                'script': script,
                'slides': slides,
                'audio': {'path': audio_path, 'duration': self._get_audio_duration(audio_path)},
                'video_concept': {
                    'title': f"{chapter_name}: {topic_data.get('summary', 'Educational Video')[:50]}",
                    'duration': f"{length} minutes",
                    'format': 'Educational slide presentation with narration',
                    'difficulty': difficulty_level,
                    'persona': persona
                }
            }
            
        except Exception as e:
            st.error(f"Video generation failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _generate_enhanced_script(self, topic_data, chapter_name, persona, length, include_examples, difficulty_level, script_guidance):
        """Generate enhanced script using local models"""
        
        # Try Ollama first (local AI)
        if OLLAMA_AVAILABLE:
            script = self._generate_script_with_ollama(
                topic_data, chapter_name, persona, length, 
                include_examples, difficulty_level, script_guidance
            )
            if script:
                return script
        
        # Fallback to Groq API
        script = self._generate_script_with_groq(
            topic_data, chapter_name, persona, length, 
            include_examples, difficulty_level, script_guidance
        )
        
        if script:
            return script
        
        # Final fallback to template-based generation
        return self._generate_template_script(topic_data, chapter_name, difficulty_level)
    
    def _generate_script_with_ollama(self, topic_data, chapter_name, persona, length, include_examples, difficulty_level, script_guidance):
        """Generate script using local Ollama model"""
        
        try:
            persona_config = self.persona_configs.get(persona, self.persona_configs["Professional Instructor"])
            
            prompt = f"""Create a {length}-minute educational video script about:
            
Chapter: {chapter_name}
Topic: {topic_data.get('summary', '')}
Key Points: {', '.join(topic_data.get('key_points', [])[:5])}
Difficulty Level: {difficulty_level}
Teaching Persona: {persona}

{script_guidance}

Script Requirements:
- Pace: {persona_config['pace']}
- Language Level: {persona_config['language_level']} 
- Include Examples: {'Yes' if include_examples else 'No'}
- Duration: Approximately {length} minutes of spoken content
- Format: Clear sections for slide transitions

Create an engaging, educational script that explains the concepts clearly for {difficulty_level} level learners."""

            response = ollama.chat(model='llama3.2', messages=[
                {'role': 'user', 'content': prompt}
            ])
            
            if response and 'message' in response:
                script = response['message']['content']
                return self._clean_and_structure_script(script)
            
        except Exception as e:
            st.warning(f"Ollama script generation failed: {e}")
        
        return None
    
    def _generate_script_with_groq(self, topic_data, chapter_name, persona, length, include_examples, difficulty_level, script_guidance):
        """Generate script using Groq API"""
        
        try:
            persona_config = self.persona_configs.get(persona, self.persona_configs["Professional Instructor"])
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.GROQ_API_KEY}"
            }
            
            prompt = f"""Create a professional educational video script:
            
Topic: {chapter_name} - {topic_data.get('summary', '')}
Difficulty: {difficulty_level}
Duration: {length} minutes
Persona: {persona} ({persona_config['language_level']} language, {persona_config['pace']} pace)

{script_guidance}

Key Points to Cover:
{chr(10).join('- ' + point for point in topic_data.get('key_points', [])[:5])}

Create a structured script with clear slide sections, engaging narration, and appropriate {difficulty_level} level explanations."""

            payload = {
                "model": self.config.GROQ_MODEL,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 2000,
                "temperature": 0.7
            }
            
            response = requests.post(
                f"{self.config.GROQ_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                script = result['choices'][0]['message']['content']
                return self._clean_and_structure_script(script)
            
        except Exception as e:
            st.warning(f"Groq script generation failed: {e}")
        
        return None
    
    def _generate_template_script(self, topic_data, chapter_name, difficulty_level):
        """Generate script using templates as fallback"""
        
        template_intro = {
            'beginner': "Welcome to today's lesson! We're going to learn about something really interesting and important.",
            'intermediate': "Today we'll explore an important concept that builds on what you already know.",
            'advanced': "In this comprehensive analysis, we'll examine the complex relationships and implications of this topic."
        }
        
        script_parts = []
        
        # Introduction
        intro = template_intro.get(difficulty_level, template_intro['intermediate'])
        script_parts.append(f"[SLIDE 1: Title]\n{intro} Our topic today is {chapter_name}.")
        
        # Main content
        summary = topic_data.get('summary', 'This important educational topic')
        script_parts.append(f"[SLIDE 2: Overview]\n{summary}")
        
        # Key points
        key_points = topic_data.get('key_points', [])
        for i, point in enumerate(key_points[:5], 3):
            script_parts.append(f"[SLIDE {i}: Key Point]\nLet's examine this important aspect: {point}")
        
        # Conclusion
        conclusion = "Let's summarize what we've learned today and how it connects to the bigger picture."
        script_parts.append(f"[SLIDE {len(key_points) + 3}: Conclusion]\n{conclusion}")
        
        return '\n\n'.join(script_parts)
    
    def _clean_and_structure_script(self, raw_script):
        """Clean and structure the generated script"""
        
        # Remove extra whitespace
        script = ' '.join(raw_script.split())
        
        # Add slide markers if missing
        if '[SLIDE' not in script:
            # Split into logical sections and add slide markers
            sections = script.split('. ')
            structured_sections = []
            
            slide_num = 1
            for i in range(0, len(sections), 3):  # Group every 3 sentences
                section_text = '. '.join(sections[i:i+3])
                structured_sections.append(f"[SLIDE {slide_num}]\n{section_text}")
                slide_num += 1
            
            script = '\n\n'.join(structured_sections)
        
        return script
    
    def _create_educational_slides(self, script, topic_data, difficulty_level):
        """Create educational slides from script"""
        
        slides = []
        
        # Parse script for slide sections
        slide_sections = self._parse_script_sections(script)
        
        for i, (slide_title, slide_content) in enumerate(slide_sections):
            slide_path = self._create_single_slide(
                slide_title, slide_content, i + 1, len(slide_sections), difficulty_level
            )
            
            if slide_path:
                slides.append({
                    'title': slide_title,
                    'content': slide_content[:200] + "..." if len(slide_content) > 200 else slide_content,
                    'path': slide_path,
                    'slide_number': i + 1
                })
        
        return slides
    
    def _parse_script_sections(self, script):
        """Parse script into slide sections"""
        
        sections = []
        
        # Look for slide markers
        slide_pattern = r'\[SLIDE\s*(\d+)[^\]]*\](.*?)(?=\[SLIDE|\Z)'
        matches = re.findall(slide_pattern, script, re.DOTALL | re.IGNORECASE)
        
        if matches:
            for slide_num, content in matches:
                title = f"Slide {slide_num}"
                # Try to extract title from first line
                content_lines = content.strip().split('\n')
                if content_lines and len(content_lines[0].strip()) < 100:
                    title = content_lines[0].strip()
                    content = '\n'.join(content_lines[1:]).strip()
                
                sections.append((title, content.strip()))
        else:
            # Fallback: split content into logical sections
            sentences = script.split('. ')
            for i in range(0, len(sentences), 4):  # 4 sentences per slide
                section_content = '. '.join(sentences[i:i+4])
                sections.append((f"Educational Content {i//4 + 1}", section_content))
        
        return sections
    
    def _create_single_slide(self, title, content, slide_num, total_slides, difficulty_level):
        """Create a single educational slide"""
        
        if not PIL_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            st.warning("Slide generation requires PIL and matplotlib")
            return None
        
        try:
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(16, 9))
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis('off')
            
            # Background
            bg_color = {
                'beginner': '#f0f8ff',    # Light blue
                'intermediate': '#f8f8ff', # Ghost white  
                'advanced': '#f5f5f5'     # Light gray
            }.get(difficulty_level, '#f8f8ff')
            
            fig.patch.set_facecolor(bg_color)
            
            # Title
            title_color = {
                'beginner': '#2e8b57',     # Sea green
                'intermediate': '#4682b4', # Steel blue
                'advanced': '#2f4f4f'      # Dark slate gray
            }.get(difficulty_level, '#4682b4')
            
            ax.text(5, 9, title, fontsize=28, fontweight='bold', 
                   ha='center', va='center', color=title_color, wrap=True)
            
            # Content
            # Adjust content for difficulty level
            if difficulty_level == 'beginner':
                font_size = 18
                max_chars = 300
            elif difficulty_level == 'intermediate':
                font_size = 16
                max_chars = 400
            else:  # advanced
                font_size = 14
                max_chars = 500
            
            display_content = content[:max_chars] + "..." if len(content) > max_chars else content
            
            # Break content into lines
            words = display_content.split()
            lines = []
            current_line = []
            
            for word in words:
                current_line.append(word)
                if len(' '.join(current_line)) > 60:  # Wrap at ~60 characters
                    lines.append(' '.join(current_line[:-1]))
                    current_line = [word]
            
            if current_line:
                lines.append(' '.join(current_line))
            
            # Display content lines
            y_pos = 7
            for line in lines[:8]:  # Max 8 lines
                ax.text(5, y_pos, line, fontsize=font_size, ha='center', va='center',
                       color='#333333', wrap=True)
                y_pos -= 0.8
            
            # Footer with slide number and difficulty
            ax.text(9, 0.5, f"Slide {slide_num}/{total_slides} | {difficulty_level.title()}", 
                   fontsize=12, ha='right', va='center', color='#666666')
            
            # Logo/branding area
            ax.text(0.5, 0.5, "IntelliLearn AI", fontsize=12, ha='left', va='center', 
                   color='#666666', style='italic')
            
            # Save slide
            slide_path = os.path.join(self.config.SLIDES_OUTPUT_DIR, f"slide_{slide_num:02d}.png")
            plt.tight_layout()
            plt.savefig(slide_path, dpi=150, bbox_inches='tight', facecolor=bg_color)
            plt.close()
            
            return slide_path
            
        except Exception as e:
            st.warning(f"Slide creation failed: {e}")
            return None
    
    def _generate_narration_with_difficulty(self, script, persona, difficulty_level):
        """Generate narration adapted to difficulty level"""
        
        # Clean script text for TTS
        narration_text = self._extract_narration_text(script)
        
        # Adjust narration for difficulty
        narration_text = self._adapt_narration_for_difficulty(narration_text, difficulty_level)
        
        # Generate audio
        if self.tts_engine:  # Coqui TTS
            return self._generate_audio_coqui(narration_text, persona, difficulty_level)
        elif self.use_gtts:  # gTTS fallback
            return self._generate_audio_gtts(narration_text, difficulty_level)
        else:
            st.warning("No TTS engine available")
            return None
    
    def _extract_narration_text(self, script):
        """Extract clean text for narration from script"""
        
        # Remove slide markers
        text = re.sub(r'\[SLIDE[^\]]*\]', '', script)
        
        # Clean up
        text = ' '.join(text.split())
        
        # Limit length for TTS
        max_length = 2000  # Character limit
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text
    
    def _adapt_narration_for_difficulty(self, text, difficulty_level):
        """Adapt narration text for difficulty level"""
        
        if difficulty_level == 'beginner':
            # Add pauses and slower pacing cues
            text = text.replace('. ', '. ... ')
            text = text.replace('!', '! ...')
            text = text.replace('?', '? ...')
            
            # Add encouraging phrases
            text = "Let's learn something exciting! " + text + " Great job following along!"
            
        elif difficulty_level == 'advanced':
            # Remove some pauses for faster delivery
            text = text.replace('...', '')
            text = text.replace('  ', ' ')
        
        return text
    
    def _generate_audio_coqui(self, text, persona, difficulty_level):
        """Generate audio using Coqui TTS"""
        
        try:
            audio_path = os.path.join(self.config.AUDIO_OUTPUT_DIR, f"narration_{int(time.time())}.wav")
            
            # Generate TTS
            self.tts_engine.tts_to_file(text=text, file_path=audio_path)
            
            return audio_path
            
        except Exception as e:
            st.warning(f"Coqui TTS failed: {e}")
            return self._generate_audio_gtts(text, difficulty_level)
    
    def _generate_audio_gtts(self, text, difficulty_level):
        """Generate audio using gTTS as fallback"""
        
        if not GTTS_AVAILABLE:
            return None
        
        try:
            # Adjust speed for difficulty
            slow_speech = (difficulty_level == 'beginner')
            
            tts = gTTS(text=text[:1000], lang='en', slow=slow_speech)
            
            audio_path = os.path.join(self.config.AUDIO_OUTPUT_DIR, f"narration_{int(time.time())}.mp3")
            tts.save(audio_path)
            
            return audio_path
            
        except Exception as e:
            st.warning(f"gTTS failed: {e}")
            return None
    
    def _assemble_educational_video(self, slides, audio_path, script, difficulty_level):
        """Assemble final educational video"""
        
        if not MOVIEPY_AVAILABLE:
            st.error("MoviePy required for video assembly")
            return None
        
        if not slides:
            st.error("No slides available for video assembly")
            return None
        
        try:
            video_clips = []
            
            # Calculate timing
            if audio_path and os.path.exists(audio_path):
                total_audio_duration = self._get_audio_duration(audio_path)
            else:
                total_audio_duration = len(slides) * self.video_settings['duration_per_slide']
            
            slide_duration = total_audio_duration / len(slides) if slides else 5
            
            # Create video clips from slides
            for slide in slides:
                if slide['path'] and os.path.exists(slide['path']):
                    # Create image clip
                    clip = ImageClip(slide['path']).set_duration(slide_duration)
                    
                    # Add fade transition
                    clip = clip.fadein(0.5).fadeout(0.5)
                    
                    video_clips.append(clip)
            
            if not video_clips:
                st.error("No valid slide clips created")
                return None
            
            # Concatenate video clips
            video = concatenate_videoclips(video_clips, method='compose')
            
            # Add audio if available
            if audio_path and os.path.exists(audio_path):
                audio = AudioFileClip(audio_path)
                
                # Adjust video duration to match audio
                if audio.duration != video.duration:
                    video = video.set_duration(audio.duration)
                
                video = video.set_audio(audio)
            
            # Save final video
            video_path = os.path.join(
                self.config.VIDEO_OUTPUT_DIR, 
                f"educational_video_{difficulty_level}_{int(time.time())}.mp4"
            )
            
            # Write video file
            video.write_videofile(
                video_path,
                fps=self.video_settings['fps'],
                codec='libx264',
                audio_codec='aac',
                temp_audiofile_path=os.path.join(self.config.TEMP_DIR, 'temp_audio.m4a'),
                remove_temp=True
            )
            
            # Clean up
            video.close()
            if audio_path and os.path.exists(audio_path):
                AudioFileClip(audio_path).close()
            
            return video_path
            
        except Exception as e:
            st.error(f"Video assembly failed: {str(e)}")
            return None
    
    def _get_audio_duration(self, audio_path):
        """Get duration of audio file"""
        
        if not audio_path or not os.path.exists(audio_path):
            return 30  # Default duration
        
        try:
            if MOVIEPY_AVAILABLE:
                audio = AudioFileClip(audio_path)
                duration = audio.duration
                audio.close()
                return duration
            else:
                return 30  # Fallback
        except:
            return 30
    
    def create_simple_presentation_video(self, topic_data, difficulty_level="intermediate"):
        """Create simple presentation video without complex dependencies"""
        
        try:
            st.info("🎬 Creating simple educational presentation...")
            
            # Generate simple script
            script = self._generate_simple_script(topic_data, difficulty_level)
            
            # Create text-based slides (simpler approach)
            slides = self._create_simple_slides(script, topic_data, difficulty_level)
            
            # Generate simple audio
            audio_path = self._generate_simple_audio(script, difficulty_level)
            
            result = {
                'success': True,
                'script': script,
                'slides': slides,
                'audio': {'path': audio_path} if audio_path else None,
                'video_concept': {
                    'title': f"Simple Educational Video: {topic_data.get('summary', 'Topic')[:50]}",
                    'format': 'Text-based educational slides',
                    'difficulty': difficulty_level
                }
            }
            
            return result
            
        except Exception as e:
            st.error(f"Simple video creation failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _generate_simple_script(self, topic_data, difficulty_level):
        """Generate simple script without external dependencies"""
        
        summary = topic_data.get('summary', 'Educational topic')
        key_points = topic_data.get('key_points', [])
        
        if difficulty_level == 'beginner':
            intro = "Let's learn about this interesting topic in a simple way."
            point_intro = "Here's something important to remember:"
        elif difficulty_level == 'advanced':
            intro = "We'll analyze this complex topic comprehensively."
            point_intro = "Critical analysis reveals:"
        else:
            intro = "Today we'll explore this educational topic."
            point_intro = "Key point:"
        
        script_parts = [
            f"[Introduction] {intro} {summary}",
        ]
        
        for i, point in enumerate(key_points[:5], 1):
            script_parts.append(f"[Point {i}] {point_intro} {point}")
        
        script_parts.append("[Conclusion] This concludes our educational overview. Thank you for learning with us!")
        
        return '\n\n'.join(script_parts)
    
    def _create_simple_slides(self, script, topic_data, difficulty_level):
        """Create simple text-based slide descriptions"""
        
        slides = []
        sections = script.split('\n\n')
        
        for i, section in enumerate(sections, 1):
            # Extract title and content
            if ']' in section:
                title = section.split(']')[0].replace('[', '')
                content = section.split(']', 1)[1].strip()
            else:
                title = f"Slide {i}"
                content = section
            
            slides.append({
                'title': title,
                'content': content,
                'slide_number': i,
                'path': None  # No actual image file created in simple mode
            })
        
        return slides
    
    def _generate_simple_audio(self, script, difficulty_level):
        """Generate simple audio narration"""
        
        # Extract text from script
        text = re.sub(r'\[[^\]]+\]', '', script)
        text = ' '.join(text.split())
        
        if self.use_gtts:
            return self._generate_audio_gtts(text, difficulty_level)
        else:
            st.info("Audio generation not available - video will be slides only")
            return None
    
    def get_video_generation_capabilities(self):
        """Get current video generation capabilities"""
        
        capabilities = {
            'moviepy_available': MOVIEPY_AVAILABLE,
            'coqui_tts_available': COQUI_TTS_AVAILABLE,
            'gtts_available': GTTS_AVAILABLE,
            'pil_available': PIL_AVAILABLE,
            'matplotlib_available': MATPLOTLIB_AVAILABLE,
            'ollama_available': OLLAMA_AVAILABLE,
            'full_video_generation': MOVIEPY_AVAILABLE and PIL_AVAILABLE and MATPLOTLIB_AVAILABLE,
            'audio_generation': COQUI_TTS_AVAILABLE or GTTS_AVAILABLE,
            'slide_generation': PIL_AVAILABLE and MATPLOTLIB_AVAILABLE,
            'local_script_generation': OLLAMA_AVAILABLE
        }
        
        return capabilities

# Global video generator instance
opensource_video_generator = OpenSourceVideoGenerator()
