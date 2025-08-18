import streamlit as st
import io
import tempfile
import requests
from config import Config
import base64
import time
from typing import Optional, Dict
import json

# Try to import audio libraries
try:
    from audio_recorder_streamlit import audio_recorder
    AUDIO_RECORDER_AVAILABLE = True
except ImportError:
    AUDIO_RECORDER_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import sounddevice as sd
    import wavio
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

class VoiceChatProcessor:
    def __init__(self):
        """Initialize voice chat processor with all available options"""
        self.config = Config()
        
        # Initialize OpenAI client if available
        self.openai_client = None
        if OPENAI_AVAILABLE and self.config.OPENAI_API_KEY:
            try:
                self.openai_client = openai.OpenAI(api_key=self.config.OPENAI_API_KEY)
                st.success("✅ OpenAI client initialized for premium voice features")
            except Exception as e:
                st.warning(f"OpenAI client initialization failed: {e}")
        
        # Voice settings for different difficulty levels
        self.voice_settings = {
            "beginner": {
                "speed": 0.8,
                "voice": "alloy",
                "pitch": "normal",
                "clarity": "high",
                "pause_duration": 0.5
            },
            "intermediate": {
                "speed": 1.0,
                "voice": "echo", 
                "pitch": "normal",
                "clarity": "normal",
                "pause_duration": 0.3
            },
            "advanced": {
                "speed": 1.2,
                "voice": "fable",
                "pitch": "normal", 
                "clarity": "normal",
                "pause_duration": 0.2
            }
        }
    
    def setup_voice_chat_interface(self):
        """Setup comprehensive voice chat interface"""
        
        st.subheader("🎤 Voice-Enabled Learning Chat")
        
        # Current difficulty level
        difficulty_level = st.session_state.get('difficulty_level', 'intermediate')
        st.info(f"🎯 **Voice Settings:** {difficulty_level.title()} level (adjusted speed and complexity)")
        
        # Voice input methods
        voice_method = st.radio(
            "🎙️ Choose Voice Input Method:",
            ["Web Recorder", "File Upload", "Text-to-Speech Only"],
            help="Web Recorder: Record directly in browser | File Upload: Upload audio file | Text-to-Speech: Convert text responses to speech"
        )
        
        if voice_method == "Web Recorder":
            self._setup_web_recorder()
        elif voice_method == "File Upload":
            self._setup_file_upload()
        else:
            self._setup_text_to_speech_only()
    
    def _setup_web_recorder(self):
        """Setup web-based audio recorder"""
        
        if not AUDIO_RECORDER_AVAILABLE:
            st.error("❌ Audio recorder not available. Please install: pip install audio-recorder-streamlit")
            return
        
        st.markdown("### 🎙️ Record Your Question")
        
        # Audio recorder component
        audio_bytes = audio_recorder(
            text="🎤 Click to record your question",
            recording_color="#e74c3c",
            neutral_color="#3498db",
            icon_name="microphone",
            icon_size="1x",
            sample_rate=44100
        )
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            
            # Process voice input
            with st.spinner("🎧 Converting speech to text..."):
                transcript = self._speech_to_text(audio_bytes)
            
            if transcript:
                st.success(f"🎯 **You said:** {transcript}")
                self._process_voice_conversation(transcript)
            else:
                st.error("❌ Could not understand audio. Please try again.")
    
    def _setup_file_upload(self):
        """Setup file upload for audio"""
        
        st.markdown("### 📁 Upload Audio File")
        
        uploaded_audio = st.file_uploader(
            "Upload your audio question",
            type=['wav', 'mp3', 'flac', 'm4a', 'ogg'],
            help="Record on your device and upload the audio file"
        )
        
        if uploaded_audio:
            st.audio(uploaded_audio, format=f"audio/{uploaded_audio.type.split('/')[-1]}")
            
            if st.button("🎧 Process Audio", type="primary"):
                with st.spinner("🎧 Converting speech to text..."):
                    transcript = self._speech_to_text(uploaded_audio.getvalue())
                
                if transcript:
                    st.success(f"🎯 **You said:** {transcript}")
                    self._process_voice_conversation(transcript)
                else:
                    st.error("❌ Could not understand audio. Please try again.")
    
    def _setup_text_to_speech_only(self):
        """Setup text-to-speech only mode"""
        
        st.markdown("### 💬 Text Input with Voice Response")
        
        user_text = st.text_area(
            "Type your question:",
            placeholder="Enter your question about the textbook content...",
            height=100
        )
        
        if user_text and st.button("🗣️ Get Voice Response", type="primary"):
            self._process_voice_conversation(user_text, skip_stt=True)
    
    def _speech_to_text(self, audio_data):
        """Convert speech to text with multiple fallback options"""
        
        try:
            # Method 1: Try Groq Whisper first (best quality, free)
            transcript = self._groq_whisper_stt(audio_data)
            if transcript:
                return transcript
            
            # Method 2: Try OpenAI Whisper (premium quality)
            if self.openai_client:
                            transcript = self._openai_whisper_stt(audio_data)
            if transcript:
                return transcript
            
            # Method 3: Fallback to basic speech recognition (if available)
            st.warning("⚠️ Premium speech recognition failed, using basic fallback")
            return self._fallback_speech_recognition(audio_data)
            
        except Exception as e:
            st.error(f"❌ Speech-to-text failed: {str(e)}")
            return None
    
    def _groq_whisper_stt(self, audio_data):
        """Speech-to-text using Groq Whisper API"""
        
        if not self.config.GROQ_API_KEY:
            return None
        
        try:
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_data)
                tmp_path = tmp_file.name
            
            headers = {
                "Authorization": f"Bearer {self.config.GROQ_API_KEY}",
            }
            
            with open(tmp_path, 'rb') as audio_file:
                files = {"file": audio_file}
                data = {
                    "model": "whisper-large-v3",
                    "language": "en",
                    "response_format": "json"
                }
                
                response = requests.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=30
                )
            
            # Cleanup temp file
            import os
            os.unlink(tmp_path)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('text', '').strip()
            else:
                st.warning(f"Groq Whisper API error: {response.status_code}")
                return None
                
        except Exception as e:
            st.warning(f"Groq Whisper failed: {e}")
            return None
    
    def _openai_whisper_stt(self, audio_data):
        """Speech-to-text using OpenAI Whisper API"""
        
        if not self.openai_client:
            return None
        
        try:
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_data)
                tmp_path = tmp_file.name
            
            with open(tmp_path, 'rb') as audio_file:
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            
            # Cleanup temp file
            import os
            os.unlink(tmp_path)
            
            return transcript.strip() if isinstance(transcript, str) else transcript.text.strip()
            
        except Exception as e:
            st.warning(f"OpenAI Whisper failed: {e}")
            return None
    
    def _fallback_speech_recognition(self, audio_data):
        """Fallback speech recognition method"""
        
        try:
            # Try using SpeechRecognition library if available
            import speech_recognition as sr
            
            r = sr.Recognizer()
            
            # Save audio data to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_data)
                tmp_path = tmp_file.name
            
            # Load audio file
            with sr.AudioFile(tmp_path) as source:
                audio = r.record(source)
            
            # Recognize speech
            text = r.recognize_google(audio)
            
            # Cleanup
            import os
            os.unlink(tmp_path)
            
            return text
            
        except ImportError:
            st.warning("⚠️ Basic speech recognition not available")
            return None
        except Exception as e:
            st.warning(f"Fallback speech recognition failed: {e}")
            return None
    
    def _process_voice_conversation(self, user_input, skip_stt=False):
        """Process voice conversation with AI response and TTS"""
        
        difficulty_level = st.session_state.get('difficulty_level', 'intermediate')
        
        # Add user message to chat history
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
        
        # Add user message
        st.session_state.chat_messages.append({
            "role": "user",
            "content": user_input,
            "type": "voice" if not skip_stt else "text",
            "timestamp": time.time()
        })
        
        # Get AI response
        with st.spinner("🤖 Generating AI response..."):
            ai_response = self._get_ai_response_with_difficulty(user_input, difficulty_level)
        
        # Add AI response to chat
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": ai_response,
            "type": "text",
            "timestamp": time.time()
        })
        
        # Display conversation
        self._display_conversation()
        
        # Generate voice response
        if st.checkbox("🔊 Enable Voice Response", value=True):
            with st.spinner("🎵 Generating voice response..."):
                audio_response = self._text_to_speech(ai_response, difficulty_level)
                
            if audio_response:
                st.subheader("🔊 AI Voice Response")
                st.audio(audio_response, format="audio/mp3")
        
        # Update gamification
        self._update_voice_interaction_progress()
    
    def _get_ai_response_with_difficulty(self, user_message, difficulty_level):
        """Get AI response adapted to difficulty level"""
        
        # Difficulty-specific prompts
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
        
        # Add subject context
        subject = st.session_state.get('selected_subject', 'General')
        system_prompt += f"\n\nCurrent Subject: {subject}"
        
        # Add textbook context if available
        if st.session_state.get('textbook_processed', False):
            system_prompt += "\nYou have access to the user's processed textbook content. Reference it when relevant."
        
        try:
            # Use existing chatbot if available
            if hasattr(st.session_state, 'chatbot') and st.session_state.chatbot:
                return st.session_state.chatbot.get_comprehensive_response(
                    user_message, 
                    system_override=system_prompt
                )
            
            # Fallback to direct API call
            return self._direct_api_call(user_message, system_prompt)
            
        except Exception as e:
            return f"I apologize for the error: {str(e)}. Please try asking your question again."
    
    def _direct_api_call(self, user_message, system_prompt):
        """Direct API call to Groq"""
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.GROQ_API_KEY}"
            }
            
            payload = {
                "model": self.config.GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "max_tokens": self.config.MAX_TOKENS,
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
                return result['choices'][0]['message']['content']
            else:
                return "I'm having trouble processing your request right now. Please try again."
                
        except Exception as e:
            return f"I'm experiencing technical difficulties: {str(e)}. Please try again."
    
    def _text_to_speech(self, text, difficulty_level):
        """Convert text to speech with difficulty-appropriate settings"""
        
        voice_settings = self.voice_settings.get(difficulty_level, self.voice_settings["intermediate"])
        
        # Try premium OpenAI TTS first
        if self.openai_client:
            try:
                response = self.openai_client.audio.speech.create(
                    model="tts-1",
                    voice=voice_settings["voice"],
                    speed=voice_settings["speed"],
                    input=text[:4000]  # API limit
                )
                
                return io.BytesIO(response.content)
                
            except Exception as e:
                st.warning(f"Premium TTS failed: {e}, using fallback")
        
        # Fallback to gTTS
        if GTTS_AVAILABLE:
            try:
                # Adjust text for difficulty level
                processed_text = self._process_text_for_tts(text, difficulty_level)
                
                # Use slower speech for beginners
                slow_speech = (difficulty_level == "beginner")
                
                tts = gTTS(
                    text=processed_text[:1000],  # gTTS character limit
                    lang='en',
                    slow=slow_speech
                )
                
                audio_buffer = io.BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)
                
                return audio_buffer
                
            except Exception as e:
                st.error(f"Text-to-speech failed: {e}")
                return None
        
        st.warning("⚠️ Text-to-speech not available")
        return None
    
    def _process_text_for_tts(self, text, difficulty_level):
        """Process text for better TTS based on difficulty level"""
        
        processed_text = text
        
        if difficulty_level == "beginner":
            # Add pauses for better comprehension
            processed_text = processed_text.replace('. ', '. ... ')
            processed_text = processed_text.replace('! ', '! ... ')
            processed_text = processed_text.replace('? ', '? ... ')
            
            # Slow down complex words
            complex_words = ['photosynthesis', 'mitochondria', 'chromosome', 'ecosystem']
            for word in complex_words:
                if word in processed_text.lower():
                    spaced_word = ' '.join(word)
                    processed_text = processed_text.replace(word, spaced_word)
        
        elif difficulty_level == "advanced":
            # Remove some pauses for faster speech
            processed_text = processed_text.replace('...', '')
        
        return processed_text
    
    def _display_conversation(self):
        """Display the voice conversation history"""
        
        st.subheader("💬 Conversation History")
        
        # Display recent messages
        recent_messages = st.session_state.chat_messages[-6:]  # Show last 6 messages
        
        for message in recent_messages:
            with st.chat_message(message["role"]):
                content = message["content"]
                msg_type = message.get("type", "text")
                
                if msg_type == "voice":
                    st.markdown(f"🎤 **[Voice Input]** {content}")
                else:
                    st.markdown(content)
                
                # Show timestamp for recent messages
                if "timestamp" in message:
                    timestamp = time.strftime("%H:%M:%S", time.localtime(message["timestamp"]))
                    st.caption(f"⏰ {timestamp}")
    
    def _update_voice_interaction_progress(self):
        """Update progress for voice interactions"""
        
        # Update user progress for gamification
        if 'user_progress' in st.session_state:
            st.session_state.user_progress['questions_asked'] += 1
            
            # Voice interaction bonus points
            st.session_state.user_progress['total_points'] += 10
            
            # Update learning streak
            if 'gamification' in st.session_state:
                st.session_state.gamification.update_learning_streak(
                    st.session_state.user_progress
                )
                
                # Check for voice-specific achievements
                st.session_state.gamification.check_achievements(
                    st.session_state.user_progress
                )
    
    def setup_voice_settings_panel(self):
        """Setup voice settings configuration panel"""
        
        with st.sidebar:
            st.markdown("---")
            st.subheader("🎵 Voice Settings")
            
            difficulty_level = st.session_state.get('difficulty_level', 'intermediate')
            current_settings = self.voice_settings[difficulty_level]
            
            st.write(f"**Current Level:** {difficulty_level.title()}")
            
            # Show current voice settings
            with st.expander("🔧 Voice Configuration"):
                st.write(f"**Speed:** {current_settings['speed']}x")
                st.write(f"**Voice:** {current_settings['voice']}")
                st.write(f"**Clarity:** {current_settings['clarity']}")
                
                # Allow override
                if st.checkbox("⚙️ Override Settings"):
                    custom_speed = st.slider("Speech Speed", 0.5, 2.0, current_settings['speed'])
                    
                    if self.openai_client:
                        voice_options = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
                        custom_voice = st.selectbox("Voice Type", voice_options, 
                                                  index=voice_options.index(current_settings['voice']))
                        
                        # Update settings temporarily
                        self.voice_settings[difficulty_level]['speed'] = custom_speed
                        self.voice_settings[difficulty_level]['voice'] = custom_voice
            
            # Voice interaction stats
            if st.session_state.get('user_progress'):
                voice_interactions = st.session_state.user_progress.get('questions_asked', 0)
                st.metric("🎤 Voice Interactions", voice_interactions)
    
    def create_voice_interaction_summary(self):
        """Create summary of voice interactions"""
        
        if 'chat_messages' not in st.session_state:
            return "No voice interactions recorded."
        
        voice_messages = [msg for msg in st.session_state.chat_messages 
                         if msg.get('type') == 'voice']
        
        if not voice_messages:
            return "No voice interactions found."
        
        summary = []
        summary.append("# 🎤 Voice Interaction Summary\n")
        summary.append(f"**Total Voice Messages:** {len(voice_messages)}")
        summary.append(f"**Current Difficulty Level:** {st.session_state.get('difficulty_level', 'intermediate').title()}")
        summary.append("")
        
        # Recent voice interactions
        summary.append("## Recent Voice Questions:")
        for msg in voice_messages[-5:]:  # Last 5 voice messages
            timestamp = ""
            if 'timestamp' in msg:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(msg['timestamp']))
            
            summary.append(f"- **{timestamp}:** {msg['content'][:100]}...")
        
        return "\n".join(summary)
    
    def export_voice_conversation(self):
        """Export voice conversation for download"""
        
        if 'chat_messages' not in st.session_state:
            return None
        
        export_data = {
            'conversation_export': {
                'export_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                'difficulty_level': st.session_state.get('difficulty_level', 'intermediate'),
                'subject': st.session_state.get('selected_subject', 'General'),
                'messages': st.session_state.chat_messages,
                'voice_settings': self.voice_settings,
                'total_voice_interactions': len([m for m in st.session_state.chat_messages if m.get('type') == 'voice'])
            }
        }
        
        return json.dumps(export_data, indent=2)

# Global voice processor instance
voice_processor = VoiceChatProcessor()

