#!/usr/bin/env python3
"""
IntelliLearn AI Model Setup Script
Automatically download and configure all required open-source models
"""

import os
import sys
import subprocess
import streamlit as st
from pathlib import Path
import requests
from typing import List, Dict
import json
import time

class ModelSetupManager:
    def __init__(self):
        """Initialize model setup manager"""
        
        # Get project root directory
        self.project_root = Path(__file__).parent.parent
        self.models_dir = self.project_root / "cache" / "models"
        self.config_file = self.project_root / "cache" / "model_setup.json"
        
        # Ensure directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configurations
        self.model_configs = {
            'sentence_transformers': {
                'name': 'Sentence Transformers - all-MiniLM-L6-v2',
                'model_id': 'all-MiniLM-L6-v2',
                'size': '80MB',
                'description': 'Fast sentence embedding model for educational content analysis',
                'required': True,
                'setup_function': self._setup_sentence_transformers
            },
            'spacy_english': {
                'name': 'spaCy English Language Model',
                'model_id': 'en_core_web_sm',
                'size': '15MB',
                'description': 'English NLP model for text processing and analysis',
                'required': True,
                'setup_function': self._setup_spacy_model
            },
            'ollama_llama': {
                'name': 'Ollama Llama 3.2 (Local AI)',
                'model_id': 'llama3.2',
                'size': '2.0GB',
                'description': 'Local AI model for script generation and content analysis',
                'required': False,
                'setup_function': self._setup_ollama_model
            },
            'blip_caption': {
                'name': 'BLIP Image Captioning',
                'model_id': 'Salesforce/blip-image-captioning-base',
                'size': '990MB',
                'description': 'Image understanding model for educational diagrams',
                'required': False,
                'setup_function': self._setup_blip_model
            },
            'clip_vision': {
                'name': 'CLIP Vision-Language Model',
                'model_id': 'ViT-B/32',
                'size': '150MB',
                'description': 'Vision-language model for image classification',
                'required': False,
                'setup_function': self._setup_clip_model
            },
            'coqui_tts': {
                'name': 'Coqui TTS Voice Model',
                'model_id': 'tts_models/en/ljspeech/tacotron2-DDC_ph',
                'size': '100MB',
                'description': 'High-quality text-to-speech for video generation',
                'required': False,
                'setup_function': self._setup_coqui_tts
            }
        }
        
        self.setup_status = self._load_setup_status()
    
    def _load_setup_status(self):
        """Load setup status from config file"""
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {}
    
    def _save_setup_status(self):
        """Save setup status to config file"""
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.setup_status, f, indent=2)
        except Exception as e:
            print(f"Failed to save setup status: {e}")
    
    def check_system_requirements(self):
        """Check system requirements for model setup"""
        
        requirements = {
            'python_version': sys.version_info >= (3, 8),
            'pip_available': self._command_available('pip'),
            'git_available': self._command_available('git'),
            'disk_space': self._check_disk_space(),
            'internet_connection': self._check_internet_connection()
        }
        
        return requirements
    
    def _command_available(self, command):
        """Check if command is available"""
        
        try:
            subprocess.run([command, '--version'], 
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _check_disk_space(self):
        """Check available disk space (simplified)"""
        
        try:
            # This is a simplified check - in production you'd check actual disk space
            return True
        except Exception:
            return False
    
    def _check_internet_connection(self):
        """Check internet connectivity"""
        
        try:
            response = requests.get('https://httpbin.org/status/200', timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def setup_all_models(self, required_only=True):
        """Setup all models based on requirements"""
        
        print("🚀 IntelliLearn AI - Model Setup Starting...")
        
        # Check system requirements
        requirements = self.check_system_requirements()
        
        if not all(requirements.values()):
            print("❌ System requirements check failed:")
            for req, status in requirements.items():
                print(f"   {req}: {'✅' if status else '❌'}")
            return False
        
        print("✅ System requirements check passed")
        
        # Setup models
        models_to_setup = [
            (key, config) for key, config in self.model_configs.items()
            if not required_only or config['required']
        ]
        
        total_models = len(models_to_setup)
        successful_setups = 0
        
        for i, (model_key, model_config) in enumerate(models_to_setup, 1):
            print(f"\n[{i}/{total_models}] Setting up {model_config['name']}...")
            print(f"Description: {model_config['description']}")
            print(f"Size: {model_config['size']}")
            
            try:
                success = model_config['setup_function'](model_config)
                
                if success:
                    print(f"✅ {model_config['name']} setup completed")
                    self.setup_status[model_key] = {
                        'status': 'completed',
                        'timestamp': time.time(),
                        'model_id': model_config['model_id']
                    }
                    successful_setups += 1
                else:
                    print(f"❌ {model_config['name']} setup failed")
                    self.setup_status[model_key] = {
                        'status': 'failed',
                        'timestamp': time.time(),
                        'error': 'Setup function returned False'
                    }
                
            except Exception as e:
                print(f"❌ {model_config['name']} setup error: {str(e)}")
                self.setup_status[model_key] = {
                    'status': 'error',
                    'timestamp': time.time(),
                    'error': str(e)
                }
        
        # Save setup status
        self._save_setup_status()
        
        # Summary
        print(f"\n🎯 Setup Summary: {successful_setups}/{total_models} models setup successfully")
        
        if successful_setups == total_models:
            print("🎉 All models setup completed! Your IntelliLearn AI platform is ready!")
            return True
        else:
            print(f"⚠️ {total_models - successful_setups} models failed to setup. Check individual error messages above.")
            return False
    
    def _setup_sentence_transformers(self, config):
        """Setup Sentence Transformers model"""
        
        try:
            from sentence_transformers import SentenceTransformer
            
            print("Downloading sentence transformers model...")
            model = SentenceTransformer(config['model_id'])
            
            # Test the model
            test_sentences = ["This is a test sentence.", "This is another test."]
            embeddings = model.encode(test_sentences)
            
            if embeddings is not None and len(embeddings) == 2:
                print("✅ Sentence Transformers model verified")
                return True
            else:
                print("❌ Model verification failed")
                return False
                
        except ImportError:
            print("❌ sentence-transformers package not installed")
            print("💡 Run: pip install sentence-transformers>=5.1.0")
            return False
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False
    
    def _setup_spacy_model(self, config):
        """Setup spaCy English model"""
        
        try:
            # Download spaCy model
            result = subprocess.run([
                sys.executable, '-m', 'spacy', 'download', config['model_id']
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Test the model
                import spacy
                nlp = spacy.load(config['model_id'])
                doc = nlp("This is a test sentence.")
                
                if len(doc) > 0:
                    print("✅ spaCy model verified")
                    return True
                else:
                    print("❌ Model verification failed")
                    return False
            else:
                print(f"❌ spaCy download failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False
    
    def _setup_ollama_model(self, config):
        """Setup Ollama local AI model"""
        
        try:
            # Check if Ollama is installed
            result = subprocess.run(['ollama', '--version'], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                print("❌ Ollama not installed")
                print("💡 Install from: https://ollama.ai/")
                return False
            
            print("Downloading Ollama model (this may take a while)...")
            
            # Pull the model
            result = subprocess.run(['ollama', 'pull', config['model_id']], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                # Test the model
                import ollama
                
                response = ollama.chat(model=config['model_id'], messages=[
                    {'role': 'user', 'content': 'Hello, this is a test.'}
                ])
                
                if response and 'message' in response:
                    print("✅ Ollama model verified")
                    return True
                else:
                    print("❌ Model verification failed")
                    return False
            else:
                print(f"❌ Ollama pull failed: {result.stderr}")
                return False
                
        except ImportError:
            print("❌ ollama package not installed")
            print("💡 Run: pip install ollama>=0.1.0")
            return False
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False
    
    def _setup_blip_model(self, config):
        """Setup BLIP image captioning model"""
        
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            print("Downloading BLIP model...")
            processor = BlipProcessor.from_pretrained(config['model_id'])
            model = BlipForConditionalGeneration.from_pretrained(config['model_id'])
            
            # Test with a simple tensor
            import torch
            from PIL import Image
            import numpy as np
            
            # Create a simple test image
            test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            
            inputs = processor(test_image, return_tensors="pt")
            
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_length=20)
            
            caption = processor.decode(generated_ids[0], skip_special_tokens=True)
            
            if caption:
                print("✅ BLIP model verified")
                return True
            else:
                print("❌ Model verification failed")
                return False
                
        except ImportError:
            print("❌ transformers package not installed")
            print("💡 Run: pip install transformers>=4.44.0")
            return False
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False
    
    def _setup_clip_model(self, config):
        """Setup CLIP vision-language model"""
        
        try:
            import clip
            import torch
            
            print("Downloading CLIP model...")
            model, preprocess = clip.load(config['model_id'])
            
            # Test the model
            import numpy as np
            from PIL import Image
            
            # Create a simple test image
            test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            
            image_input = preprocess(test_image).unsqueeze(0)
            text_inputs = clip.tokenize(["a test image", "a random picture"])
            
            with torch.no_grad():
                logits_per_image, logits_per_text = model(image_input, text_inputs)
            
            if logits_per_image is not None:
                print("✅ CLIP model verified")
                return True
            else:
                print("❌ Model verification failed")
                return False
                
        except ImportError:
            print("❌ clip-by-openai package not installed")
            print("💡 Run: pip install clip-by-openai>=1.0")
            return False
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False
    
    def _setup_coqui_tts(self, config):
        """Setup Coqui TTS model"""
        
        try:
            from TTS.api import TTS
            
            print("Downloading Coqui TTS model...")
            tts = TTS(config['model_id'])
            
            # Test the model
            test_text = "This is a test of the text to speech system."
            temp_file = self.models_dir / "tts_test.wav"
            
            tts.tts_to_file(text=test_text, file_path=str(temp_file))
            
            if temp_file.exists():
                # Clean up test file
                temp_file.unlink()
                print("✅ Coqui TTS model verified")
                return True
            else:
                print("❌ Model verification failed")
                return False
                
        except ImportError:
            print("❌ TTS package not installed")
            print("💡 Run: pip install TTS>=0.20.0")
            return False
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False
    
    def display_setup_status_streamlit(self):
        """Display setup status in Streamlit interface"""
        
        st.subheader("🤖 Model Setup Status")
        
        for model_key, model_config in self.model_configs.items():
            status_info = self.setup_status.get(model_key, {'status': 'not_setup'})
            
            with st.expander(f"{model_config['name']} - {model_config['size']}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Description:** {model_config['description']}")
                    st.write(f"**Required:** {'Yes' if model_config['required'] else 'Optional'}")
                    st.write(f"**Model ID:** {model_config['model_id']}")
                    
                    if status_info['status'] == 'completed':
                        st.success("✅ Setup completed")
                        if 'timestamp' in status_info:
                            setup_time = time.strftime("%Y-%m-%d %H:%M:%S", 
                                                     time.localtime(status_info['timestamp']))
                            st.caption(f"Setup on: {setup_time}")
                    elif status_info['status'] == 'failed':
                        st.error("❌ Setup failed")
                        if 'error' in status_info:
                            st.caption(f"Error: {status_info['error']}")
                    elif status_info['status'] == 'error':
                        st.error("💥 Setup error")
                        if 'error' in status_info:
                            st.caption(f"Error: {status_info['error']}")
                    else:
                        st.warning("⏳ Not setup")
                
                with col2:
                    if st.button(f"Setup", key=f"setup_{model_key}"):
                        with st.spinner(f"Setting up {model_config['name']}..."):
                            try:
                                success = model_config['setup_function'](model_config)
                                if success:
                                    st.success("✅ Setup completed!")
                                    self.setup_status[model_key] = {
                                        'status': 'completed',
                                        'timestamp': time.time(),
                                        'model_id': model_config['model_id']
                                    }
                                else:
                                    st.error("❌ Setup failed!")
                                    self.setup_status[model_key] = {
                                        'status': 'failed',
                                        'timestamp': time.time()
                                    }
                                
                                self._save_setup_status()
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"💥 Setup error: {str(e)}")
                                self.setup_status[model_key] = {
                                    'status': 'error',
                                    'timestamp': time.time(),
                                    'error': str(e)
                                }
                                self._save_setup_status()
        
        # Batch setup buttons
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Setup Required Models", type="primary"):
                with st.spinner("Setting up required models..."):
                    self.setup_all_models(required_only=True)
                st.rerun()
        
        with col2:
            if st.button("📦 Setup All Models"):
                with st.spinner("Setting up all models..."):
                    self.setup_all_models(required_only=False)
                st.rerun()
        
        with col3:
            if st.button("🔄 Check Status"):
                st.rerun()

def main():
    """Main setup function"""
    
    print("🚀 IntelliLearn AI - Model Setup")
    print("=" * 50)
    
    setup_manager = ModelSetupManager()
    
    # Check if running in Streamlit
    try:
        import streamlit as st
        # If we're here, we're in Streamlit
        setup_manager.display_setup_status_streamlit()
        return
    except:
        pass
    
    # Command line interface
    if len(sys.argv) > 1:
        if sys.argv[1] == '--required-only':
            setup_manager.setup_all_models(required_only=True)
        elif sys.argv[1] == '--all':
            setup_manager.setup_all_models(required_only=False)
        elif sys.argv[1] == '--check':
            requirements = setup_manager.check_system_requirements()
            print("\nSystem Requirements:")
            for req, status in requirements.items():
                print(f"  {req}: {'✅' if status else '❌'}")
        else:
            print("Usage: python model_setup.py [--required-only|--all|--check]")
    else:
        # Interactive setup
        print("\nChoose setup option:")
        print("1. Setup required models only")
        print("2. Setup all models")
        print("3. Check system requirements")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            setup_manager.setup_all_models(required_only=True)
        elif choice == '2':
            setup_manager.setup_all_models(required_only=False)
        elif choice == '3':
            requirements = setup_manager.check_system_requirements()
            print("\nSystem Requirements:")
            for req, status in requirements.items():
                print(f"  {req}: {'✅' if status else '❌'}")
        elif choice == '4':
            print("Setup cancelled.")
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
