import torch

# Global variable declarations at module level (CRITICAL FIX)
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("⚠️ CLIP not available - some multimodal features disabled")

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False
    print("⚠️ BLIP not available - some multimodal features disabled")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io
import tempfile
import os
from typing import Dict, List, Optional, Tuple
import base64
import re


class OpenSourceMultimodalProcessor:
    def __init__(self):
        # CRITICAL FIX: Declare global variables at start of __init__
        global CLIP_AVAILABLE, BLIP_AVAILABLE, TESSERACT_AVAILABLE
        
        # Initialize device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize CLIP if available
        self.clip_model = None
        self.clip_preprocess = None
        
        if CLIP_AVAILABLE:
            try:
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
                print("✅ CLIP model loaded successfully")
            except Exception as e:
                print(f"⚠️ CLIP loading failed: {e}")
                CLIP_AVAILABLE = False
        
        # Initialize BLIP if available
        self.blip_processor = None
        self.blip_model = None
        if BLIP_AVAILABLE:
            try:
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_model.to(self.device)
                print("✅ BLIP model loaded successfully")
            except Exception as e:
                print(f"⚠️ BLIP loading failed: {e}")
                BLIP_AVAILABLE = False
        
        # Educational image categories
        self.educational_categories = [
            "diagram", "chart", "graph", "illustration", "photograph",
            "map", "timeline", "flowchart", "table", "equation",
            "molecular structure", "anatomical diagram", "geological formation",
            "historical artifact", "scientific instrument", "mathematical graph"
        ]
        
        # Educational context keywords
        self.educational_keywords = [
            "process", "cycle", "structure", "system", "function",
            "relationship", "comparison", "analysis", "classification",
            "experiment", "observation", "hypothesis", "theory",
            "principle", "concept", "example", "application"
        ]
    
    def analyze_educational_images(self, pdf_doc, difficulty_level="intermediate"):
        """Analyze all images in a PDF document for educational content"""
        
        if not pdf_doc:
            return {}
        
        st.info("🖼️ Analyzing images in document...")
        
        image_analysis = {}
        
        try:
            # Extract images from PDF
            images_info = self._extract_images_from_pdf(pdf_doc)
            
            if not images_info:
                st.warning("No images found in document")
                return {}
            
            st.success(f"📸 Found {len(images_info)} images to analyze")
            
            # Analyze each image
            progress_bar = st.progress(0)
            
            for i, (page_num, image_data) in enumerate(images_info):
                progress_bar.progress((i + 1) / len(images_info))
                
                try:
                    # Convert image data to PIL Image
                    image = self._convert_to_pil_image(image_data)
                    
                    if image:
                        # Comprehensive image analysis
                        analysis = self._analyze_single_image(
                            image, page_num, difficulty_level
                        )
                        
                        if analysis:
                            image_key = f"page_{page_num}_image_{i}"
                            image_analysis[image_key] = analysis
                            
                except Exception as e:
                    st.warning(f"Failed to analyze image on page {page_num}: {e}")
                    continue
            
            progress_bar.progress(1.0)
            st.success(f"✅ Completed analysis of {len(image_analysis)} images")
            
            return image_analysis
            
        except Exception as e:
            st.error(f"Image analysis failed: {str(e)}")
            return {}
    
    def _extract_images_from_pdf(self, pdf_doc):
        """Extract images from PDF document"""
        import fitz  # Add this import
        images_info = []
        
        for page_num in range(pdf_doc.page_count):
            page = pdf_doc[page_num]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Extract image data
                    xref = img[0]
                    pix = fitz.Pixmap(pdf_doc, xref)
                    
                    # Convert to image data
                    if pix.n - pix.alpha < 4:  # RGB or grayscale
                        img_data = pix.tobytes("png")
                        images_info.append((page_num, img_data))
                    
                    pix = None  # Free memory
                    
                except Exception as e:
                    st.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")
                    continue
        
        return images_info
    
    def _convert_to_pil_image(self, image_data):
        """Convert image data to PIL Image"""
        try:
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Basic image enhancement
            image = self._enhance_image(image)
            
            return image
            
        except Exception as e:
            st.warning(f"Image conversion failed: {e}")
            return None
    
    def _enhance_image(self, image):
        """Enhance image quality for better analysis"""
        try:
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
            # Resize if too large (for processing efficiency)
            max_size = 1024
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            return image
            
        except Exception:
            return image  # Return original if enhancement fails
    
    def _analyze_single_image(self, image, page_num, difficulty_level):
        """Comprehensive analysis of a single image"""
        
        analysis = {
            'page_number': page_num,
            'image_size': image.size,
            'description': '',
            'educational_elements': [],
            'text_content': '',
            'concepts': [],
            'image_type': 'unknown',
            'educational_value': 'unknown',
            'difficulty_assessment': difficulty_level,
            'suggested_questions': []
        }
        
        try:
            # Method 1: BLIP Image Captioning
            if BLIP_AVAILABLE and self.blip_model:
                analysis['description'] = self._generate_image_caption_blip(image)
            
            # Method 2: CLIP Classification
            if CLIP_AVAILABLE and self.clip_model:
                analysis['image_type'] = self._classify_image_type_clip(image)
                analysis['educational_elements'] = self._identify_educational_elements_clip(image)
            
            # Method 3: OCR Text Extraction
            if TESSERACT_AVAILABLE:
                analysis['text_content'] = self._extract_text_from_image(image)
            
            # Method 4: Basic Computer Vision Analysis
            analysis.update(self._analyze_image_properties(image))
            
            # Method 5: Educational Context Analysis
            analysis['concepts'] = self._extract_concepts_from_image_analysis(analysis)
            analysis['educational_value'] = self._assess_educational_value(analysis)
            
            # Method 6: Generate Questions
            analysis['suggested_questions'] = self._generate_image_questions(analysis, difficulty_level)
            
            return analysis
            
        except Exception as e:
            st.warning(f"Image analysis failed: {e}")
            return analysis
    
    def _generate_image_caption_blip(self, image):
        """Generate image caption using BLIP model"""
        try:
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                generated_ids = self.blip_model.generate(**inputs, max_length=50)
            
            caption = self.blip_processor.decode(generated_ids[0], skip_special_tokens=True)
            
            # Clean and enhance caption for educational context
            caption = self._enhance_caption_for_education(caption)
            
            return caption
            
        except Exception as e:
            st.warning(f"BLIP captioning failed: {e}")
            return "Image description not available"
    
    def _classify_image_type_clip(self, image):
        """Classify image type using CLIP"""
        try:
            # Prepare image
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            # Prepare text prompts for classification
            text_prompts = [f"a {category}" for category in self.educational_categories]
            text_inputs = clip.tokenize(text_prompts).to(self.device)
            
            with torch.no_grad():
                # Get predictions
                logits_per_image, logits_per_text = self.clip_model(image_input, text_inputs)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            
            # Get top prediction
            top_idx = np.argmax(probs)
            confidence = probs[0][top_idx]
            
            if confidence > 0.3:  # Confidence threshold
                return self.educational_categories[top_idx]
            else:
                return "educational_image"
                
        except Exception as e:
            st.warning(f"CLIP classification failed: {e}")
            return "unknown"
    
    def _identify_educational_elements_clip(self, image):
        """Identify educational elements using CLIP"""
        try:
            # Prepare image
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            # Check for various educational elements
            element_prompts = [
                "text labels", "arrows", "numbers", "measurements",
                "scientific notation", "mathematical equations", "diagrams",
                "charts", "graphs", "illustrations", "photographs",
                "anatomical parts", "molecular structures", "maps"
            ]
            
            identified_elements = []
            
            for element in element_prompts:
                text_prompt = f"an image with {element}"
                text_input = clip.tokenize([text_prompt]).to(self.device)
                
                with torch.no_grad():
                    logits_per_image, _ = self.clip_model(image_input, text_input)
                    prob = torch.softmax(logits_per_image, dim=-1).cpu().numpy()[0][0]
                
                if prob > 0.4:  # Threshold for element detection
                    identified_elements.append(element)
            
            return identified_elements
            
        except Exception as e:
            st.warning(f"Element identification failed: {e}")
            return []
    
    def _extract_text_from_image(self, image):
        """Extract text from image using OCR"""
        if not TESSERACT_AVAILABLE:
            return ""
        
        try:
            # Convert PIL image to numpy array for OpenCV
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Preprocess image for better OCR
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get better text extraction
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Extract text
            text = pytesseract.image_to_string(thresh, config='--psm 6')
            
            # Clean extracted text
            text = ' '.join(text.split())  # Remove extra whitespace
            text = text.strip()
            
            return text if len(text) > 3 else ""
            
        except Exception as e:
            st.warning(f"OCR failed: {e}")
            return ""
    
    def _analyze_image_properties(self, image):
        """Analyze basic image properties"""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Calculate basic properties
            properties = {
                'aspect_ratio': image.size[0] / image.size[1],
                'color_variance': np.var(img_array),
                'brightness': np.mean(img_array),
                'is_grayscale': len(img_array.shape) == 2 or (len(img_array.shape) == 3 and img_array.shape[2] == 1),
                'dominant_colors': self._get_dominant_colors(img_array),
                'complexity_score': self._calculate_visual_complexity(img_array)
            }
            
            return properties
            
        except Exception as e:
            st.warning(f"Property analysis failed: {e}")
            return {}
    
    def _get_dominant_colors(self, img_array):
        """Get dominant colors in image"""
        try:
            # Reshape image to list of pixels
            pixels = img_array.reshape(-1, img_array.shape[-1])
            
            # Use k-means clustering to find dominant colors
            from sklearn.cluster import KMeans
            
            # Limit to reasonable number of pixels for performance
            if len(pixels) > 10000:
                indices = np.random.choice(len(pixels), 10000, replace=False)
                pixels = pixels[indices]
            
            kmeans = KMeans(n_clusters=3, random_state=42)
            kmeans.fit(pixels)
            
            colors = kmeans.cluster_centers_.astype(int)
            
            # Convert to color names/descriptions
            color_descriptions = []
            for color in colors:
                if len(color) == 3:  # RGB
                    r, g, b = color
                    if r > 200 and g > 200 and b > 200:
                        color_descriptions.append("white/light")
                    elif r < 50 and g < 50 and b < 50:
                        color_descriptions.append("black/dark")
                    elif r > g and r > b:
                        color_descriptions.append("red/warm")
                    elif g > r and g > b:
                        color_descriptions.append("green")
                    elif b > r and b > g:
                        color_descriptions.append("blue/cool")
                    else:
                        color_descriptions.append("mixed/neutral")
            
            return color_descriptions[:3]
            
        except Exception:
            return ["mixed"]
    
    def _calculate_visual_complexity(self, img_array):
        """Calculate visual complexity score"""
        try:
            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Calculate edge density as complexity measure
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Calculate texture complexity using local standard deviation
            kernel = np.ones((9, 9), np.float32) / 81
            local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel)
            texture_complexity = np.mean(np.sqrt(local_variance))
            
            # Combine metrics
            complexity_score = (edge_density * 0.6 + (texture_complexity / 255) * 0.4)
            
            return min(complexity_score, 1.0)
            
        except Exception:
            return 0.5  # Default medium complexity
    
    def _enhance_caption_for_education(self, caption):
        """Enhance image caption for educational context"""
        
        # Add educational context words
        educational_replacements = {
            "shows": "illustrates",
            "picture": "educational diagram",
            "image": "illustration", 
            "photo": "educational photograph"
        }
        
        enhanced_caption = caption
        for old_word, new_word in educational_replacements.items():
            enhanced_caption = enhanced_caption.replace(old_word, new_word)
        
        return enhanced_caption
    
    def _extract_concepts_from_image_analysis(self, analysis):
        """Extract educational concepts from image analysis"""
        concepts = set()
        
        # From description
        description = analysis.get('description', '').lower()
        for keyword in self.educational_keywords:
            if keyword in description:
                concepts.add(keyword)
        
        # From text content
        text_content = analysis.get('text_content', '').lower()
        
        # Extract scientific terms, equations, etc.
        scientific_patterns = [
            r'\b[A-Z][a-z]+\b',  # Proper nouns (likely scientific terms)
            r'\b\w+tion\b',      # -tion words (often scientific)
            r'\b\w+ism\b',       # -ism words  
            r'\b\w+ogy\b',       # -ogy words
        ]
        
        for pattern in scientific_patterns:
            matches = re.findall(pattern, text_content)
            concepts.update(match.lower() for match in matches if len(match) > 3)
        
        # From image type
        image_type = analysis.get('image_type', '')
        if image_type != 'unknown':
            concepts.add(image_type)
        
        # From educational elements
        elements = analysis.get('educational_elements', [])
        concepts.update(elements)
        
        return list(concepts)[:10]  # Limit to top 10
    
    def _assess_educational_value(self, analysis):
        """Assess educational value of the image"""
        
        value_score = 0
        
        # Text content adds value
        if analysis.get('text_content', ''):
            value_score += 0.3
        
        # Educational elements add value
        elements = analysis.get('educational_elements', [])
        value_score += min(len(elements) * 0.1, 0.4)
        
        # Recognized concepts add value
        concepts = analysis.get('concepts', [])
        value_score += min(len(concepts) * 0.05, 0.3)
        
        # Image type recognition adds value
        if analysis.get('image_type', 'unknown') != 'unknown':
            value_score += 0.2
        
        # Complexity adds some value
        complexity = analysis.get('complexity_score', 0.5)
        if 0.3 < complexity < 0.8:  # Not too simple, not too complex
            value_score += 0.1
        
        # Classify educational value
        if value_score >= 0.8:
            return "high"
        elif value_score >= 0.5:
            return "medium"
        elif value_score >= 0.2:
            return "low"
        else:
            return "minimal"
    
    def _generate_image_questions(self, analysis, difficulty_level):
        """Generate questions based on image analysis"""
        
        questions = []
        
        description = analysis.get('description', '')
        text_content = analysis.get('text_content', '')
        concepts = analysis.get('concepts', [])
        image_type = analysis.get('image_type', '')
        
        # Question templates based on difficulty
        if difficulty_level == "beginner":
            templates = [
                "What do you see in this {}?",
                "Identify the main parts shown in this image.",
                "What is the purpose of this {}?",
            ]
        elif difficulty_level == "intermediate":
            templates = [
                "Explain the relationship between the elements shown in this {}.",
                "How does this {} demonstrate the concept of {}?",
                "What conclusions can you draw from this {}?",
            ]
        else:  # advanced
            templates = [
                "Analyze the data/information presented in this {} and discuss its implications.",
                "Critically evaluate the method/process shown in this {}.",
                "How would you modify or improve what is shown in this {}?",
            ]
        
        # Generate questions based on analysis
        for template in templates[:2]:  # Limit to 2 questions per image
            try:
                if '{}' in template:
                    if image_type != 'unknown':
                        question = template.format(image_type)
                    else:
                        question = template.format("image")
                else:
                    question = template
                
                questions.append(question)
            except:
                continue
        
        # Add concept-specific questions if concepts identified
        if concepts and difficulty_level != "beginner":
            primary_concept = concepts[0] if concepts else "concept"
            questions.append(f"How does this image relate to the concept of {primary_concept}?")
        
        return questions[:3]  # Maximum 3 questions per image
    
    def create_image_summary_report(self, image_analysis):
        """Create a comprehensive summary report of image analysis"""
        
        if not image_analysis:
            return "No images found for analysis."
        
        report = []
        report.append("# 📸 Image Analysis Report\n")
        
        # Summary statistics
        total_images = len(image_analysis)
        images_with_text = len([img for img in image_analysis.values() if img.get('text_content')])
        high_value_images = len([img for img in image_analysis.values() if img.get('educational_value') == 'high'])
        
        report.append(f"**Total Images Analyzed:** {total_images}")
        report.append(f"**Images with Text:** {images_with_text}")
        report.append(f"**High Educational Value Images:** {high_value_images}")
        report.append("")
        
        # Detailed analysis for each image
        for image_key, analysis in image_analysis.items():
            report.append(f"## {image_key}")
            report.append(f"**Page:** {analysis.get('page_number', 'Unknown')}")
            report.append(f"**Type:** {analysis.get('image_type', 'Unknown')}")
            report.append(f"**Educational Value:** {analysis.get('educational_value', 'Unknown')}")
            
            if analysis.get('description'):
                report.append(f"**Description:** {analysis['description']}")
            
            if analysis.get('text_content'):
                report.append(f"**Text Content:** {analysis['text_content'][:200]}...")
            
            if analysis.get('concepts'):
                concepts_str = ", ".join(analysis['concepts'][:5])
                report.append(f"**Key Concepts:** {concepts_str}")
            
            if analysis.get('suggested_questions'):
                report.append("**Suggested Questions:**")
                for i, question in enumerate(analysis['suggested_questions'], 1):
                    report.append(f"{i}. {question}")
            
            report.append("")
        
        return "\n".join(report)
    
    def extract_text_from_all_images(self, image_analysis):
        """Extract all text content from analyzed images"""
        
        all_text = []
        
        for image_key, analysis in image_analysis.items():
            text_content = analysis.get('text_content', '').strip()
            if text_content and len(text_content) > 5:  # Minimum meaningful text
                page_num = analysis.get('page_number', 'Unknown')
                all_text.append({
                    'page': page_num,
                    'image_id': image_key,
                    'text': text_content,
                    'concepts': analysis.get('concepts', [])
                })
        
        return all_text
    
    def get_image_based_quiz_questions(self, image_analysis, difficulty_level="intermediate"):
        """Generate quiz questions based on all analyzed images"""
        
        all_questions = []
        
        for image_key, analysis in image_analysis.items():
            questions = analysis.get('suggested_questions', [])
            
            for question in questions:
                quiz_question = {
                    'question': question,
                    'type': 'short_answer',  # Image-based questions are typically open-ended
                    'difficulty': difficulty_level,
                    'image_reference': image_key,
                    'page_number': analysis.get('page_number'),
                    'concepts': analysis.get('concepts', []),
                    'educational_value': analysis.get('educational_value', 'medium')
                }
                
                all_questions.append(quiz_question)
        
        # Sort by educational value and page number
        all_questions.sort(key=lambda x: (
            {'high': 3, 'medium': 2, 'low': 1, 'minimal': 0}.get(x['educational_value'], 0),
            x.get('page_number', 999)
        ), reverse=True)
        
        return all_questions[:20]  # Return top 20 questions


# Global instance for easy access (moved to bottom to avoid circular imports)
multimodal_processor = OpenSourceMultimodalProcessor()
