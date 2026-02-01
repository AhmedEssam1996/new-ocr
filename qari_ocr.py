"""
QARI v0.3 OCR Integration
QARI (Quality Arabic Recognition Interface) - Advanced Arabic OCR
"""

import os
import cv2
import numpy as np
from typing import Optional, List, Tuple
import arabic_reshaper
from bidi.algorithm import get_display
import requests
import re


class QARIOCR:
    """
    QARI v0.3 OCR Processor for Arabic and English text.
    High-quality Arabic OCR with layout preservation.
    """
    
    def __init__(self, use_gpu: bool = False, enable_llm_postprocessing: bool = False,
                 llm_model: str = "mistral"):
        """
        Initialize QARI OCR processor.
        
        Args:
            use_gpu: Whether to use GPU if available
            enable_llm_postprocessing: Enable LLM-based text correction
            llm_model: Ollama model name for postprocessing
        """
        self.use_gpu = use_gpu
        self.enable_llm_postprocessing = enable_llm_postprocessing
        self.llm_model = llm_model
        
        # Try to import QARI - if not available, use fallback
        self.qari_available = False
        try:
            # Try different possible import names for QARI
            try:
                import qari
                self.qari = qari
                self.qari_available = True
                print("QARI v0.3 loaded successfully!")
            except ImportError:
                try:
                    from qari_ocr import QARI
                    self.qari = QARI()
                    self.qari_available = True
                    print("QARI OCR loaded successfully!")
                except ImportError:
                    # Fallback to PaddleOCR if QARI not available
                    from paddleocr import PaddleOCR
                    self.ocr_engine = PaddleOCR(use_angle_cls=True, lang='ar+en', use_gpu=use_gpu)
                    print("Using PaddleOCR as fallback (QARI not found)")
        except Exception as e:
            print(f"Warning: Could not load QARI. Error: {e}")
            print("Falling back to PaddleOCR...")
            from paddleocr import PaddleOCR
            self.ocr_engine = PaddleOCR(use_angle_cls=True, lang='ar+en', use_gpu=use_gpu)
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image to improve OCR accuracy.
        
        Steps:
        1. Convert to grayscale
        2. Enhance contrast using CLAHE
        3. Denoise using bilateral filter
        
        Args:
            image_path: Path to input image
            
        Returns:
            Preprocessed image as numpy array
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise using bilateral filter
        denoised = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
        
        return denoised
    
    def extract_text_qari(self, image_path: str) -> str:
        """
        Extract text using QARI v0.3 if available.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Extracted text string
        """
        if self.qari_available:
            try:
                # QARI API may vary - try common patterns
                if hasattr(self.qari, 'recognize'):
                    result = self.qari.recognize(image_path)
                elif hasattr(self.qari, 'ocr'):
                    result = self.qari.ocr(image_path)
                elif hasattr(self.qari, 'extract_text'):
                    result = self.qari.extract_text(image_path)
                else:
                    # Try calling directly
                    result = self.qari(image_path)
                
                # Handle different return formats
                if isinstance(result, str):
                    return result
                elif isinstance(result, dict):
                    return result.get('text', '') or result.get('result', '')
                elif isinstance(result, list):
                    # Join list of text lines
                    return '\n'.join(str(item) for item in result)
                else:
                    return str(result)
            except Exception as e:
                print(f"QARI extraction error: {e}")
                return self.extract_text_fallback(image_path)
        else:
            return self.extract_text_fallback(image_path)
    
    def extract_text_fallback(self, image_path: str) -> str:
        """
        Fallback OCR using PaddleOCR if QARI is not available.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Extracted text string
        """
        try:
            result = self.ocr_engine.ocr(image_path, cls=True)
            
            text_lines = []
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:
                        text = line[1][0]
                        if text.strip():
                            text_lines.append(text.strip())
            
            return '\n'.join(text_lines)
        except Exception as e:
            raise RuntimeError(f"OCR extraction failed: {e}")
    
    def extract_text_with_layout(self, image_path: str) -> str:
        """
        Extract text while preserving layout.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Text with preserved line breaks
        """
        # Preprocess image
        preprocessed = self.preprocess_image(image_path)
        
        # Save preprocessed image temporarily
        temp_path = image_path.replace('.', '_preprocessed.')
        cv2.imwrite(temp_path, preprocessed)
        
        try:
            # Extract text using QARI or fallback
            text = self.extract_text_qari(temp_path)
            return text
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def is_arabic_text(self, text: str) -> bool:
        """Check if text contains Arabic characters."""
        arabic_range = '\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF'
        return bool(re.search(f'[{arabic_range}]', text))
    
    def process_arabic_text(self, text: str) -> str:
        """
        Process Arabic text: reshape letters and fix RTL direction.
        
        Args:
            text: Raw Arabic text
            
        Returns:
            Properly formatted Arabic text
        """
        if not text:
            return ""
        
        try:
            reshaped = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped)
            return bidi_text
        except Exception:
            return text
    
    def process_mixed_text(self, text: str) -> str:
        """
        Process text that may contain both Arabic and English.
        Only processes Arabic parts, leaves English unchanged.
        
        Args:
            text: Mixed Arabic/English text
            
        Returns:
            Processed text with Arabic parts formatted correctly
        """
        if not text:
            return ""
        
        lines = text.split('\n')
        processed_lines = []
        
        for line in lines:
            if not line.strip():
                processed_lines.append(line)
                continue
            
            if self.is_arabic_text(line):
                processed_line = self.process_arabic_text(line)
            else:
                processed_line = line
            
            processed_lines.append(processed_line)
        
        return '\n'.join(processed_lines)
    
    def postprocess_with_llm(self, text: str) -> str:
        """
        Post-process text using local LLM (Ollama) for minor corrections.
        Preserves layout while fixing OCR errors.
        
        Args:
            text: Text to correct
            
        Returns:
            Corrected text with layout preserved
        """
        if not self.enable_llm_postprocessing:
            return text
        
        prompt = (
            "صحح الأخطاء البسيطة في النص التالي الناتجة عن OCR "
            "دون تغيير التخطيط أو المسافات أو ترتيب الأسطر. "
            "احتفظ بالمسافات والفواصل كما هي:\n\n"
            f"{text}\n\n"
            "النص المصحح (مع الحفاظ على التخطيط):"
        )
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "options": {"temperature": 0.2}
                },
                timeout=60
            )
            result = response.json()
            corrected = result.get("response", text)
            return self.process_mixed_text(corrected)
        except Exception as e:
            print(f"LLM postprocessing error: {e}")
            return text
    
    def process_image_text(self, image_path: str, llm_model: Optional[str] = None) -> str:
        """
        Main function to process image and extract text with layout preservation.
        
        Args:
            image_path: Path to input image
            llm_model: Optional LLM model name (overrides initialization setting)
            
        Returns:
            Formatted text string preserving layout
        """
        # Override LLM model if provided
        original_llm_model = self.llm_model
        if llm_model:
            self.llm_model = llm_model
            self.enable_llm_postprocessing = True
        
        try:
            # Extract text with layout preservation
            raw_text = self.extract_text_with_layout(image_path)
            
            if not raw_text.strip():
                return ""
            
            # Process mixed Arabic/English text
            processed_text = self.process_mixed_text(raw_text)
            
            # Post-process with LLM if enabled
            if self.enable_llm_postprocessing:
                processed_text = self.postprocess_with_llm(processed_text)
            
            return processed_text
            
        finally:
            # Restore original LLM model setting
            if llm_model:
                self.llm_model = original_llm_model


# Standalone function for easy use
def process_image_text(image_path: str, 
                      use_gpu: bool = False,
                      llm_model: Optional[str] = None) -> str:
    """
    Process image and extract text with layout preservation using QARI v0.3.
    
    Convenience function that creates a QARIOCR instance and processes the image.
    
    Args:
        image_path: Path to input image
        use_gpu: Whether to use GPU for processing
        llm_model: Optional LLM model name for postprocessing (e.g., "mistral")
                  If None, LLM postprocessing is disabled
                  
    Returns:
        Formatted text string preserving layout
    """
    enable_llm = llm_model is not None
    processor = QARIOCR(
        use_gpu=use_gpu,
        enable_llm_postprocessing=enable_llm,
        llm_model=llm_model or "mistral"
    )
    return processor.process_image_text(image_path, llm_model=llm_model)


if __name__ == "__main__":
    # Example usage
    print("QARI v0.3 OCR Processor")
    print("=" * 50)
    
    test_image = "test_image.jpg"  # Replace with your image path
    
    if os.path.exists(test_image):
        print(f"Processing image: {test_image}")
        
        # Process without LLM
        result = process_image_text(
            image_path=test_image,
            use_gpu=False,
            llm_model=None
        )
        
        print("\nExtracted text (without LLM):")
        print(result)
        print("\n" + "=" * 50)
        
        # Process with LLM correction
        result_with_llm = process_image_text(
            image_path=test_image,
            use_gpu=False,
            llm_model="mistral"
        )
        
        print("\nExtracted text (with LLM correction):")
        print(result_with_llm)
    else:
        print(f"Image not found: {test_image}")
        print("Please provide a valid image path.")

