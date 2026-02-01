import os
from flask import Flask, request, render_template, jsonify
import requests
from werkzeug.utils import secure_filename
from qari_ocr import QARIOCR, process_image_text

# Flask App Setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize QARI v0.3 OCR processor
qari_processor = QARIOCR(use_gpu=False, enable_llm_postprocessing=False)

# OCR Functions
def extract_arabic_text(image_path):
    """Extract Arabic text from image using QARI v0.3"""
    try:
        # Extract text using QARI v0.3
        cleaned_text = qari_processor.extract_text_with_layout(image_path)
        
        # Process Arabic text (reshaping + bidi)
        processed_text = qari_processor.process_mixed_text(cleaned_text)
        
        # Return both raw and processed text
        return cleaned_text, processed_text
    except Exception as e:
        # Return error message if OCR fails
        error_msg = f"OCR Error: {str(e)}"
        return error_msg, error_msg

def process_with_llm_ollama(text):
    """Send OCR text to local Ollama LLM for error correction and improvement"""
    prompt = (
        "صحح من الأخطاء النحوية واملأ الفراغات وأزل التشويش الناتج عن الأداة لمخرجات النص التالي دون تغيير معناه:\n\n"
        f"{text}\n\nالنص المصحح:"
    )
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral", "prompt": prompt, "options": {"temperature": 0.2}},
            timeout=60
        )
        result = response.json()
        # LLM's raw output
        llm_raw = result.get("response", "")
        # Clean and process the LLM output using QARI processor
        cleaned_llm_text = qari_processor.process_mixed_text(llm_raw)
        return llm_raw, cleaned_llm_text
    except Exception as e:
        return str(e), str(e)

# Routes
@app.route("/", methods=['GET'])
def index():
    """Serve the main HTML page"""
    return render_template("index.html")

@app.route("/ocr", methods=['POST'])
def ocr():
    """Handle image upload, OCR processing, and LLM post-processing"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Extract Arabic text using PaddleOCR (works directly with image path)
        ocr_raw, ocr_arabic_display = extract_arabic_text(filepath)
        
        # LLM Postprocessing
        llm_raw, llm_arabic_display = process_with_llm_ollama(ocr_arabic_display)
        
        return jsonify({
            'ocr_text': ocr_arabic_display,
            'llm_text': llm_arabic_display
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)

