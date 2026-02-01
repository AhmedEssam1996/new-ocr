# Arabic OCR Web Application (Offline)

A complete offline Arabic OCR web application that extracts text from images and improves it using a local LLM.

## Features

- ✅ **Fully Offline**: Works completely locally, no cloud APIs required
- ✅ **Arabic OCR**: Extracts Arabic text from images using PaddleOCR
- ✅ **Easy Setup**: Pure Python library, no system dependencies required
- ✅ **LLM Post-processing**: Corrects OCR errors using local Ollama LLM
- ✅ **RTL Support**: Right-to-left layout optimized for Arabic
- ✅ **Clean UI**: Simple, modern interface

## Requirements

### System Dependencies

1. **Python 3.8+**
2. **Ollama** with Arabic-supporting model
   - Download from [ollama.com](https://ollama.com/download)
   - Pull model: `ollama pull mistral` or `ollama pull llama3`

**Note**: PaddleOCR is a pure Python library with no system dependencies. It will automatically download Arabic models on first use.

### Python Dependencies

All Python packages are listed in `requirements.txt`.

## Installation

### Step 1: Install Ollama

1. Download and install Ollama from [ollama.com](https://ollama.com/download)
2. Pull an Arabic-supporting model:
```bash
ollama pull mistral
# OR
ollama pull llama3
```

### Step 2: Setup Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

## Running the Application

### Step 1: Start Ollama (if not running)

Ollama should be running in the background. If not, start it:
```bash
ollama serve
```

### Step 2: Start Flask Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

### Step 3: Open in Browser

Navigate to [http://localhost:5000](http://localhost:5000) in your web browser.

## Usage

1. **Upload Image**: Click the file input and select an image containing Arabic text
2. **Start OCR**: Click "ابدأ التحويل" (Start Conversion) button
3. **View Results**: 
   - First textarea shows raw OCR output
   - Second textarea shows LLM-corrected and improved text

## Project Structure

```
arabic-ocr-app/
├── app.py                 # Flask backend server
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── uploads/              # Temporary upload folder (auto-created)
├── static/
│   ├── style.css        # CSS styles
│   └── script.js        # Frontend JavaScript
└── templates/
    └── index.html       # Main HTML page
```

## Configuration

### Change Ollama Model

Edit `app.py`, line 46:
```python
json={"model": "mistral", ...}  # Change "mistral" to "llama3" or other model
```

### PaddleOCR Configuration

PaddleOCR is configured for Arabic (`lang='ar'`) in `app.py`. To use GPU (if available), change:
```python
ocr_engine = PaddleOCR(use_angle_cls=True, lang='ar', use_gpu=True)
```

## Troubleshooting

### PaddleOCR First Run
- On first use, PaddleOCR will download Arabic models (~100MB)
- Ensure you have internet connection for the initial download
- Models are cached locally for offline use afterward

### Ollama Connection Error
- Ensure Ollama is running: `ollama serve`
- Check if model is available: `ollama list`
- Verify Ollama is accessible at `http://localhost:11434`

### OCR Results Poor
- Use high-quality images with clear text
- Ensure good contrast between text and background
- Try different image formats (PNG, JPG)

## License

This project is provided as-is for educational and personal use.

