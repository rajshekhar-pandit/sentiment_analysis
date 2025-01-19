# Emotion Detection App 😊

## Overview
An interactive web application built with Streamlit that analyzes emotions in text using a pre-trained deep learning model. The app detects various emotions like joy, sadness, anger, and more from input text and visualizes the results.

## Features
- Single text emotion analysis with real-time results
- Batch processing for multiple texts
- Interactive visualization of emotion scores
- Export analysis results to JSON format
- Clean and intuitive user interface

## Technologies Used
- Python 3.12
- Streamlit 1.31.1
- Transformers 4.37.2 (Hugging Face)
- PyTorch 2.2.0
- Matplotlib 3.8.2
- DistilRoBERTa-based emotion detection model

## Installation
1. Clone the repository:
```bash
[git clone [repository-url]](https://github.com/rajshekhar-pandit/sentiment_analysis/)
cd app
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate # Linux/Mac
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage
1. Start the application:
```bash
streamlit run app.py
```

2. Choose analysis mode from the sidebar:
   - Single Text Analysis
   - Batch Text Analysis
   - About

3. For single text analysis:
   - Enter text in the input area
   - Click "Analyze" to see results
   - View emotion scores and visualization

4. For batch analysis:
   - Enter multiple texts separated by semicolons
   - Click "Analyze Batch"
   - Export results to JSON if needed

## Model Information
The app uses the `emotion-english-distilroberta-base` model by J. Hartmann, which is optimized for emotion detection in English text. The model can identify various emotional states with high accuracy.

## Limitations
- Works best with English text
- May not be accurate for very short texts or single words
- Requires internet connection for first-time model download

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
