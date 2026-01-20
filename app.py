"""
Flask application for sentiment analysis using DistilBERT model.
Provides a REST API endpoint for text sentiment prediction.
"""

from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None

# Model configuration
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    """Load the sentiment analysis model and tokenizer."""
    global model, tokenizer
    
    try:
        logger.info(f"Loading model: {MODEL_NAME}")
        logger.info(f"Using device: {DEVICE}")
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.to(DEVICE)
        model.eval()  # Set to evaluation mode
        
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": DEVICE
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict sentiment for input text.
    
    Expected JSON payload:
    {
        "text": "Your text here"
    }
    
    Returns:
    {
        "sentiment": "POSITIVE" or "NEGATIVE",
        "score": 0.95,
        "text": "Your text here"
    }
    """
    if model is None or tokenizer is None:
        return jsonify({
            "error": "Model not loaded"
        }), 503
    
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "No JSON data provided"
            }), 400
        
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                "error": "Text field is required and cannot be empty"
            }), 400
        
        # Tokenize and predict
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Move inputs to the same device as model
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Extract results
        scores = predictions[0].cpu().numpy()
        positive_score = float(scores[1])
        negative_score = float(scores[0])
        
        # Determine sentiment
        sentiment = "POSITIVE" if positive_score > negative_score else "NEGATIVE"
        confidence_score = max(positive_score, negative_score)
        
        result = {
            "sentiment": sentiment,
            "score": round(confidence_score, 4),
            "positive_score": round(positive_score, 4),
            "negative_score": round(negative_score, 4),
            "text": text
        }
        
        logger.info(f"Prediction: {sentiment} (score: {confidence_score:.4f})")
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500

@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API information."""
    return jsonify({
        "service": "Sentiment Analysis API",
        "model": MODEL_NAME,
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)"
        },
        "example_request": {
            "text": "I love this product!"
        }
    }), 200

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run Flask development server (not for production)
    app.run(host='0.0.0.0', port=5000, debug=False)
else:
    # Load model when using Gunicorn
    load_model()
