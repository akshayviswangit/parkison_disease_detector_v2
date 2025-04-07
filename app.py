from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from voice_extraction import extract_features, assess_parkinsons
from transformers import pipeline
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the handwriting model
handwriting_model = pipeline("image-classification", 
                           "gianlab/swin-tiny-patch4-window7-224-finetuned-parkinson-classification")

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    voice_file = None
    handwriting_file = None
    
    # Handle voice file
    if 'voice' in request.files:
        file = request.files['voice']
        if file and allowed_file(file.filename, {'wav'}):
            filename = secure_filename(f"{uuid.uuid4()}.wav")
            voice_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(voice_path)
            voice_file = voice_path

    # Handle handwriting file
    if 'handwriting' in request.files:
        file = request.files['handwriting']
        if file and allowed_file(file.filename, {'png', 'jpg', 'jpeg'}):
            filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
            handwriting_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(handwriting_path)
            handwriting_file = handwriting_path

    results = {
        'voice_analysis': analyze_voice(voice_file) if voice_file else None,
        'handwriting_analysis': analyze_handwriting(handwriting_file) if handwriting_file else None
    }

    # Perform late fusion if both analyses are available
    if results['voice_analysis'] and results['handwriting_analysis']:
        results['combined'] = late_fusion(
            results['voice_analysis'],
            results['handwriting_analysis']
        )
    
    # Clean up uploaded files
    if voice_file and os.path.exists(voice_file):
        os.remove(voice_file)
    if handwriting_file and os.path.exists(handwriting_file):
        os.remove(handwriting_file)

    return jsonify(results)

def analyze_voice(file_path):
    features = extract_features(file_path)
    prediction, risk_factors, risk_details = assess_parkinsons(features)
    return {
        'prediction': prediction,
        'risk_factors': risk_factors,
        'risk_details': risk_details
    }

def analyze_handwriting(file_path):
    result = handwriting_model(file_path)
    if result and isinstance(result, list) and len(result) > 0:
        prediction = result[0]
        has_parkinsons = 'parkinson' in prediction['label'].lower()
        return {
            'prediction': "Yes" if has_parkinsons else "No",
            'confidence': float(prediction['score'])
        }
    return {
        'prediction': "No",
        'confidence': 0.0
    }

def late_fusion(voice_result, handwriting_result):
    voice_score = 1 if voice_result['prediction'] == "Yes" else 0
    hw_score = 1 if handwriting_result['prediction'] == "Yes" else 0
    
    voice_weight = 0.5
    hw_weight = 0.5
    
    combined_score = (voice_score * voice_weight + hw_score * hw_weight)
    final_prediction = "Yes" if combined_score >= 0.5 else "No"
    confidence = combined_score if final_prediction == "Yes" else (1 - combined_score)
    
    return {
        'prediction': final_prediction,
        'confidence': confidence
    }

if __name__ == '__main__':
    app.run(debug=True)