# Parkinson's Disease Multi-Modal Detector

A web-based application that detects early signs of Parkinson's Disease using voice recordings and handwriting samples. The system employs a multi-modal approach combining voice analysis and handwriting analysis for more accurate predictions.

## Features

- **Voice Analysis**: Extracts multiple vocal features including:
  - Jitter and Shimmer measurements
  - Harmonic to Noise Ratio (HNR)
  - Pitch Period Entropy (PPE)
  - Fundamental frequency features

- **Handwriting Analysis**: Uses deep learning to analyze handwriting patterns
  - Powered by a fine-tuned Swin Transformer model
  - Detects tremors and other PD-related patterns

- **Combined Analysis**: Late fusion technique to combine both modalities
  - Weighted prediction combining voice and handwriting results
  - Confidence scores for each prediction

- **User Management**:
  - User registration and login system
  - Secure session management
  - Password protection for user data

## Installation

1. Clone the repository:
```sh
git clone [repository-url]
cd parkison_disease_detector_v2
```

2. Create and activate a virtual environment:
```sh
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

3. Install required packages:
```sh
pip install -r requirements.txt
```

## Usage

1. Start the login server:
```sh
python login/app1.py
```

2. Start the main application server:
```sh
python app.py
```

3. Access the application:
   - Open a web browser and navigate to `http://localhost:3000`
   - Register a new account or login
   - You will be redirected to the main application

## Technical Details

### Voice Analysis
- Uses `librosa` for audio processing
- Extracts 22 different vocal features
- Implements real-time voice recording

### Handwriting Analysis
- Uses `transformers` library with Swin Transformer
- Pre-trained model: "gianlab/swin-tiny-patch4-window7-224-finetuned-parkinson-classification"

### Web Interface
- Built with Flask
- Responsive design using custom CSS
- Real-time analysis feedback

## Project Structure
```
├── app.py                 # Main Flask application
├── combined_detector.py   # Desktop GUI version
├── voice_extraction.py    # Voice analysis module
├── login/                 # User authentication
├── static/               # Static files (CSS, JS)
├── templates/            # HTML templates
└── data/                 # Training data
```

## Dependencies

- Flask
- librosa
- transformers
- sounddevice
- numpy
- pandas
- sqlite3
- tkinter (for desktop version)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.
