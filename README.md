# Freddie 2.0 - Expressive Robotic Face with AI Integration

An advanced robotic face system combining real-time emotion recognition, AI-powered conversations, and expressive servo-controlled animations. Freddie can detect emotions in speech using a custom-trained model, respond with appropriate facial expressions, track faces, and engage in natural conversations using Google Gemini.


## ✨ Features

- **Custom-Trained Speech Emotion Recognition** - Self-developed model detecting 7 emotions from voice in real-time
- **Expressive Facial Animations** - 13 servo channels for facial control + 3 for neck movement
- **AI-Powered Conversations** - Natural language understanding via Google Gemini API
- **Phoneme-based Lip Synchronization** - Realistic mouth movements synchronized with speech
- **Face Tracking** - MediaPipe-powered head tracking where Freddie's neck follows detected faces to maintain eye contact
- **Web Search Integration** - Can fetch real-time information during conversations
- **Sleep Mode** - Reduces activity by pausing face tracking and expressions while listening for wake phrases
- **Conversation Memory** - Maintains context across interactions


## 🎯 Custom Speech Emotion Recognition Model

**This project features a custom-trained Speech Emotion Recognition (SER) model** that I developed using multiple emotional speech datasets. The model is not pre-trained but was built through:

- **Data Collection**: Aggregated 9 different emotional speech datasets plus custom recordings by me
- **Feature Engineering**: Implemented 165-dimensional feature extraction including:
  - Mel-frequency spectrograms (40 bands)
  - MFCC coefficients with delta and delta-delta features
  - Pitch (F0) statistics and voice activity metrics
  - Energy and spectral characteristics
- **Model Training**: Trained a Support Vector Classifier (SVC) with RBF kernel
- **Fine-tuning**: Applied custom bias adjustments for real-world performance
- **Validation**: Achieved robust emotion detection across 7 emotion classes

The complete training pipeline is included in this repository (`scripts\SER_training_scripts\ser_train_baseline.py`), allowing you to:
- Retrain the model with your own data
- Modify the feature extraction process
- Experiment with different classifiers
- Customize for specific emotion sets


## 🛠 Hardware Requirements

- **Microcontroller**: Arduino (ESP32-S3)
- **Servos**: 16 total servos
  - **13 Facial Control Servos:**
    - 2x Eyebrow vertical movement
    - 2x Eyebrow angle
    - 2x Eye horizontal movement  
    - 2x Eye vertical movement
    - 2x Eyelids
    - 1x Jaw
    - 2x Mouth corners
  - **3 Neck Control Servos:**
    - 1x Neck yaw (left/right rotation)
    - 2x Neck pitch/roll (work together for up/down tilt and side tilt via differential control)
- **Camera**: USB webcam for face tracking
- **Microphone**: microphone for speech input
- **Computer**: Windows PC with Python 3.11+


## 📦 Installation

### 1. Clone the Repository
- cd freddie-2.0

### 2. Install Python Dependencies
- pip install -r requirements.txt

### 3. Download Piper TTS
- Download Piper from https://github.com/rhasspy/piper/releases
- Extract to a folder (e.g., C:\piper\)
- Download a voice model (recommended: en_US-norman-medium.onnx)

### 4. Set Environment Variables
- $env:GEMINI_API_KEY = "your-google-gemini-api-key"
- $env:OPENWEATHER_API_KEY = "your-openweather-api-key"  # Optional
- $env:SERPAPI_KEY = "your-serpapi-key"  # Optional for web search

### 5. Upload Arduino Code
- Open mask.ino in Arduino IDE
- Adjust pin assignments if needed
- Upload to your Arduino board

### 6. Configure File Paths
- PIPER_EXE = r"C:\path\to\piper.exe"
- VOICE_ONNX = r"C:\path\to\en_US-norman-medium.onnx"
- SERIAL_PORT = "COM2"  # Your Arduino port


## 🚀 Usage
- python freddie_integrated_clean.py --continuous # for continuous conversation


## Training Your Own SER Model
- This project includes the complete pipeline I used to train the emotion recognition model:

#### 1. Prepare your audio data
python ser_make_slices.py  # Segments audio into 3-second windows

#### 2. Train the model from scratch
python ser_train_baseline.py  # Extracts features and trains SVC

#### 3. Test your trained model in real-time
python ser_run_rt.py --mic

- The included model (ser_svc_auto.joblib) was trained by me using this exact pipeline


## 📁 Project Files

### Main Scripts
- `freddie_integrated_clean.py` - Main program that runs Freddie
- `phoneme_viseme_detector.py` - Lip sync and viseme detection module
- `web_search_functions.py` - Web search and API integration

### Custom SER Model Development
- `ser_train_baseline.py` - Script I used to train the emotion recognition model
- `ser_make_slices.py` - Preprocesses audio into 3-second training windows
- `ser_run_rt.py` - Test the SER model standalone with microphone input
- `ser_svc_auto.joblib` - The custom-trained SER model
- `ser_svc_auto_labels.json` - Emotion class labels for the model

### Hardware
- `mask/mask.ino` - Arduino code for controlling 16 servos

### Configuration
- `requirements.txt` - Python package dependencies
- `LICENSE` - MIT License
- `README.md` - Project documentation


## ⚙️ Configuration

### Emotion Detection Settings
SER_BIAS = {
    "neutral": 1.3,
    "happy": 0.5,
    "angry": 0.5,
    "sad": -0.25,
    "shocked": 2.0,
    "fear": -0.2,
    "disgust": 0.1
}

### Servo Channel Mapping
// Facial Servos (0-12)
CH_L_BROW_V = 0   # Left eyebrow vertical
CH_L_BROW_A = 1   # Left eyebrow angle
CH_R_BROW_V = 2   # Right eyebrow vertical
CH_R_BROW_A = 3   # Right eyebrow angle
CH_R_X = 4        # Right eye X
CH_R_Y = 5        # Right eye Y
CH_R_LID = 6      # Right eyelid
CH_L_X = 7        # Left eye X
CH_L_Y = 8        # Left eye Y
CH_L_LID = 9      # Left eyelid
CH_JAW = 10       # Jaw
CH_R_MOUTH = 11   # Right mouth corner
CH_L_MOUTH = 12   # Left mouth corner

// Neck Servos (13-15)
CH_NECK_YAW = 13   # Neck left/right rotation
CH_NECK_PITCH = 14 # Neck up/down tilt
CH_NECK_ROLL = 15  # Neck side tilt


## 🎥 Demo

Check out Freddie 2.0 in action:  
[![Freddie 2.0 Demo](https://img.shields.io/badge/YouTube-Watch%20Demo-red?logo=youtube)](https://youtu.be/OZfJSSJ9VQY)


## 📜 Credits & Attributions 

### Open-Source Libraries
- [NumPy](https://numpy.org/)  
- [SciPy](https://scipy.org/)  
- [scikit-learn](https://scikit-learn.org/)  
- [joblib](https://joblib.readthedocs.io/)  
- [librosa](https://librosa.org/)  
- [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/)  
- [sounddevice](https://python-sounddevice.readthedocs.io/)  
- [soundfile](https://pysoundfile.readthedocs.io/)  
- [OpenCV](https://opencv.org/)  
- [MediaPipe](https://developers.google.com/mediapipe)  
- [Piper TTS](https://github.com/rhasspy/piper)  
- [Whisper](https://github.com/openai/whisper)  
- [Google Generative AI](https://ai.google.dev/)  
- [pyserial](https://github.com/pyserial/pyserial)  
- [requests](https://requests.readthedocs.io/)  

### Speech Emotion Datasets Used for Training
- [ASVP-ESD](https://www.kaggle.com/datasets/dejolilandry/asvpesdspeech-nonspeech-emotional-utterances) — South China University of Technology  
- [CREMA-D](https://www.kaggle.com/datasets/ejlok1/cremad) — Cao, Cooper, Keutmann, Gur, & Verma (2014)  
- [ESD](https://www.kaggle.com/datasets/nguyenthanhlim/emotional-speech-dataset-esd) — Emotional Speech Dataset  
- [EmoV-DB](https://github.com/numediart/EmoV-DB) — Adigwe et al., 2018 ([arXiv:1806.09514](https://arxiv.org/abs/1806.09514))  
- [JL Corpus](https://www.kaggle.com/datasets/tli725/jl-corpus) — James, Tian, Watson (Interspeech 2018)  
- [MELD](https://www.kaggle.com/datasets/zaber666/meld-dataset/data) — Poria et al. (2018), derived from EmotionLines (Chen et al., 2018)  
- [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) — Livingstone & Russo (2018), CC BY-NC-SA 4.0  
- [SAVEE](https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee) — University of Surrey  
- [TESS](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) — Dupuis & Pichora-Fuller (2010), University of Toronto  

### 3D Models
- [Neck mechanism for animatronics and puppets](https://www.thingiverse.com/thing:4670841) — Hendrikx  
- [Animatronic Eyes (Compact, Arduino)](https://makerworld.com/en/models/1217039-animatronic-eyes-compact-with-arduino) — Morgan Manly  


## 🧾 License
MIT License — see the [LICENSE](./LICENSE) file for details.


## 👤 Author

**Rambod Taherian**  
- GitHub: [@rambodt](https://github.com/rambodt)  
- LinkedIn: [Rambod Taherian](https://www.linkedin.com/in/rambod-taherian)  
- Project Status: Active Development  
- Version: 2.0