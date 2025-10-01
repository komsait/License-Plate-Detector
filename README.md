# License Plate Detection & OCR System

AI-powered license plate detection and text recognition system using YOLOv8 and EasyOCR. This project provides real-time license plate detection from video files with high accuracy and a modern dark-themed web interface.



##  Features

- ** Model Used**: pretrained model for license detection
- ** Advanced OCR**: EasyOCR integration for text extraction with preprocessing
- ** Real-time Processing**: Live video analysis with frame-by-frame detection
- ** Data Export**: CSV export functionality for detected license plates
- ** Duplicate Detection**: filtering to avoid duplicate detections



### Installation

#### Local Development
1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**
   ```bash
   streamlit run Final/app3.py
   ```

3. **Open your browser**
   Navigate to `http://localhost:8501`

#### Cloud Deployment (Streamlit Cloud, Heroku, etc.)
1. **Use deployment-specific requirements**
   ```bash
   pip install -r requirements-deploy.txt
   ```


## 📋 Requirements

```
streamlit>=1.28.0
ultralytics>=8.0.0
easyocr>=1.7.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
torch>=2.0.0
torchvision>=0.15.0
```

##  How It Works

### 1. **Detection steps**
```
Video Input → YOLOv8 Detection → License Plate Cropping → Preprocessing → OCR → Text Extraction
```

### 2. **Preprocessing Steps**
- **Resize**: 3x upscaling for better OCR accuracy
- **Grayscale Conversion**: Reduces noise and improves text clarity
- **Gaussian Blur**: Smooths the image for better thresholding
- **Otsu Thresholding**: Automatic binary conversion
- **Morphological Operations**: Connects broken characters
- **Contour Filtering**: Removes noise and focuses on text regions

### 3. **OCR Enhancement**
- **EasyOCR Integration**: Multi-language text recognition
- **Confidence Filtering**: Only accepts high-confidence detections (>30%)
- **Text Cleaning**: Standardizes format and removes special characters
- **Duplicate Prevention**: Fuzzy matching to avoid repeated entries



1. **Upload Video**: Drag and drop or select a video file (MP4, AVI, MOV)
2. **Start Processing**: Click the process button to begin detection
3. **View Results**: Watch real-time detection with bounding boxes and text overlays
4. **Export Data**: Download detected license plates as CSV


## Advanced Features

### Smart Duplicate Detection
- Uses fuzzy string matching to identify similar license plates
- Configurable similarity threshold
- Prevents database pollution with repeated entries

### Real-time Processing
- Frame-by-frame analysis with configurable sampling rate
- Progress tracking with visual progress bar
- Live statistics updates

### Data Management
- Session-based data storage
- Export functionality with timestamps
- Clear data options for fresh starts

