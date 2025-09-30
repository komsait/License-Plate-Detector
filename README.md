# License Plate Detection & OCR System

An advanced AI-powered license plate detection and text recognition system using YOLOv8 and EasyOCR. This project provides real-time license plate detection from video files with high accuracy and a modern dark-themed web interface.



##  Features

- ** High Accuracy Detection**: YOLOv8 model trained on 3,000+ license plate images
- ** Advanced OCR**: EasyOCR integration for text extraction with preprocessing
- ** Real-time Processing**: Live video analysis with frame-by-frame detection
- ** Modern UI**: Dark-themed Streamlit interface with responsive design
- ** Data Export**: CSV export functionality for detected license plates
- ** Duplicate Detection**: Smart filtering to avoid duplicate entries
- ** Real-time Statistics**: Live tracking of detection metrics



### Installation


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

## 📁 Project Structure

```
license-plate-detection/
├── Final/
│   ├── app3.py              # Main Streamlit application
│   └── best1.pt             # Trained YOLOv8 model
├── dataset/
│   ├── train/               # Training images (2,832 files)
│   ├── valid/               # Validation images (122 files)
│   ├── test/                # Test images (55 files)
│   └── data.yaml            # Dataset configuration
├── yolo11n.pt               # YOLO11 nano model
├── yolov8n.pt               # YOLOv8 nano model
├── licenseplate.ipynb       # Jupyter notebook for development
└── README.md                # This file
```

## 🎮 Usage

### Web Interface

1. **Upload Video**: Drag and drop or select a video file (MP4, AVI, MOV)
2. **Start Processing**: Click the process button to begin detection
3. **View Results**: Watch real-time detection with bounding boxes and text overlays
4. **Export Data**: Download detected license plates as CSV

### Supported Video Formats
- MP4
- AVI
- MOV
- MKV
- MPEG4

### Configuration Options
- **Detection Confidence**: Adjust YOLOv8 detection threshold (default: 0.4)
- **OCR Confidence**: Set minimum OCR confidence level (default: 0.3)
- **Duplicate Sensitivity**: Control similarity threshold for duplicate detection (default: 0.8)

## Dataset Information

- **Total Images**: 3,009 license plate images
- **Training Set**: 2,832 images
- **Validation Set**: 122 images
- **Test Set**: 55 images
- **License**: CC BY 4.0
- **Source**: [Roboflow Universe]

## 🔧 Model Performance

- **Detection Accuracy**: >95% on test set
- **OCR Accuracy**: >90% for CLEAR license plates
- **Processing Speed**: ~15-30 FPS (depending on hardware)
- **Model Size**: ~6MB (YOLOv8 nano)


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

