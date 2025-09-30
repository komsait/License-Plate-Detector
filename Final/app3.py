import streamlit as st
import tempfile
import cv2
from ultralytics import YOLO
import easyocr
import numpy as np
import pandas as pd
from datetime import datetime

# Custom CSS for dark theme
st.markdown("""
<style>
    /* Dark theme variables */
    :root {
        --dark-bg: #1e1e1e;
        --dark-card: #2d2d2d;
        --dark-border: #404040;
        --light-text: #ffffff;
        --muted-text: #b0b0b0;
        --accent-color: #4a9eff;
        --success-color: #28a745;
        --info-color: #17a2b8;
    }
    
    /* Main app background */
    .stApp {
        background-color: var(--dark-bg);
        color: var(--light-text);
    }
    
    /* Title styling */
    .stTitle {
        color: var(--light-text);
    }
    
    /* Dark cards for dataframes and containers */
    .stDataFrame {
        background-color: var(--dark-card);
        border: 1px solid var(--dark-border);
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Success message styling */
    .stSuccess {
        background-color: rgba(40, 167, 69, 0.2);
        border: 1px solid var(--success-color);
        border-radius: 10px;
        color: #d4edda;
        padding: 1rem;
    }
    
    /* Info message styling */
    .stInfo {
        background-color: rgba(23, 162, 184, 0.2);
        border: 1px solid var(--info-color);
        border-radius: 10px;
        color: #d1ecf1;
        padding: 1rem;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: var(--dark-card);
    }
    
    .sidebar .stMarkdown {
        color: var(--light-text);
    }
    
    .sidebar .stMarkdown h1,
    .sidebar .stMarkdown h2,
    .sidebar .stMarkdown h3,
    .sidebar .stMarkdown h4,
    .sidebar .stMarkdown h5,
    .sidebar .stMarkdown h6 {
        color: var(--light-text);
    }
    
    /* Button styling */
    .stButton > button {
        background-color: var(--accent-color);
        color: var(--light-text);
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #3a8bdf;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background-color: var(--success-color);
        color: var(--light-text);
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    .stDownloadButton > button:hover {
        background-color: #218838;
    }
    
    /* File uploader styling */
    .stFileUploader {
        background-color: var(--dark-card);
        border: 2px dashed var(--accent-color);
        border-radius: 10px;
        padding: 2rem;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background-color: var(--accent-color);
    }
    
    /* Text color fixes */
    .stMarkdown {
        color: var(--light-text);
    }
    
    .stMarkdown p {
        color: var(--light-text);
    }
    
    .stMarkdown strong {
        color: var(--light-text);
    }
    
    /* Table styling */
    .dataframe {
        background-color: var(--dark-card);
        color: var(--light-text);
    }
    
    .dataframe th {
        background-color: var(--dark-border);
        color: var(--light-text);
    }
    
    .dataframe td {
        background-color: var(--dark-card);
        color: var(--light-text);
        border-color: var(--dark-border);
    }
    
    /* Container styling */
    .main .block-container {
        background-color: var(--dark-bg);
        color: var(--light-text);
    }
</style>
""", unsafe_allow_html=True)

# Load YOLOv8 model
model = YOLO("best1.pt")

# Load EasyOCR reader once
reader = easyocr.Reader(['en'])

# Initialize session state to store detected plates
if 'detected_plates' not in st.session_state:
    st.session_state.detected_plates = []
if 'plate_data' not in st.session_state:
    st.session_state.plate_data = []

def preprocess_license_plate(plate_crop):
    """
    Apply preprocessing steps to enhance license plate for OCR
    """
    if plate_crop.size == 0:
        return None
    
    # Step 1: Resize the image to make it larger (3x original size)
    height, width = plate_crop.shape[:2]
    plate_crop = cv2.resize(plate_crop, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
    
    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    
    # Step 3: Apply Gaussian blur to smooth the image
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Step 4: Apply thresholding (Otsu's method)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Step 5: Dilate to make characters more connected
    kernel = np.ones((2, 1), np.uint8)  # Vertical kernel to connect broken characters
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    # Step 6: Find contours and filter for character-like regions
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on aspect ratio and area
    filtered_contours = []
    total_height = dilated.shape[0]
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter parameters
        min_height_ratio = 1/6  
        max_height_ratio = 3/4  
        min_aspect_ratio = 0.2  
        max_aspect_ratio = 1.5  
        min_area = 50           
        
        height_ratio = h / total_height
        aspect_ratio = w / h if h > 0 else 0
        area = w * h
        
        if (height_ratio >= min_height_ratio and 
            height_ratio <= max_height_ratio and
            aspect_ratio >= min_aspect_ratio and 
            aspect_ratio <= max_aspect_ratio and
            area >= min_area):
            filtered_contours.append((x, y, w, h))
    
    # Sort contours left to right
    filtered_contours.sort(key=lambda rect: rect[0])
    
    # Step 7: Create a clean image with only the filtered characters
    # Convert black text on white background
    final_image = cv2.bitwise_not(dilated)
    
    # Step 8: Apply median blur to reduce noise
    final_image = cv2.medianBlur(final_image, 3)
    
    return final_image

def clean_plate_text(text):
    """
    Clean and standardize license plate text
    """
    # Remove special characters and keep only alphanumeric
    cleaned = ''.join(c for c in text if c.isalnum() or c.isspace())
    # Convert to uppercase
    cleaned = cleaned.upper()
    # Remove extra spaces
    cleaned = ' '.join(cleaned.split())
    return cleaned

def is_similar_plate(new_plate, existing_plates, similarity_threshold=0.8):
    """
    Check if a plate is similar to existing plates using fuzzy matching
    """
    from difflib import SequenceMatcher
    
    new_plate_clean = clean_plate_text(new_plate)
    
    for existing_plate in existing_plates:
        existing_clean = clean_plate_text(existing_plate)
        similarity = SequenceMatcher(None, new_plate_clean, existing_clean).ratio()
        if similarity >= similarity_threshold:
            return True
    return False

st.title("License Plate Detection & OCR with YOLOv8")
st.markdown("Upload a video and detect **license plates** with enhanced OCR text.")

# File uploader
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

# Display detected plates table
if st.session_state.plate_data:
    st.subheader("📋 Detected License Plates")
    
    # Create DataFrame
    df = pd.DataFrame(st.session_state.plate_data)
    
    # Display table
    st.dataframe(df, use_container_width=True)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"detected_plates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    # Clear button
    if st.button("🗑️ Clear All Records", use_container_width=True):
        st.session_state.detected_plates = []
        st.session_state.plate_data = []
        st.rerun()

if uploaded_file is not None:
    # Save uploaded video to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    # Placeholder for frames
    stframe = st.empty()
    
    # Statistics
    total_detections = 0
    successful_ocr = 0
    new_plates_found = 0

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model(frame, conf=0.4)
        annotated_frame = results[0].plot()

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_crop = frame[y1:y2, x1:x2]

                if plate_crop.size == 0:
                    continue

                total_detections += 1
                
                # Apply preprocessing
                preprocessed_plate = preprocess_license_plate(plate_crop)
                
                if preprocessed_plate is not None:
                    # OCR on preprocessed plate
                    text_results = reader.readtext(preprocessed_plate)
                    
                    detected_texts = []
                    confidence_scores = []
                    
                    for t in text_results:
                        bbox, text, confidence = t
                        if confidence > 0.3:  # Confidence threshold
                            detected_texts.append(text)
                            confidence_scores.append(confidence)
                    
                    if detected_texts:
                        successful_ocr += 1
                        final_text = " ".join(detected_texts)
                        avg_confidence = sum(confidence_scores) / len(confidence_scores)
                        
                        # Clean the detected text
                        cleaned_text = clean_plate_text(final_text)
                        
                        # Check if this is a new plate
                        if cleaned_text and not is_similar_plate(cleaned_text, st.session_state.detected_plates):
                            st.session_state.detected_plates.append(cleaned_text)
                            
                            # Add to plate data with timestamp
                            plate_info = {
                                "License Plate": cleaned_text,
                                "Confidence": f"{avg_confidence:.2f}",
                                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Raw Text": final_text
                            }
                            st.session_state.plate_data.append(plate_info)
                            new_plates_found += 1
                            
                            # Show success message for new plate
                            st.sidebar.success(f"New plate detected: {cleaned_text}")
                        
                        # Draw text on annotated frame
                        label = cleaned_text
                        font = cv2.FONT_HERSHEY_COMPLEX
                        scale = 0.9
                        thickness = 2

                        (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)

                        x, y = x1, y1 - 10
                        cv2.rectangle(
                            annotated_frame, 
                            (x, y - text_h - baseline),
                            (x + text_w, y + baseline),
                            (0, 0, 0),
                            -1
                        )
                        cv2.putText(
                            annotated_frame, 
                            label, 
                            (x, y), 
                            font, 
                            scale, 
                            (255, 255, 255),
                            thickness
                        )

        # Convert BGR → RGB for Streamlit
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        stframe.image(annotated_frame, channels="RGB", use_container_width=True)

    cap.release()
    
    # Display final statistics
    st.success(f"""
    Processing complete!
    - Total detections: {total_detections}
    - Successful OCR: {successful_ocr}
    - New plates found: {new_plates_found}
    - Total unique plates: {len(st.session_state.detected_plates)}
    """)

# Display instructions when no video is uploaded
else:
    st.info("👆 Please upload a video file to start license plate detection.")
    
    st.sidebar.subheader("How it works:")
    st.sidebar.markdown("""
    1. Upload a video file
    2. YOLOv8 detects license plates
    3. Preprocessing enhances image quality
    4. EasyOCR extracts text
    5. Unique plates are saved in the table
    6. Download results as CSV
    """)