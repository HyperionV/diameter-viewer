# Circle Diameter Viewer

A Streamlit application that detects circles in images and displays their properties including diameter, radius, and area.

## Features

- Upload any image to detect circles
- Interactive parameters for circle detection tuning
- View detailed measurements for each circle
- Select individual circles to see their properties
- Optional scaling to convert pixel measurements to real-world units
- **Multilingual support** (English and Vietnamese)
- **Quick guide for beginners** with simple, non-technical instructions

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:

```bash
streamlit run app/app.py
```

2. Open your web browser to the URL provided by Streamlit (typically http://localhost:8501)

3. Upload an image containing circles

4. Adjust the detection parameters in the sidebar if needed

5. Select a circle from the dropdown to view its properties

## For Non-Technical Users

This application is designed to be accessible for everyone, including non-IT users! Here's how to use it:

1. **Getting Started**: Click the "Use example image" checkbox to see the app in action with a sample image
2. **Uploading Images**: Use the file uploader at the top to add your own pictures
3. **Finding Circles**: Circles will be automatically detected and numbered on your image
4. **Viewing Measurements**: Click on any numbered circle to see its size details
5. **Real-world Measurements**: If you know the scale of your image (e.g., pixels per mm), use the scale option to convert to real units
6. **Language Option**: Switch between English and Vietnamese using the dropdown in the sidebar

## Parameter Tuning

The sidebar provides several parameters to tune the circle detection:

- **DP (Accumulator Resolution)**: Controls the resolution of the accumulator. Higher values reduce detection precision but increase speed.
- **Minimum Distance**: Minimum distance between detected circle centers. Increase to avoid multiple detections of the same circle.
- **Param1**: Parameter controlling edge detection sensitivity.
- **Param2**: Accumulator threshold - lower values detect more circles, potentially including false positives.
- **Minimum/Maximum Radius**: Limits for the size of circles to detect.

## Requirements

- Python 3.7+
- Streamlit
- OpenCV
- NumPy
- PIL (Pillow)

## License

MIT
