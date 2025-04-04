# Circle Diameter Viewer

A Streamlit application that detects circles in images and displays their properties including diameter, radius, and area.

## Features

- Upload any image to detect circles
- Interactive parameters for circle detection tuning
- View detailed measurements for each circle
- Select individual circles to see their properties
- Optional scaling to convert pixel measurements to real-world units

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
