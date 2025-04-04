import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math
import io
import os

# Set page configuration
st.set_page_config(
    page_title="Circle Diameter Viewer",
    layout="wide"
)

# Title and description
st.title("Circle Diameter Viewer")
st.markdown(
    "Upload an image to detect circles and view their diameters, radii, and other properties.")

# Sidebar for parameters
st.sidebar.header("Detection Parameters")

# Advanced options toggle
show_advanced = st.sidebar.checkbox("Show Advanced Options", value=False)

# Basic parameters
dp = st.sidebar.slider("DP (Accumulator Resolution)", 1.0, 5.0, 1.0, 0.1,
                       help="Inverse ratio of the accumulator resolution to the image resolution")
min_dist = st.sidebar.slider("Minimum Distance", 10, 100, 20,
                             help="Minimum distance between detected circle centers")
param1 = st.sidebar.slider("Edge Detection Threshold", 10, 300, 50,
                           help="Gradient value used for edge detection")
param2 = st.sidebar.slider("Circle Detection Threshold", 10, 100, 30,
                           help="Accumulator threshold - smaller values detect more circles")
min_radius = st.sidebar.slider("Minimum Radius", 0, 100, 10,
                               help="Minimum circle radius in pixels")
max_radius = st.sidebar.slider("Maximum Radius", 10, 300, 100,
                               help="Maximum circle radius in pixels")

# Advanced parameters
if show_advanced:
    st.sidebar.subheader("Pre-processing Options")
    apply_blur = st.sidebar.checkbox("Apply Gaussian Blur", value=True)
    blur_kernel = st.sidebar.slider("Blur Kernel Size", 3, 15, 5, 2,
                                    help="Size of the Gaussian blur kernel (must be odd)")
    apply_contrast = st.sidebar.checkbox("Enhance Contrast", value=False)
    contrast_clip = st.sidebar.slider("CLAHE Clip Limit", 1.0, 5.0, 2.0, 0.1,
                                      help="Clip limit for contrast enhancement")
    apply_canny = st.sidebar.checkbox(
        "Apply Canny Edge Detection", value=False)
    canny_low = st.sidebar.slider("Canny Low Threshold", 10, 200, 50,
                                  help="Low threshold for Canny edge detection")
    canny_high = st.sidebar.slider("Canny High Threshold", 50, 500, 150,
                                   help="High threshold for Canny edge detection")

# Function to detect circles


def detect_circles(image, dp, min_dist, param1, param2, min_radius, max_radius,
                   apply_blur=True, blur_kernel=5, apply_contrast=False, contrast_clip=2.0,
                   apply_canny=False, canny_low=50, canny_high=150):
    # Convert image to grayscale if it has multiple channels
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply pre-processing based on settings
    # Ensure blur kernel is odd
    if blur_kernel % 2 == 0:
        blur_kernel += 1

    # Apply Gaussian blur if selected
    if apply_blur:
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    # Apply contrast enhancement if selected
    if apply_contrast:
        clahe = cv2.createCLAHE(clipLimit=contrast_clip, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # Apply Canny edge detection if selected
    if apply_canny:
        gray = cv2.Canny(gray, canny_low, canny_high)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    # Optional: display pre-processed image
    return circles, gray

# Function to draw circles and numbers on the image


def draw_circles_with_numbers(image, circles, highlight_idx=None):
    # Create a copy of the image to draw on
    img_with_circles = image.copy()

    # If circles were found
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i, (x, y, r) in enumerate(circles[0, :]):
            # Set circle color - highlight the selected circle
            circle_color = (0, 255, 0)  # Default: green
            if highlight_idx is not None and i == highlight_idx:
                circle_color = (255, 0, 255)  # Magenta for highlighted circle

            # Draw the outer circle
            cv2.circle(img_with_circles, (x, y), r, circle_color, 2)
            # Draw the center of the circle
            cv2.circle(img_with_circles, (x, y), 2, (0, 0, 255), 3)
            # Draw number for identification - prevent overflow by ensuring positive values
            text_x = max(0, x - 10)
            text_y = max(20, y - 10)
            cv2.putText(img_with_circles, str(i+1), (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return img_with_circles

# Function to calculate circle properties


def calculate_circle_properties(radius_px, pixel_scale=1.0):
    radius = radius_px * pixel_scale
    diameter = 2 * radius
    area = math.pi * (radius ** 2)
    circumference = 2 * math.pi * radius

    return {
        "radius": radius,
        "diameter": diameter,
        "area": area,
        "circumference": circumference
    }

# Create example image with circles for demonstration


def create_example_image():
    # Create a blank image
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    img.fill(255)  # White background

    # Draw circles
    centers = [(150, 150, 50), (300, 200, 70), (200, 350, 40), (350, 350, 30)]
    for x, y, r in centers:
        cv2.circle(img, (x, y), r, (0, 0, 0), 2)

    return img

# Function to find the nearest circle to a clicked point


def find_nearest_circle(circles, click_x, click_y):
    if circles is None:
        return None

    circles = np.uint16(np.around(circles))
    min_dist = float('inf')
    nearest_idx = None

    for i, (x, y, r) in enumerate(circles[0, :]):
        # Check if the click is inside or near the circle
        dist = math.sqrt((x - click_x)**2 + (y - click_y)**2)

        # If the click is inside the circle or within 10 pixels of the edge
        if dist <= r + 10:
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

    return nearest_idx


# Initialize session state for selected circle
if 'selected_circle_idx' not in st.session_state:
    st.session_state.selected_circle_idx = None

# File uploader for image input
uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"])

# Process image
if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Convert to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
else:
    # Use example image
    use_example = st.checkbox("Use example image", value=True)
    if use_example:
        image = create_example_image()
        image_rgb = image.copy()
    else:
        st.markdown("""
        ### Please upload an image to begin.
        The application will detect circles in the image and allow you to view their measurements.
        
        #### Tips for best results:
        - Use images with clear, well-defined circles
        - Adjust the detection parameters in the sidebar if circles are not properly detected
        - For accurate real-world measurements, provide the correct scale (pixels per unit)
        """)
        st.stop()

# Detect circles with advanced parameters if enabled
if show_advanced:
    circles, processed_gray = detect_circles(
        image, dp, min_dist, param1, param2, min_radius, max_radius,
        apply_blur, blur_kernel, apply_contrast, contrast_clip,
        apply_canny, canny_low, canny_high
    )
else:
    circles, processed_gray = detect_circles(
        image, dp, min_dist, param1, param2, min_radius, max_radius
    )

# Create columns for layout
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Image with Detected Circles")

    # Draw circles on the image with highlighting
    image_with_circles = draw_circles_with_numbers(
        image_rgb, circles, st.session_state.selected_circle_idx)

    # Display image and capture clicks
    image_placeholder = st.empty()
    image_placeholder.image(image_with_circles, use_container_width=True)

    # Add click interaction if circles are found
    if circles is not None:
        st.markdown("**👆 Click on a circle to select it and view its details**")

        # Add a clickable image using streamlit's image_click capability
        if 'image_clicked' not in st.session_state:
            st.session_state.image_clicked = False

        # Get the actual displayed size to scale the coordinates
        img_height, img_width = image_rgb.shape[:2]

        # Create a clickable area
        click_col = st.columns(1)[0]
        with click_col:
            clicked = st.button("Reset Selection")
            if clicked:
                st.session_state.selected_circle_idx = None
                st.rerun()

        # For now we use a coordinate input as a workaround since Streamlit doesn't
        # directly support image clicks in the standard library
        click_cols = st.columns(2)
        with click_cols[0]:
            click_x = st.number_input("X coordinate", 0, img_width, img_width //
                                      2 if 'click_x' not in st.session_state else st.session_state.get('click_x', 0), key="click_x")
        with click_cols[1]:
            click_y = st.number_input("Y coordinate", 0, img_height, img_height //
                                      2 if 'click_y' not in st.session_state else st.session_state.get('click_y', 0), key="click_y")

        clicked = st.button("Select Circle at Coordinates")
        if clicked:
            nearest_idx = find_nearest_circle(circles, click_x, click_y)
            if nearest_idx is not None:
                st.session_state.selected_circle_idx = nearest_idx
                st.rerun()
            else:
                st.warning("No circle found at or near these coordinates")

    # Show pre-processed image if advanced options are enabled
    if show_advanced:
        show_processed = st.checkbox("Show Pre-processed Image", value=False)
        if show_processed:
            st.subheader("Pre-processed Image")
            st.image(processed_gray, use_container_width=True)

with col2:
    st.subheader("Circle Information")

    if circles is not None:
        circle_count = circles.shape[1]
        st.write(f"Number of circles detected: {circle_count}")

        # Create a selection dropdown for circles
        circle_numbers = [f"Circle {i+1}" for i in range(circle_count)]
        if circle_numbers:
            selected_circle = st.selectbox(
                "Select a circle to view details",
                circle_numbers,
                index=st.session_state.selected_circle_idx if st.session_state.selected_circle_idx is not None else 0
            )

            # Update the session state to match the dropdown
            idx = int(selected_circle.split()[-1]) - 1
            if idx != st.session_state.selected_circle_idx:
                st.session_state.selected_circle_idx = idx
                st.rerun()

            if selected_circle:
                # Get the selected circle data
                x, y, r = circles[0, idx]

                # Calculate properties
                properties = calculate_circle_properties(r)

                # Display properties
                st.markdown("#### Circle Properties")
                st.markdown(f"**Center:** ({x}, {y})")
                st.markdown(f"**Radius:** {r:.2f} pixels")
                st.markdown(f"**Diameter:** {2*r:.2f} pixels")
                st.markdown(f"**Area:** {math.pi * (r ** 2):.2f} sq. pixels")
                st.markdown(f"**Circumference:** {2 * math.pi * r:.2f} pixels")

                # Optional: Add scaling to real-world units
                st.markdown("---")
                st.markdown("#### Optional: Scale to Real-world Units")
                scale_factor = st.number_input("Pixels per unit (e.g., pixels/mm)",
                                               min_value=0.01, value=1.0, step=0.01)

                if scale_factor > 0:
                    real_radius = r / scale_factor
                    real_diameter = 2 * real_radius
                    real_area = math.pi * (real_radius ** 2)
                    real_circumference = 2 * math.pi * real_radius

                    st.markdown(f"**Radius:** {real_radius:.2f} units")
                    st.markdown(f"**Diameter:** {real_diameter:.2f} units")
                    st.markdown(f"**Area:** {real_area:.2f} sq. units")
                    st.markdown(
                        f"**Circumference:** {real_circumference:.2f} units")
    else:
        st.write("No circles detected. Try adjusting the parameters in the sidebar.")

        # Provide suggestions for improving detection
        st.markdown("""
        #### Suggestions to improve detection:
        1. Try decreasing the Circle Detection Threshold (param2)
        2. Adjust the Min/Max Radius values to match your circles
        3. Increase Edge Detection Threshold (param1) for clearer circles
        4. Enable Advanced Options for more fine-grained control
        """)
