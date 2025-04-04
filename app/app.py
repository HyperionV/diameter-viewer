import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math
import io
import os

# Dictionary for multilingual support
translations = {
    "en": {
        "page_title": "Circle Diameter Viewer",
        "app_title": "Circle Diameter Viewer",
        "app_description": "Upload an image to detect circles and view their diameters, radii, and other properties.",
        "detection_params": "Detection Parameters",
        "advanced_options": "Show Advanced Options",
        "dp_title": "DP (Accumulator Resolution)",
        "dp_help": "Inverse ratio of the accumulator resolution to the image resolution",
        "min_dist_title": "Minimum Distance",
        "min_dist_help": "Minimum distance between detected circle centers",
        "param1_title": "Edge Detection Threshold",
        "param1_help": "Gradient value used for edge detection",
        "param2_title": "Circle Detection Threshold",
        "param2_help": "Accumulator threshold - smaller values detect more circles",
        "min_radius_title": "Minimum Radius",
        "min_radius_help": "Minimum circle radius in pixels",
        "max_radius_title": "Maximum Radius",
        "max_radius_help": "Maximum circle radius in pixels",
        "preprocessing_options": "Pre-processing Options",
        "apply_blur": "Apply Gaussian Blur",
        "blur_kernel": "Blur Kernel Size",
        "blur_kernel_help": "Size of the Gaussian blur kernel (must be odd)",
        "enhance_contrast": "Enhance Contrast",
        "clahe_clip": "CLAHE Clip Limit",
        "clahe_clip_help": "Clip limit for contrast enhancement",
        "apply_canny": "Apply Canny Edge Detection",
        "canny_low": "Canny Low Threshold",
        "canny_low_help": "Low threshold for Canny edge detection",
        "canny_high": "Canny High Threshold",
        "canny_high_help": "High threshold for Canny edge detection",
        "upload_image": "Upload an image",
        "use_example": "Use example image",
        "upload_prompt": "### Please upload an image to begin.\nThe application will detect circles in the image and allow you to view their measurements.\n\n#### Tips for best results:\n- Use images with clear, well-defined circles\n- Adjust the detection parameters in the sidebar if circles are not properly detected\n- For accurate real-world measurements, provide the correct scale (pixels per unit)",
        "detected_circles": "Image with Detected Circles",
        "click_prompt": "👆 Click on a circle to select it and view its details",
        "reset_selection": "Reset Selection",
        "x_coordinate": "X coordinate",
        "y_coordinate": "Y coordinate",
        "select_circle_coords": "Select Circle at Coordinates",
        "no_circle_warning": "No circle found at or near these coordinates",
        "show_preprocessed": "Show Pre-processed Image",
        "preprocessed_image": "Pre-processed Image",
        "circle_info": "Circle Information",
        "num_circles": "Number of circles detected: ",
        "select_circle": "Select a circle to view details",
        "circle_properties": "#### Circle Properties",
        "center": "**Center:** ",
        "radius": "**Radius:** ",
        "diameter": "**Diameter:** ",
        "area": "**Area:** ",
        "circumference": "**Circumference:** ",
        "pixels": " pixels",
        "sq_pixels": " sq. pixels",
        "scale_section": "#### Optional: Scale to Real-world Units",
        "scale_factor": "Pixels per unit (e.g., pixels/mm)",
        "units": " units",
        "sq_units": " sq. units",
        "no_circles": "No circles detected. Try adjusting the parameters in the sidebar.",
        "suggestions": "#### Suggestions to improve detection:\n1. Try decreasing the Circle Detection Threshold (param2)\n2. Adjust the Min/Max Radius values to match your circles\n3. Increase Edge Detection Threshold (param1) for clearer circles\n4. Enable Advanced Options for more fine-grained control",
        "quick_guide": "## Quick Guide for New Users",
        "quick_guide_content": """
        ### Welcome to the Circle Diameter Viewer! 👋

        This app helps you measure circles in your images. Here's how to use it:

        #### 🔍 Basic Steps:
        1. **Upload a picture** using the file uploader above
        2. **Circles will be automatically detected** and numbered in the image
        3. **Select a circle** either by:
           - Clicking on the "Select Circle at Coordinates" button after entering X and Y values
           - Using the dropdown menu on the right side
        4. **View measurements** including radius, diameter, and area of the selected circle

        #### 💡 Need Better Results?
        - If circles aren't detected correctly, adjust the "Circle Detection Threshold" in the left sidebar (lower values find more circles)
        - For real-world measurements (like centimeters), use the "Optional: Scale to Real-world Units" section
        
        #### 🛠️ Tips for Success:
        - Use clear images with well-defined circles
        - Make sure there's good contrast between circles and background
        - Try the example image first to see how the app works
        """
    },
    "vi": {
        "page_title": "Công Cụ Đo Đường Kính Hình Tròn",
        "app_title": "Công Cụ Đo Đường Kính Hình Tròn",
        "app_description": "Tải lên một hình ảnh để phát hiện các hình tròn và xem đường kính, bán kính và các thuộc tính khác.",
        "detection_params": "Thông Số Phát Hiện",
        "advanced_options": "Hiển Thị Tùy Chọn Nâng Cao",
        "dp_title": "DP (Độ Phân Giải Bộ Tích Lũy)",
        "dp_help": "Tỷ lệ nghịch của độ phân giải bộ tích lũy so với độ phân giải hình ảnh",
        "min_dist_title": "Khoảng Cách Tối Thiểu",
        "min_dist_help": "Khoảng cách tối thiểu giữa các tâm hình tròn được phát hiện",
        "param1_title": "Ngưỡng Phát Hiện Cạnh",
        "param1_help": "Giá trị gradient được sử dụng cho phát hiện cạnh",
        "param2_title": "Ngưỡng Phát Hiện Hình Tròn",
        "param2_help": "Ngưỡng bộ tích lũy - giá trị nhỏ hơn phát hiện nhiều hình tròn hơn",
        "min_radius_title": "Bán Kính Tối Thiểu",
        "min_radius_help": "Bán kính tối thiểu của hình tròn tính bằng pixel",
        "max_radius_title": "Bán Kính Tối Đa",
        "max_radius_help": "Bán kính tối đa của hình tròn tính bằng pixel",
        "preprocessing_options": "Tùy Chọn Tiền Xử Lý",
        "apply_blur": "Áp Dụng Làm Mờ Gaussian",
        "blur_kernel": "Kích Thước Kernel Làm Mờ",
        "blur_kernel_help": "Kích thước của kernel làm mờ Gaussian (phải là số lẻ)",
        "enhance_contrast": "Tăng Cường Độ Tương Phản",
        "clahe_clip": "Giới Hạn Clip CLAHE",
        "clahe_clip_help": "Giới hạn clip cho việc tăng cường độ tương phản",
        "apply_canny": "Áp Dụng Phát Hiện Cạnh Canny",
        "canny_low": "Ngưỡng Thấp Canny",
        "canny_low_help": "Ngưỡng thấp cho phát hiện cạnh Canny",
        "canny_high": "Ngưỡng Cao Canny",
        "canny_high_help": "Ngưỡng cao cho phát hiện cạnh Canny",
        "upload_image": "Tải lên một hình ảnh",
        "use_example": "Sử dụng hình ảnh mẫu",
        "upload_prompt": "### Vui lòng tải lên một hình ảnh để bắt đầu.\nỨng dụng sẽ phát hiện các hình tròn trong hình ảnh và cho phép bạn xem các kích thước của chúng.\n\n#### Mẹo để có kết quả tốt nhất:\n- Sử dụng hình ảnh có hình tròn rõ ràng\n- Điều chỉnh các thông số phát hiện ở thanh bên nếu hình tròn không được phát hiện đúng\n- Để đo lường chính xác trong thực tế, cung cấp tỷ lệ đúng (pixel trên đơn vị)",
        "detected_circles": "Hình Ảnh với Các Hình Tròn Được Phát Hiện",
        "click_prompt": "👆 Nhấp vào một hình tròn để chọn và xem chi tiết",
        "reset_selection": "Đặt Lại Lựa Chọn",
        "x_coordinate": "Tọa độ X",
        "y_coordinate": "Tọa độ Y",
        "select_circle_coords": "Chọn Hình Tròn tại Tọa Độ",
        "no_circle_warning": "Không tìm thấy hình tròn tại hoặc gần tọa độ này",
        "show_preprocessed": "Hiển Thị Hình Ảnh Đã Tiền Xử Lý",
        "preprocessed_image": "Hình Ảnh Đã Tiền Xử Lý",
        "circle_info": "Thông Tin Hình Tròn",
        "num_circles": "Số lượng hình tròn được phát hiện: ",
        "select_circle": "Chọn một hình tròn để xem chi tiết",
        "circle_properties": "#### Thuộc Tính Hình Tròn",
        "center": "**Tâm:** ",
        "radius": "**Bán kính:** ",
        "diameter": "**Đường kính:** ",
        "area": "**Diện tích:** ",
        "circumference": "**Chu vi:** ",
        "pixels": " pixel",
        "sq_pixels": " pixel vuông",
        "scale_section": "#### Tùy chọn: Chuyển đổi sang đơn vị thực tế",
        "scale_factor": "Pixel trên đơn vị (ví dụ: pixel/mm)",
        "units": " đơn vị",
        "sq_units": " đơn vị vuông",
        "no_circles": "Không phát hiện hình tròn nào. Thử điều chỉnh các thông số ở thanh bên.",
        "suggestions": "#### Đề xuất để cải thiện phát hiện:\n1. Thử giảm Ngưỡng Phát Hiện Hình Tròn (param2)\n2. Điều chỉnh giá trị Bán Kính Tối Thiểu/Tối Đa để phù hợp với hình tròn của bạn\n3. Tăng Ngưỡng Phát Hiện Cạnh (param1) để có hình tròn rõ ràng hơn\n4. Bật Tùy Chọn Nâng Cao để kiểm soát chi tiết hơn",
        "quick_guide": "## Hướng Dẫn Nhanh",
        "quick_guide_content": """
        Ứng dụng này giúp bạn đo lường các hình tròn trong hình ảnh của bạn. Đây là cách sử dụng:

        #### 🔍 Các Bước Cơ Bản:
        1. **Tải lên một hình ảnh** sử dụng công cụ tải lên ở trên
        2. **Các hình tròn sẽ được tự động phát hiện** và đánh số trong hình ảnh
        3. **Chọn một hình tròn** bằng cách:
           - Nhấp vào nút "Chọn Hình Tròn tại Tọa Độ" sau khi nhập giá trị X và Y
           - Sử dụng menu thả xuống ở bên phải
        4. **Xem các kích thước** bao gồm bán kính, đường kính và diện tích của hình tròn đã chọn

        #### 💡 Cần Kết Quả Tốt Hơn?
        - Nếu hình tròn không được phát hiện chính xác, điều chỉnh "Ngưỡng Phát Hiện Hình Tròn" ở thanh bên trái (giá trị thấp hơn sẽ tìm thấy nhiều hình tròn hơn)
        - Để đo lường thực tế (như centimét), sử dụng phần "Tùy chọn: Chuyển đổi sang đơn vị thực tế"
        
        #### 🛠️ Mẹo để Thành Công:
        - Sử dụng hình ảnh rõ ràng với hình tròn được định nghĩa tốt
        - Đảm bảo có độ tương phản tốt giữa hình tròn và nền
        - Thử hình ảnh mẫu trước để xem cách ứng dụng hoạt động
        """
    }
}

# Set page configuration
st.set_page_config(
    page_title="Circle Diameter Viewer",
    layout="wide"
)

# Language selection
language = st.sidebar.selectbox(
    "Language / Ngôn ngữ",
    ["English", "Tiếng Việt"],
    index=0
)

# Set language code
lang_code = "en" if language == "English" else "vi"
t = translations[lang_code]  # Get translations for selected language

# Title and description
st.title(t["app_title"])
st.markdown(t["app_description"])

# Quick guide section (new addition)
with st.expander(t["quick_guide"], expanded=False):
    st.markdown(t["quick_guide_content"])

# Sidebar for parameters
st.sidebar.header(t["detection_params"])

# Advanced options toggle
show_advanced = st.sidebar.checkbox(t["advanced_options"], value=False)

# Basic parameters
dp = st.sidebar.slider(t["dp_title"], 1.0, 5.0, 1.0, 0.1,
                       help=t["dp_help"])
min_dist = st.sidebar.slider(t["min_dist_title"], 10, 100, 20,
                             help=t["min_dist_help"])
param1 = st.sidebar.slider(t["param1_title"], 10, 300, 50,
                           help=t["param1_help"])
param2 = st.sidebar.slider(t["param2_title"], 10, 100, 30,
                           help=t["param2_help"])
min_radius = st.sidebar.slider(t["min_radius_title"], 0, 100, 10,
                               help=t["min_radius_help"])
max_radius = st.sidebar.slider(t["max_radius_title"], 10, 300, 100,
                               help=t["max_radius_help"])

# Advanced parameters
if show_advanced:
    st.sidebar.subheader(t["preprocessing_options"])
    apply_blur = st.sidebar.checkbox(t["apply_blur"], value=True)
    blur_kernel = st.sidebar.slider(t["blur_kernel"], 3, 15, 5, 2,
                                    help=t["blur_kernel_help"])
    apply_contrast = st.sidebar.checkbox(t["enhance_contrast"], value=False)
    contrast_clip = st.sidebar.slider(t["clahe_clip"], 1.0, 5.0, 2.0, 0.1,
                                      help=t["clahe_clip_help"])
    apply_canny = st.sidebar.checkbox(t["apply_canny"], value=False)
    canny_low = st.sidebar.slider(t["canny_low"], 10, 200, 50,
                                  help=t["canny_low_help"])
    canny_high = st.sidebar.slider(t["canny_high"], 50, 500, 150,
                                   help=t["canny_high_help"])

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
    t["upload_image"], type=["jpg", "jpeg", "png"])

# Process image
if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Convert to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
else:
    # Use example image
    use_example = st.checkbox(t["use_example"], value=True)
    if use_example:
        image = create_example_image()
        image_rgb = image.copy()
    else:
        st.markdown(t["upload_prompt"])
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
    st.subheader(t["detected_circles"])

    # Draw circles on the image with highlighting
    image_with_circles = draw_circles_with_numbers(
        image_rgb, circles, st.session_state.selected_circle_idx)

    # Display image and capture clicks
    image_placeholder = st.empty()
    image_placeholder.image(image_with_circles, use_container_width=True)

    # Add click interaction if circles are found
    if circles is not None:
        st.markdown(t["click_prompt"])

        # Add a clickable image using streamlit's image_click capability
        if 'image_clicked' not in st.session_state:
            st.session_state.image_clicked = False

        # Get the actual displayed size to scale the coordinates
        img_height, img_width = image_rgb.shape[:2]

        # Create a clickable area
        click_col = st.columns(1)[0]
        with click_col:
            clicked = st.button(t["reset_selection"])
            if clicked:
                st.session_state.selected_circle_idx = None
                st.rerun()

        # For now we use a coordinate input as a workaround since Streamlit doesn't
        # directly support image clicks in the standard library
        click_cols = st.columns(2)
        with click_cols[0]:
            click_x = st.number_input(t["x_coordinate"], 0, img_width, img_width //
                                      2 if 'click_x' not in st.session_state else st.session_state.get('click_x', 0), key="click_x")
        with click_cols[1]:
            click_y = st.number_input(t["y_coordinate"], 0, img_height, img_height //
                                      2 if 'click_y' not in st.session_state else st.session_state.get('click_y', 0), key="click_y")

        clicked = st.button(t["select_circle_coords"])
        if clicked:
            nearest_idx = find_nearest_circle(circles, click_x, click_y)
            if nearest_idx is not None:
                st.session_state.selected_circle_idx = nearest_idx
                st.rerun()
            else:
                st.warning(t["no_circle_warning"])

    # Show pre-processed image if advanced options are enabled
    if show_advanced:
        show_processed = st.checkbox(t["show_preprocessed"], value=False)
        if show_processed:
            st.subheader(t["preprocessed_image"])
            st.image(processed_gray, use_container_width=True)

with col2:
    st.subheader(t["circle_info"])

    if circles is not None:
        circle_count = circles.shape[1]
        st.write(f"{t['num_circles']}{circle_count}")

        # Create a selection dropdown for circles
        circle_numbers = [
            f"{t['select_circle'].split()[0]} {i+1}" for i in range(circle_count)]
        if circle_numbers:
            selected_circle = st.selectbox(
                t["select_circle"],
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
                st.markdown(t["circle_properties"])
                st.markdown(f"{t['center']}({x}, {y})")
                st.markdown(f"{t['radius']}{r:.2f}{t['pixels']}")
                st.markdown(f"{t['diameter']}{2*r:.2f}{t['pixels']}")
                st.markdown(
                    f"{t['area']}{math.pi * (r ** 2):.2f}{t['sq_pixels']}")
                st.markdown(
                    f"{t['circumference']}{2 * math.pi * r:.2f}{t['pixels']}")

                # Optional: Add scaling to real-world units
                st.markdown("---")
                st.markdown(t["scale_section"])
                scale_factor = st.number_input(t["scale_factor"],
                                               min_value=0.01, value=1.0, step=0.01)

                if scale_factor > 0:
                    real_radius = r / scale_factor
                    real_diameter = 2 * real_radius
                    real_area = math.pi * (real_radius ** 2)
                    real_circumference = 2 * math.pi * real_radius

                    st.markdown(f"{t['radius']}{real_radius:.2f}{t['units']}")
                    st.markdown(
                        f"{t['diameter']}{real_diameter:.2f}{t['units']}")
                    st.markdown(f"{t['area']}{real_area:.2f}{t['sq_units']}")
                    st.markdown(
                        f"{t['circumference']}{real_circumference:.2f}{t['units']}")
    else:
        st.write(t["no_circles"])

        # Provide suggestions for improving detection
        st.markdown(t["suggestions"])
