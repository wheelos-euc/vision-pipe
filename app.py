# Copyright 2026 The WheelOS Team. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Created Date: 2026-03-03
# Author: daohu527


import streamlit as st
import cv2
import tempfile
import os
import glob
import time
import base64
from PIL import Image
from ultralytics import YOLO

# 1. Basic configuration (use wide layout)
st.set_page_config(page_title="AI Vision Dashboard",
                   layout="wide", page_icon="🎯")
st.title("🎯 Industrial Visual Inspection (Streamlined)")


@st.cache_resource
def load_model(path):
    try:
        return YOLO(path)
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None


def get_available_models(search_dir="runs"):
    # Search for runs/detect/**/weights/*.pt recursively under each exp/weights
    pattern = os.path.join(search_dir, "detect", "**", "weights", "*.pt")
    pt_files = glob.glob(pattern, recursive=True)
    # Deduplicate and sort for consistent sidebar display
    pt_files = sorted(list(set(pt_files)))
    # Do not return a default; return empty list if none found
    return pt_files


# --- Sidebar: Configuration ---
st.sidebar.header("⚙️ Global Settings")
available_models = get_available_models()
if not available_models:
    st.sidebar.error(
        "No .pt weight files found under runs/detect/**/weights. Please place weights in the appropriate directory and try again.")
    st.stop()
model_path = st.sidebar.selectbox(
    "Select weights file", options=available_models, format_func=lambda x: os.path.basename(x))
conf_th = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25)
target_width = st.sidebar.slider(
    "Render width (recommended 640)", 400, 1024, 640)

model = load_model(model_path)
if not model:
    st.stop()

# --- Main interface ---
tab1, tab2 = st.tabs(["🎥 Live Video Detection", "🖼️ Single Image Detection"])

# ----------------- Tab 1: Video monitoring -----------------
with tab1:
    v_col_main, v_col_side = st.columns([3, 1])

    # Place upload and control widgets at the top of the main column
    # Render the video container above them for a cleaner layout
    with v_col_main:
        # Use an empty container to render HTML (video displayed above upload controls)
        st_video_html = st.empty()

        uploader_col, control_col = st.columns([3, 1])
        with uploader_col:
            video_file = st.file_uploader(
                "Upload surveillance video", type=['mp4', 'avi', 'mov'], key="vid_upload")
        with control_col:
            run_video = st.checkbox(
                "▶️ Start Detection / Loop Playback", value=True)

    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)

        # Get original video FPS to control playback speed (default to 25)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if not video_fps or video_fps > 60:
            video_fps = 25.0

        # Compute aspect ratio
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        target_height = int(target_width * (orig_h / orig_w)
                            ) if orig_w > 0 else int(target_width * 0.75)

        with v_col_side:
            st.subheader("📊 Live Detection Results")
            st_result_text = st.empty()  # placeholder for updating text

        try:
            while cap.isOpened() and run_video:
                loop_start_time = time.time()  # Record start time for this frame

                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop playback
                    continue

                # 1. Run AI inference
                results = model.predict(frame, conf=conf_th, verbose=False)
                res_plotted = results[0].plot()
                res_resized = cv2.resize(
                    res_plotted, (target_width, target_height))

                # 2. Count detected objects in current frame
                current_counts = {}
                for c in results[0].boxes.cls:
                    label = model.names[int(c)]
                    current_counts[label] = current_counts.get(label, 0) + 1

                # 3. Encode frame as Base64 to bypass Streamlit media file management
                # This avoids "Missing file" errors.
                _, buffer = cv2.imencode('.jpg', res_resized, [
                                         cv2.IMWRITE_JPEG_QUALITY, 80])
                b64_str = base64.b64encode(buffer).decode("utf-8")

                # Render via HTML <img> tag for smooth native browser display
                html_img = f'<img src="data:image/jpeg;base64,{b64_str}" style="width: 100%; border-radius: 8px;">'
                st_video_html.markdown(html_img, unsafe_allow_html=True)

                # 4. Update right-side results panel
                if current_counts:
                    res_md = ""
                    for label, num in current_counts.items():
                        res_md += f"- **{label}**: {num}\n"
                    st_result_text.success(res_md)
                else:
                    st_result_text.info("Scene clear (no targets)")

                # 5. Frame-rate control: avoid processing frames faster than
                # the original video rate to prevent browser freezes.
                # Ensure per-frame processing time matches the source frame interval.
                process_time = time.time() - loop_start_time
                target_frame_time = 1.0 / video_fps
                if process_time < target_frame_time:
                    time.sleep(target_frame_time - process_time)

        finally:
            cap.release()
            try:
                os.remove(tfile.name)
            except Exception:
                pass

# ----------------- Tab 2: Image detection -----------------
with tab2:
    col1, col2 = st.columns([2, 1])
    img_file = st.file_uploader(
        "Upload image for detection", type=['jpg', 'jpeg', 'png'], key="img_upload")

    if img_file:
        input_img = Image.open(img_file)
        with st.spinner('AI detecting...'):
            results = model.predict(input_img, conf=conf_th)
            res_plotted = results[0].plot()[:, :, ::-1]

        with col1:
            st.image(res_plotted, caption="Detection Result", width="stretch")

        with col2:
            st.subheader("📋 Detection Details")
            counts = {}
            for c in results[0].boxes.cls:
                label = model.names[int(c)]
                counts[label] = counts.get(label, 0) + 1

            if counts:
                for label, num in counts.items():
                    st.success(f"**{label}**: {num}")
            else:
                st.info("No targets detected")
