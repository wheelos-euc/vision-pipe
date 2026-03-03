# YOLO Project — Data Conversion, Training, and Visualization

This repository contains utilities to convert labeled images to a YOLO dataset, train a YOLO model using the Ultralytics API, and visualize inference results in a Streamlit dashboard.

## Contents
- `data_convert.py` — convert labelme-style JSONs into a YOLO-format dataset under `YOLO_Dataset/`.
- `train.py` — simple training entry using `ultralytics.YOLO` (adjust model path and training hyperparameters inside if needed).
- `app.py` — Streamlit app for visualizing video/image inference using trained `.pt` weights.
- `YOLO_Dataset/` — default output folder for converted datasets (`data.yaml`, `images/`, `labels/`).

## Requirements
Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Make sure you have a compatible GPU driver and PyTorch/cuDNN if you plan to train with GPU.

## 1) Data conversion
Convert a folder tree of labeled images (LabelMe JSONs next to images) into the YOLO dataset layout.

Example:

```bash
python data_convert.py --input ./my_images_root --output ./YOLO_Dataset --ratio 0.8
```

What it does:
- Scans top-level subdirectories under the `--input` path (ignores the output folder).
- Finds image files (`.jpg`, `.png`, `.jpeg`) and their corresponding `.json` LabelMe files.
- Converts rectangle annotations to YOLO normalized bbox format and writes `.txt` labels.
- Copies images and labels into `YOLO_Dataset/images/{train,val}` and `YOLO_Dataset/labels/{train,val}`.
- Writes `YOLO_Dataset/data.yaml` with `train`, `val`, and `names` mapping.

Notes:
- If a subfolder has many images, the script shuffles and splits them according to `--ratio`.
- The converter removes the output folder if it already exists — back up data if needed.

## 2) Training
Training is provided by `train.py` using the Ultralytics `YOLO` API. By default it loads `yolo12n.pt`.

Quick start:

1. Ensure `YOLO_Dataset/data.yaml` exists (created by the converter).
2. Edit `train.py` to change the base model or hyperparameters if required.
3. Run:

```bash
python train.py
```

Important:
- `train.py` uses `data=os.path.abspath('./YOLO_Dataset/data.yaml')` — ensure that path is correct.
- Tune `batch`, `imgsz`, `epochs`, and `device` in `train.py` for your hardware.
- Training outputs are saved under `runs/detect/<project>/<name>` by default.

## 3) Visualization (Streamlit)
The Streamlit app in `app.py` provides two tabs: live video detection (upload video) and single image detection.

Run the app:

```bash
streamlit run app.py
```

How to use:
- Place trained `.pt` weight files under `runs/detect/**/weights/` (the app searches `runs/detect/**/weights/*.pt`).
- Use the sidebar to select a weights file, set confidence threshold, and render width.
- Upload a surveillance video (mp4/avi/mov) for looped inference or upload an image for single-frame detection.

Notes:
- The app encodes frames as JPEG base64 for smoother browser rendering and includes simple frame-rate throttling to avoid freezing.
- If no `.pt` files are found, the sidebar will display an error and the app will stop.

## Tips and Troubleshooting
- If `data_convert.py` fails to find JSONs, ensure JSON filenames match image names (same basename).
- For low GPU memory, reduce `batch` in `train.py` to 8 or 4.
- Check `runs/` after training to locate saved weights (`runs/detect/<project>/<exp>/weights/best.pt`).

## License & Contributions
This project uses permissive personal-use conventions. Feel free to open issues or suggest improvements.
