import sys
print("Python version:", sys.version)
import streamlit as st
import zipfile
import tempfile
import os
import cv2
import numpy as np
import json
import shutil
import glob
from io import BytesIO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(layout="wide")
st.title("📷 Camera Calibration with CharuCo")

# --- Sidebar: File Upload ---
zip_file = st.sidebar.file_uploader("Upload ZIP of Calibration Images", type="zip")

# --- Sidebar: Camera Info ---
camera_model = st.sidebar.text_input("Camera Model")
camera_serial = st.sidebar.text_input("Camera Serial Number")
lens_model = st.sidebar.text_input("Lens Model")
distortion_model = st.sidebar.selectbox("Distortion Model", ["regular (5 parameters)", "rational (8 parameters)"])

aruco_dict_options = [d for d in dir(cv2.aruco) if d.startswith('DICT_')]
dict_option = st.sidebar.selectbox("ArUco Dictionary", aruco_dict_options, index=aruco_dict_options.index('DICT_5X5_100'))

squares_x = st.sidebar.number_input("Number of Squares X (columns)", min_value=2, value=24)
squares_y = st.sidebar.number_input("Number of Squares Y (rows)", min_value=2, value=17)
square_length = st.sidebar.number_input("Square Length [mm]", min_value=1.0, value=30.0)
marker_length = st.sidebar.number_input("Marker Length [mm]", min_value=1.0, value=22.0)

grid_size = st.sidebar.number_input("Grid Analysis Size", min_value=5, value=15)
ok_threshold = st.sidebar.number_input("OK Threshold (corners)", min_value=5, value=50)

show_analysis = st.sidebar.button("Show Analysis")

# Temporary folder to extract images
@st.cache_data(show_spinner=False)
def extract_zip(file):
    tmpdir = tempfile.mkdtemp()
    with zipfile.ZipFile(file, "r") as z:
        z.extractall(tmpdir)
    return tmpdir

if zip_file:
    image_dir = extract_zip(zip_file)
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png")))
    selected_image = st.sidebar.selectbox("Choose Image for Display", [os.path.basename(p) for p in image_paths])

    # --- Image Carousel ---
    st.subheader("🖼️ Image Preview")
    idx = [os.path.basename(p) for p in image_paths].index(selected_image)
    st.image(image_paths[idx], caption=selected_image, use_column_width=True)

    # --- Calibration Logic ---
    if show_analysis:
        st.subheader("📐 Calibration Results")

        aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_option))
        board = cv2.aruco.CharucoBoard_create(squares_x, squares_y, square_length, marker_length, aruco_dict)

        all_corners, all_ids, imsize = [], [], None
        valid_images = []

        for img_path in image_paths:
            image = cv2.imread(img_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
            if ids is not None:
                _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
                if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > ok_threshold:
                    all_corners.append(charuco_corners)
                    all_ids.append(charuco_ids)
                    valid_images.append(img_path)
                    if imsize is None:
                        imsize = gray.shape[::-1]

        if len(all_corners) < 3:
            st.error("Not enough valid images for calibration. Try lowering OK threshold.")
        else:
            flags = 0
            if distortion_model == "rational (8 parameters)":
                flags |= cv2.CALIB_RATIONAL_MODEL

            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                charucoCorners=all_corners,
                charucoIds=all_ids,
                board=board,
                imageSize=imsize,
                cameraMatrix=None,
                distCoeffs=None,
                flags=flags
            )

            # --- Results Display ---
            st.write("### 📸 Camera Matrix")
            st.code(np.array2string(camera_matrix, precision=4, separator=", "))

            st.write("### 🎯 Distortion Coefficients")
            st.code(np.array2string(dist_coeffs, precision=4, separator=", "))

            st.write("### 🖼️ Image Size")
            st.code(f"{imsize}")

            # --- Grid Analysis (2D Plot) ---
            fig, ax = plt.subplots()
            ax.set_title("CharuCo Grid Points per Image")
            ax.bar(range(len(valid_images)), [len(ids) for ids in all_ids])
            ax.axhline(ok_threshold, color="red", linestyle="--", label="OK Threshold")
            ax.set_xlabel("Image Index")
            ax.set_ylabel("Corners Detected")
            ax.legend()
            st.pyplot(fig)

            # --- Reprojection Error (per image) ---
            reproj_errors = []
            for i in range(len(valid_images)):
                imgpoints2, _ = cv2.projectPoints(all_corners[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
                error = cv2.norm(all_corners[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                reproj_errors.append(error)

            st.write("### 📊 Reprojection Error per Image")
            error_fig, error_ax = plt.subplots()
            error_ax.plot(reproj_errors, marker="o")
            error_ax.set_xlabel("Image Index")
            error_ax.set_ylabel("Reprojection Error (L2 norm)")
            st.pyplot(error_fig)

            # --- 3D Visualization of Reprojection Error ---
            fig3d = plt.figure()
            ax3d = fig3d.add_subplot(111, projection='3d')
            for i, (rvec, tvec, corners) in enumerate(zip(rvecs, tvecs, all_corners)):
                projected, _ = cv2.projectPoints(corners, rvec, tvec, camera_matrix, dist_coeffs)
                for pt, proj in zip(corners, projected):
                    x, y = pt[0]
                    xp, yp = proj[0]
                    err = np.linalg.norm([x - xp, y - yp])
                    ax3d.scatter(x, y, err, c='b', marker='o')
            ax3d.set_xlabel("X")
            ax3d.set_ylabel("Y")
            ax3d.set_zlabel("Reproj Error (L2)")
            st.pyplot(fig3d)

            # --- Extrinsics Visualization (Axes Overlay) ---
            st.subheader("🧭 Extrinsics Visualization (Axes Overlay)")
            
            selected_extrinsic_img = st.selectbox("Choose Image for Extrinsic Axes", [os.path.basename(p) for p in valid_images])
            extrinsic_idx = [os.path.basename(p) for p in valid_images].index(selected_extrinsic_img)
            
            # Reload image
            image = cv2.imread(valid_images[extrinsic_idx])
            image_axes = image.copy()
            
            # Define 3D axes (X: red, Y: green, Z: blue)
            axis_length = square_length * 0.5  # half square size
            axis_3d = np.float32([[axis_length,0,0], [0,axis_length,0], [0,0,-axis_length]]).reshape(-1,3)
            
            # Find the center of the CharuCo board for origin placement
            corner = all_corners[extrinsic_idx][0]
            origin_2d, _ = cv2.projectPoints(np.zeros((1,3)), rvecs[extrinsic_idx], tvecs[extrinsic_idx], camera_matrix, dist_coeffs)
            axes_2d, _ = cv2.projectPoints(axis_3d, rvecs[extrinsic_idx], tvecs[extrinsic_idx], camera_matrix, dist_coeffs)
            
            corner = tuple(origin_2d[0].ravel().astype(int))
            x_axis = tuple(axes_2d[0].ravel().astype(int))
            y_axis = tuple(axes_2d[1].ravel().astype(int))
            z_axis = tuple(axes_2d[2].ravel().astype(int))
            
            cv2.line(image_axes, corner, x_axis, (0,0,255), 3)  # X - Red
            cv2.line(image_axes, corner, y_axis, (0,255,0), 3)  # Y - Green
            cv2.line(image_axes, corner, z_axis, (255,0,0), 3)  # Z - Blue
            cv2.circle(image_axes, corner, 5, (255,255,255), -1)
            
            st.image(image_axes, caption="Extrinsics Axes Visualization", use_column_width=True)

            
            # --- Export Calibration ---
            if st.button("📁 Export Calibration JSON"):
                export_path = os.path.join(image_dir, f"calib_{camera_model}_{camera_serial}.json")
                calib_data = {
                    "camera_model": camera_model,
                    "camera_serial": camera_serial,
                    "lens_model": lens_model,
                    "image_size": imsize,
                    "camera_matrix": camera_matrix.tolist(),
                    "dist_coeffs": dist_coeffs.tolist(),
                    "distortion_model": distortion_model,
                    "aruco_dict": dict_option,
                }
                with open(export_path, "w") as f:
                    json.dump(calib_data, f, indent=4)
                st.success(f"Calibration exported to: {export_path}")

