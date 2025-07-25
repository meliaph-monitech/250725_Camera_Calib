import streamlit as st
import zipfile
import os
import io
import numpy as np
import matplotlib.pyplot as plt
import json
import plotly.express as px
from PIL import Image
from pathlib import Path

# Defer cv2 import to avoid Streamlit Cloud issues
# try:
#     import cv2
# except Exception as e:
#     st.error(f"OpenCV import failed: {e}")
#     st.stop()
try:
    import cv2
    from cv2 import aruco
    assert hasattr(cv2.aruco, "CharucoBoard_create"), "CharucoBoard_create not available"
except Exception as e:
    st.error(f"OpenCV ArUco module is not available: {e}")
    st.stop()


# Title
st.title("Camera Calibration with CharuCo Board")

# Sidebar Inputs
st.sidebar.header("Calibration Settings")

# Camera Info (text fields)
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

# ZIP Upload
zip_file = st.sidebar.file_uploader("Upload ZIP of Calibration Images", type="zip")
show_analysis = st.sidebar.button("Show Analysis")

if zip_file:
    # Unzip to temp directory
    temp_dir = Path("temp_images")
    if temp_dir.exists():
        for file in temp_dir.glob("*"):
            file.unlink()
    else:
        temp_dir.mkdir()

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    image_paths = sorted(list(temp_dir.glob("*.jpg")) + list(temp_dir.glob("*.png")) + list(temp_dir.glob("*.jpeg")))

    if len(image_paths) == 0:
        st.warning("No valid image files found in ZIP.")
        st.stop()

    st.subheader("Image Preview")
    st.image(str(image_paths[0]), caption="Sample Image", use_column_width=True)

    # Calibration logic
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_option))
    board = cv2.aruco.CharucoBoard_create(squares_x, squares_y, square_length, marker_length, aruco_dict)

    all_corners = []
    all_ids = []
    img_size = None

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

        if len(corners) > 0:
            _, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            if ch_corners is not None and ch_ids is not None and len(ch_corners) > 4:
                all_corners.append(ch_corners)
                all_ids.append(ch_ids)
                img_size = gray.shape[::-1]

    if len(all_corners) >= 5:
        flags = 0 if distortion_model.startswith("regular") else cv2.CALIB_RATIONAL_MODEL
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            all_corners, all_ids, board, img_size, None, None, flags=flags)

        st.subheader("Calibration Results")
        st.markdown(f"**Camera Model**: {camera_model}")
        st.markdown(f"**Serial Number**: {camera_serial}")
        st.markdown(f"**Lens Model**: {lens_model}")

        st.markdown("**Camera Matrix:**")
        st.code(np.array2string(camera_matrix, precision=4, suppress_small=True))

        st.markdown("**Distortion Coefficients:**")
        st.code(np.array2string(dist_coeffs, precision=4, suppress_small=True))

        st.markdown("**Image Size:**")
        st.code(f"{img_size}")

        if show_analysis:
            grid_x = np.linspace(0, img_size[0], grid_size)
            grid_y = np.linspace(0, img_size[1], grid_size)
            gx, gy = np.meshgrid(grid_x, grid_y)
            grid = np.stack([gx.ravel(), gy.ravel()], axis=-1)

            undistorted = cv2.undistortPoints(grid.astype(np.float32).reshape(-1, 1, 2), camera_matrix, dist_coeffs, P=camera_matrix)
            undistorted = undistorted.reshape(-1, 2)

            fig_grid = plt.figure()
            plt.scatter(grid[:, 0], grid[:, 1], label='Original', s=10)
            plt.scatter(undistorted[:, 0], undistorted[:, 1], label='Undistorted', s=10)
            plt.legend()
            plt.title("Grid Corner Analysis (Before & After Undistortion)")
            st.pyplot(fig_grid)

            # Avg projection error image-wise
            st.subheader("Average Reprojection Error")
            selected_img = st.selectbox("Select Image for Error Visualization", [str(p.name) for p in image_paths])
            img = cv2.imread(str(temp_dir / selected_img))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

            if len(corners) > 0:
                _, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
                if ch_corners is not None and ch_ids is not None:
                    retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                        ch_corners, ch_ids, board, camera_matrix, dist_coeffs, None, None)
                    if retval:
                        proj_pts, _ = cv2.projectPoints(board.chessboardCorners[ch_ids.flatten()], rvec, tvec, camera_matrix, dist_coeffs)
                        errors = np.linalg.norm(proj_pts.squeeze() - ch_corners.squeeze(), axis=1)

                        fig_err = px.scatter(x=ch_corners.squeeze()[:, 0], y=ch_corners.squeeze()[:, 1],
                                             color=errors, color_continuous_scale='Viridis',
                                             labels={'x': 'X', 'y': 'Y', 'color': 'Reprojection Error'},
                                             title='Reprojection Error per Corner')
                        st.plotly_chart(fig_err)

                        fig3d = px.scatter_3d(x=ch_corners.squeeze()[:, 0],
                                              y=ch_corners.squeeze()[:, 1],
                                              z=errors,
                                              color=errors,
                                              labels={'x': 'X', 'y': 'Y', 'z': 'L2 Error'},
                                              title="3D Error Plot",
                                              hover_name=[f"Error: {e:.2f}" for e in errors])
                        st.plotly_chart(fig3d)

        # Export calibration file
        if st.button("Export Calibration"):
            output = {
                "camera_model": camera_model,
                "serial_number": camera_serial,
                "lens_model": lens_model,
                "image_size": img_size,
                "camera_matrix": camera_matrix.tolist(),
                "distortion_coefficients": dist_coeffs.tolist(),
                "distortion_model": distortion_model
            }
            json_data = json.dumps(output, indent=2)
            st.download_button(label="Download Calibration JSON",
                               data=json_data,
                               file_name=f"calibration_{camera_model}_{camera_serial}.json",
                               mime="application/json")


    else:
        st.warning("Not enough valid CharuCo detections for calibration (need at least 5 images with 4+ corners)")
