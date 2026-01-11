import cv2
from typing import List
import numpy as np
import imageio
import copy
import glob
import os
from os.path import join
import matplotlib.pyplot as plt

def load_images(filenames: List) -> List:
    return [cv2.imread(filename) for filename in filenames]

def show_image(img, img_name):
    cv2.imshow(img_name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
def write_image(img: np.array, img_name: str):
    cv2.imwrite(img_name, img)

def get_chessboard_points(chessboard_shape, dx, dy):
    points = []
    for i in range(chessboard_shape[1]):
        for j in range(chessboard_shape[0]):
            points.append([j * dx, i * dy, 0])
    return np.array(points, dtype=np.float32)

def calibration():
    imgs_path = sorted(glob.glob("../assets/calibration/*.jpg"))
    imgs = load_images(imgs_path)

    corners = [cv2.findChessboardCorners(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),(6,4)) for img in imgs]

    corners_copy = copy.deepcopy(corners)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    imgs_gray = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in imgs]

    corners_refined = [cv2.cornerSubPix(i, cor[1], (6, 4), (-1, -1), criteria) if cor[0] else [] for i, cor in zip(imgs_gray, corners_copy)]

    imgs_copy = copy.deepcopy(imgs)

    draw_corners = [cv2.drawChessboardCorners(imgs_copy[i], (6,4), corners_refined[i], corners[i][0]) for i in range(len(imgs_copy))]

    output_folder = join(os.getcwd(), "..", "assets", "calibration", "outputs")

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    for i, img in enumerate(draw_corners):
        output_img_path = join(output_folder, f"detection_{i}.png")
        write_image(img, output_img_path)

    chessboard_points = [get_chessboard_points((6, 4), 31.5, 31.5) for _ in range(len(draw_corners))]

    valid_corners = [cor[1] for cor in corners if cor[0]]
    # Convert list to numpy array
    valid_corners = np.asarray(valid_corners, dtype=np.float32)

    image_size = (imgs_gray[0].shape[1], imgs_gray[0].shape[0])
    rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(chessboard_points, valid_corners,image_size,None,None)

    # Obtain extrinsics
    extrinsics = list(map(lambda rvec, tvec: np.hstack((cv2.Rodrigues(rvec)[0], tvec)), rvecs, tvecs))

    print("Intrinsics:\n", intrinsics)
    print("Distortion coefficients:\n", dist_coeffs)
    print("Root mean squared reprojection error:\n", rms)

    # Añadir corrección
    intrinsics = np.array(intrinsics)
    dist_coeffs = np.array(dist_coeffs)

    dsts = [cv2.undistort(img, intrinsics, dist_coeffs, None, intrinsics) for img in imgs]

    output_folder = join(os.getcwd(), "..", "assets", "calibration", "corrected")

    # Check the folder exists
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    for i, dst in enumerate(dsts):
        # show_image(img) # comentar esta linea en mac, simplemente visualizar la salida en las imagenes guardadas
        output_img_path = join(output_folder, f"corrected_{i}.png")
        write_image(dst, output_img_path)