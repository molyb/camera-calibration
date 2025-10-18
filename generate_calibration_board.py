import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse

# Paper A3R
paper_width_mm = 420
paper_height_mm = 297

# ChArUco
checker_size_mm = 22
marker_size_mm = 16
board_cell_x = 16
board_cell_y = 12


def create_canvas(width_pixel, height_pixel):
    return np.full((height_pixel, width_pixel), 255, dtype=np.uint8)  # white


def merge(canvas, image):
    h_canvas, w_canvas = canvas.shape
    h_image, w_image = image.shape

    y0 = (h_canvas - h_image) // 2
    x0 = (w_canvas - w_image) // 2

    merged = canvas.copy()
    merged[y0 : y0 + h_image, x0 : x0 + w_image] = image
    return merged


def put_params(image_in):
    # e.g. "| 16 x 12 | Checker 22 mm | Marker 16 mm | Dictionary 4X4_250 |"
    text = f"| {board_cell_x} x {board_cell_y} | Checker {checker_size_mm} mm | Marker {marker_size_mm} mm | Dictionary 4X4_250 |"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = image_in.shape[0] * 0.0006
    thickness = round(font_scale * 0.75)  # font_scaleを丸めるのはこの計算の後
    font_scale = int(font_scale)
    color = (0, 0, 0)  # black
    print(image_in.shape)
    image_out = image_in.copy()

    # 左下の座標を計算（余白を入れる）
    tolerance_pixel = int(image_out.shape[0] * 0.01)
    x = tolerance_pixel
    y = image_out.shape[0] - tolerance_pixel  # 画像下端から n [pixel] 上

    cv2.putText(
        image_out, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA
    )
    return image_out


def generate_board_image(width_pixel: int, height_pixel: int, dpi: int):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    board = cv2.aruco.CharucoBoard(
        size=(board_cell_x, board_cell_y),
        squareLength=checker_size_mm,
        markerLength=marker_size_mm,
        dictionary=dictionary,
    )

    checker_size_pixel = checker_size_mm * dpi / 25.4
    checker_size_width_pixel = int(checker_size_pixel * board_cell_x)
    checker_size_height_pixel = int(checker_size_pixel * board_cell_y)
    image = board.generateImage((checker_size_width_pixel, checker_size_height_pixel))

    canvas = create_canvas(width_pixel, height_pixel)
    image = merge(canvas, image)
    image = put_params(image)
    return image


def main():
    parser = argparse.ArgumentParser(description="Generate a Calibration board.")
    parser.add_argument(
        "--visualize", action="store_true", help="Save corner visualization images."
    )
    args = parser.parse_args()

    dpi = int(600)
    paper_width_inch = paper_width_mm / 25.4
    paper_height_inch = paper_height_mm / 25.4
    paper_width_pixel = int(dpi * paper_width_inch)
    paper_height_pixel = int(dpi * paper_height_inch)
    image = generate_board_image(paper_width_pixel, paper_height_pixel, dpi)

    if args.visualize:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        plt.imshow(image_rgb)
        plt.axis("off")
        plt.show()

    pil_img = Image.fromarray(image)

    # pil_img.save("calibration_board.png", dpi=(dpi, dpi))
    pil_img.save("calibration_board.pdf", dpi=(dpi, dpi))


if __name__ == "__main__":
    main()
