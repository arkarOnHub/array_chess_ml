import cv2
import numpy as np
import joblib
from skimage.feature import hog
import time

# === Load SVM Model (Once) ===
model = joblib.load("chess_images/prepared_data/svm_model.pkl")


# === HOG Feature Extraction Function ===
def extract_hog_features(image):
    image = cv2.resize(image, (60, 60))
    fd, _ = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
    )
    return fd


# === Piece Prediction ===
def predict_board_state(board_image):
    board_image = cv2.resize(board_image, (480, 480))
    label_mapping_inverse = {1: "w", -1: "b", 0: ""}  # w = white, b = black, empty = ""
    board_state = []

    for row in range(8):
        row_data = []
        for col in range(8):
            square = board_image[row * 60 : (row + 1) * 60, col * 60 : (col + 1) * 60]
            features = extract_hog_features(square)
            prediction = model.predict([features])[0]
            row_data.append(label_mapping_inverse[prediction])
        board_state.append(row_data)

    return board_state


# === Board Calibration ===
def calibrate_chessboard():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise Exception("Camera not found!")

    all_corners = []
    print("Calibrating board position for 10 seconds...")
    start_time = time.time()

    while time.time() - start_time < 10:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (7, 7), None)

        if found:
            all_corners.append(corners)

        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(all_corners) == 0:
        raise Exception("Failed to detect chessboard during calibration!")

    average_corners = np.mean(np.array(all_corners), axis=0).reshape(-1, 2)

    # Define the desired 480x480 square
    target_size = 480
    square_points = np.array(
        [
            [0, 0],
            [target_size - 1, 0],
            [target_size - 1, target_size - 1],
            [0, target_size - 1],
        ],
        dtype="float32",
    )

    ordered_corners = np.array(
        [
            average_corners[0],
            average_corners[6],
            average_corners[-1],
            average_corners[-7],
        ],
        dtype="float32",
    )

    transform_matrix = cv2.getPerspectiveTransform(ordered_corners, square_points)

    return transform_matrix, target_size


# === Main Process ===
def main_pipeline():
    transform_matrix, board_size = calibrate_chessboard()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise Exception("Camera not found!")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Crop and warp to 480x480 board
        cropped_board = cv2.warpPerspective(
            gray, transform_matrix, (board_size, board_size)
        )

        # Predict each square's content
        board_state = predict_board_state(cropped_board)

        # Draw the detected pieces directly on the live feed
        for row in range(8):
            for col in range(8):
                piece = board_state[row][col]
                if piece:  # Only draw if not empty
                    text_x = col * 60 + 20
                    text_y = row * 60 + 40
                    cv2.putText(
                        cropped_board,
                        piece,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

        cv2.imshow("Augmented Chessboard", cropped_board)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main_pipeline()
