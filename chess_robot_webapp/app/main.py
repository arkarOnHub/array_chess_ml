from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import numpy as np
import time
import os
import asyncio
import joblib
from skimage.feature import hog
import collections
import chess  # New import

app = FastAPI()

# Global variables
calibrated_corners = None
# New global variable to track last extracted BW FEN
last_bw_fen = None
prev_gray_frame = None
motion_history = collections.deque(maxlen=10)
stable_frame_count = 0
output_folder = "cropped_frames"
fen_result = "FEN extraction inactive"
fen_extraction_active = False
os.makedirs(output_folder, exist_ok=True)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# New global variables for FEN conversion
previous_full_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"  # Standard starting position

# Load models
hog_model = joblib.load(
    "../../chess_images/prepared_data/tf_hog_random_forest_model.pkl"
)
hsv_model = joblib.load(
    "../../chess_images/prepared_data/bw_hsv_random_forest_model.pkl"
)


# New FEN conversion functions
def parse_bw_to_board(bw_board):
    board = []
    for row in bw_board.split("/"):
        parsed_row = []
        for char in row:
            if char == "b":
                parsed_row.append("b")
            elif char == "w":
                parsed_row.append("w")
            else:
                parsed_row.extend([""] * int(char))
        board.append(parsed_row)
    return board


def parse_fen_to_board(fen):
    fen_board = fen.split(" ")[0]
    board = []
    for row in fen_board.split("/"):
        parsed_row = []
        for char in row:
            if char.isdigit():
                parsed_row.extend([""] * int(char))
            else:
                parsed_row.append(char.lower())
        board.append(parsed_row)
    return board


def compare_boards(previous_board, current_board):
    from_square = None
    to_square = None
    for rank in range(8):
        for file in range(8):
            prev_piece = previous_board[rank][file]
            curr_piece = current_board[rank][file]
            if prev_piece and not curr_piece:
                from_square = chess.square(file, 7 - rank)
            if not prev_piece and curr_piece:
                to_square = chess.square(file, 7 - rank)
    return from_square, to_square


def update_fen_from_bw(bw_fen):
    global previous_full_fen
    try:
        previous_board = parse_fen_to_board(previous_full_fen)
        current_board = parse_bw_to_board(bw_fen)

        from_square, to_square = compare_boards(previous_board, current_board)
        if from_square is None or to_square is None:
            return previous_full_fen

        move = chess.Move(from_square, to_square)
        board = chess.Board(previous_full_fen)

        if move in board.legal_moves:
            board.push(move)
            previous_full_fen = board.fen()
        else:
            print(f"Illegal move detected: {move.uci()}")
    except Exception as e:
        print(f"Error updating FEN: {e}")
    return previous_full_fen


# Existing functions remain exactly the same
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


def extract_hsv_features(image, bins=16):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv_image], [0], None, [bins], [0, 180])
    s_hist = cv2.calcHist([hsv_image], [1], None, [bins], [0, 256])
    v_hist = cv2.calcHist([hsv_image], [2], None, [bins], [0, 256])
    h_hist = cv2.normalize(h_hist, h_hist).flatten()
    s_hist = cv2.normalize(s_hist, s_hist).flatten()
    v_hist = cv2.normalize(v_hist, v_hist).flatten()
    return np.concatenate((h_hist, s_hist, v_hist))


def is_board_stable(frame, threshold=1.0, stable_threshold=3):
    global prev_gray_frame, stable_frame_count
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_gray_frame is None:
        prev_gray_frame = gray
        return False
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    motion_score = np.mean(np.linalg.norm(flow, axis=2))
    if motion_score > threshold:
        stable_frame_count = 0
        return False
    stable_frame_count += 1
    if stable_frame_count >= stable_threshold:
        prev_gray_frame = gray
        stable_frame_count = 0
        return True
    return False


def get_chessboard_corners(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2
    )
    found, corners = cv2.findChessboardCorners(
        gray, (7, 7), flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ADAPTIVE_THRESH
    )
    if not found:
        raise ValueError("Couldn't find chessboard in current frame")
    return corners.reshape(-1, 2)


def calibrate_board():
    global calibrated_corners
    print("Calibrating chessboard corners for 10 seconds...")
    all_corners = []
    start_time = time.time()
    while time.time() - start_time < 1:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            continue
        try:
            corners = get_chessboard_corners(frame)
            all_corners.append(corners)
        except Exception as e:
            print(f"Skipping frame during calibration: {e}")
        time.sleep(0.5)
    if not all_corners:
        raise RuntimeError("Calibration failed - no corners detected.")
    calibrated_corners = np.mean(np.array(all_corners), axis=0)
    print(f"Calibration complete. Averaged corners:\n{calibrated_corners}")


def crop_using_calibrated_corners(frame):
    if calibrated_corners is None:
        return frame
    h, w = frame.shape[:2]
    src_pts = np.array(
        [
            calibrated_corners[0],
            calibrated_corners[6],
            calibrated_corners[-7],
            calibrated_corners[-1],
        ],
        dtype=np.float32,
    )
    size = min(w, h)
    dst_pts = np.array(
        [[60, 60], [size - 60, 60], [60, size - 60], [size - 60, size - 60]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(frame, M, (size, size))


def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            frame = crop_using_calibrated_corners(frame)
            frame, _ = predict_fen_and_overlay(frame)
            ret, buffer = cv2.imencode(".jpg", frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
        except Exception as e:
            print(f"Error processing frame: {e}")


def predict_fen_and_overlay(frame):
    global fen_extraction_active, fen_result, previous_full_fen, last_bw_fen

    if not fen_extraction_active:
        return frame, None

    if not is_board_stable(frame):
        print("Motion detected! Skipping BW FEN extraction...")
        return frame, None

    print("Board stable! Proceeding with BW FEN extraction...")

    # Extract BW FEN as before
    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayscale_image = cv2.resize(grayscale_image, (480, 480))
    color_image = cv2.resize(frame, (480, 480))
    step = 60
    predictions = []

    for row in range(8):
        row_data = []
        for col in range(8):
            square_gray = grayscale_image[
                row * step : (row + 1) * step, col * step : (col + 1) * step
            ]
            square_color = color_image[
                row * step : (row + 1) * step, col * step : (col + 1) * step
            ]

            hog_features = extract_hog_features(square_gray)
            is_occupied = hog_model.predict([hog_features])[0]

            if is_occupied == 0:
                row_data.append("1")
                continue

            hsv_features = extract_hsv_features(square_color)
            piece_color = hsv_model.predict([hsv_features])[0]
            label = "w" if piece_color == 1 else "b"
            row_data.append(label)

        # Convert row to compressed FEN notation
        fen_row = "".join(row_data)
        compact_fen_row = ""
        count = 0

        for char in fen_row:
            if char == "1":
                count += 1
            else:
                if count > 0:
                    compact_fen_row += str(count)
                    count = 0
                compact_fen_row += char
        if count > 0:
            compact_fen_row += str(count)

        predictions.append(compact_fen_row)

    bw_fen = "/".join(predictions)

    # Check if BW FEN has changed before updating the full FEN
    if bw_fen != last_bw_fen:
        print(f"New BW FEN detected: {bw_fen}")
        last_bw_fen = bw_fen  # Update last extracted BW FEN
        previous_full_fen = update_fen_from_bw(bw_fen)  # Update real FEN
        print(previous_full_fen)
    else:
        print("No changes in BW FEN. Keeping previous FEN.")

    fen_result = bw_fen  # Keep returning BW notation for backward compatibility
    return color_image, bw_fen


# Existing endpoints remain exactly the same
@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/start_calibration")
async def start_calibration():
    calibrate_board()
    return {"message": "Calibration complete!"}


@app.get("/get_fen")
async def get_fen():
    global fen_result
    return {"fen": fen_result if fen_extraction_active else "FEN extraction inactive"}


@app.post("/toggle_fen_extraction")
async def toggle_fen_extraction():
    global fen_extraction_active
    fen_extraction_active = not fen_extraction_active
    return {
        "message": f"FEN extraction {'activated' if fen_extraction_active else 'deactivated'}"
    }


# New endpoint for full FEN
@app.get("/get_full_fen")
async def get_full_fen():
    global previous_full_fen
    return {"full_fen": previous_full_fen}


@app.get("/")
async def index():
    with open("../frontend/index.html", "r") as file:
        return HTMLResponse(content=file.read())
