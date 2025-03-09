# **Chess-Playing Robotic Arm**

## **Overview**
This project is a chess-playing robotic arm that can play against a human opponent. It uses computer vision to detect the board state, processes the moves using a chess engine, and physically moves the pieces using a SCARA robotic arm.

## **Features**
- **Computer Vision**: A top-mounted camera extracts the chessboard state and converts it to FEN notation.
- **Chess Engine Integration**: The system interfaces with a well-known chess engine to calculate the best move.
- **Robotic Arm Control**: A SCARA robot executes the moves on a physical chessboard.
- **Color Detection**: Uses black-and-white color recognition instead of individual piece classification.

## **Setup Instructions**
### **1. Clone the Repository**
```sh
https://github.com/arkarOnHub/Chess-Playing-Robot-Arm.git
cd Chess-Playing-Robot-Arm
```

### **2. Set Up the Environment**
This project requires Python and Conda. To set up the environment:

#### **Using Conda**
```sh
conda env create -f environment.yml
conda activate raspberryturk
```

#### **Or Using pip (if not using Conda)**
```sh
pip install -r requirements.txt
```

## **Hardware Requirements**
- **Camera**: Overhead camera for board recognition.
- **SCARA Robotic Arm**: Executes physical movements.
- **Chessboard with Standard Pieces**.

Thank you!
