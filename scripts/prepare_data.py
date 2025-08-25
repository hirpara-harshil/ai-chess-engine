import random
import numpy as np
import chess
from load_positions import load_positions

# Load positions
data = load_positions("data/processed/stockfish_positions.jsonl")
random.shuffle(data)

# Split into train/validation
split_idx = int(0.9 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]

print(f"Training: {len(train_data)}, Validation: {len(val_data)}")

# FEN → 8x8x12 array
def fen_to_array(fen):
    board = chess.Board(fen)
    arr = np.zeros((8,8,12), dtype=np.int8)
    piece_map = board.piece_map()
    piece_to_idx = {
        chess.PAWN:0, chess.KNIGHT:1, chess.BISHOP:2,
        chess.ROOK:3, chess.QUEEN:4, chess.KING:5
    }
    for square, piece in piece_map.items():
        row = 7 - square // 8
        col = square % 8
        idx = piece_to_idx[piece.piece_type] + (0 if piece.color else 6)
        arr[row, col, idx] = 1
    return arr

# Prepare arrays
X_train = np.array([fen_to_array(d['fen']) for d in train_data])
y_train = np.array([d['eval'] for d in train_data])

X_val = np.array([fen_to_array(d['fen']) for d in val_data])
y_val = np.array([d['eval'] for d in val_data])

print(X_train.shape, y_train.shape)
