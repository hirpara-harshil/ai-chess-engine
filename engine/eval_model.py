# engine/eval_model.py
import numpy as np
import chess
from tensorflow.keras.models import load_model

# Load once at import
MODEL_PATH = "data/processed/eval_model.keras"
eval_model = load_model(MODEL_PATH)
print(f"✅ Loaded evaluation model from {MODEL_PATH}")

# Piece plane mapping: same as training
PIECE_TO_PLANE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5
}

def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Converts python-chess Board to (8,8,12) tensor for the model.
    Plane 0-5: White pieces, Plane 6-11: Black pieces.
    """
    tensor = np.zeros((8,8,12), dtype=np.float32)
    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        plane = PIECE_TO_PLANE[piece.piece_type]
        if piece.color == chess.BLACK:
            plane += 6
        tensor[row, col, plane] = 1
    return tensor

def evaluate(board: chess.Board) -> float:
    """
    Returns evaluation in centipawns from White's perspective.
    """
    tensor = board_to_tensor(board)
    score = eval_model.predict(tensor[np.newaxis, ...], verbose=0)[0,0]
    return float(score * 100)  # convert pawns to centipawns
