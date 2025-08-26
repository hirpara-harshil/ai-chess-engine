# engine/eval_model.py
import numpy as np
import chess, chess.polyglot
from tensorflow.keras.models import load_model
from collections import OrderedDict

MODEL_PATH = "data/processed/eval_model.keras"
eval_model = load_model(MODEL_PATH)
print(f"✅ Loaded evaluation model from {MODEL_PATH}")

PIECE_TO_PLANE = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
}

def board_to_tensor(board: chess.Board) -> np.ndarray:
    tensor = np.zeros((8,8,12), dtype=np.float32)
    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        plane = PIECE_TO_PLANE[piece.piece_type] + (6 if piece.color == chess.BLACK else 0)
        tensor[row, col, plane] = 1.0
    return tensor

# --- small LRU cache keyed by zobrist ---
_EVAL_CACHE = OrderedDict()
_EVAL_CACHE_CAP = 200_000

def _cache_get(k):
    v = _EVAL_CACHE.get(k)
    if v is not None:
        _EVAL_CACHE.move_to_end(k)
    return v

def _cache_put(k, v):
    _EVAL_CACHE[k] = v
    _EVAL_CACHE.move_to_end(k)
    if len(_EVAL_CACHE) > _EVAL_CACHE_CAP:
        _EVAL_CACHE.popitem(last=False)

def evaluate(board: chess.Board) -> float:
    """
    White-perspective evaluation in centipawns (float).
    """
    k = chess.polyglot.zobrist_hash(board)
    v = _cache_get(k)
    if v is not None:
        return v

    tensor = board_to_tensor(board)[np.newaxis, ...]
    # model outputs pawns -> convert to centipawns
    score_cp = float(eval_model.predict(tensor, verbose=0)[0,0] * 100.0)
    # mild clip to keep search sane
    if score_cp > 20000: score_cp = 20000.0
    if score_cp < -20000: score_cp = -20000.0

    _cache_put(k, score_cp)
    return score_cp
