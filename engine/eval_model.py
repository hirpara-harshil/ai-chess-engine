import numpy as np
import chess, chess.polyglot
from tensorflow.keras.models import load_model
from collections import OrderedDict

MODEL_PATH = "data/processed/eval_model.keras"
eval_model = load_model(MODEL_PATH)
print(f"✅ Loaded evaluation model from {MODEL_PATH}")

# warm up once (eager call, not .predict)
try:
    _ = eval_model(np.zeros((1,8,8,12), dtype=np.float32), training=False)
    print("⚡ Model warm-up complete.")
except Exception:
    pass

PIECE_TO_PLANE = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
}

def board_to_tensor(board: chess.Board) -> np.ndarray:
    tensor = np.zeros((8,8,12), dtype=np.float32)
    for square, piece in board.piece_map().items():
        r = 7 - (square // 8); c = square % 8
        plane = PIECE_TO_PLANE[piece.piece_type] + (6 if piece.color == chess.BLACK else 0)
        tensor[r, c, plane] = 1.0
    return tensor

# tiny LRU cache
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

    x = board_to_tensor(board)[np.newaxis, ...]
    # eager call is lighter than .predict()
    out = eval_model(x, training=False).numpy()[0,0]
    score_cp = float(out * 100.0)
    _cache_put(k, score_cp)
    return score_cp
