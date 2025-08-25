# engine/utils.py
import numpy as np
import chess

# 12 planes: white P,N,B,R,Q,K then black p,n,b,r,q,k (order consistent)
PIECE_TO_PLANE = {
    (chess.PAWN, True):  0,  (chess.KNIGHT, True):  1, (chess.BISHOP, True):  2,
    (chess.ROOK, True):  3,  (chess.QUEEN, True):  4, (chess.KING, True):   5,
    (chess.PAWN, False): 6,  (chess.KNIGHT, False): 7, (chess.BISHOP, False): 8,
    (chess.ROOK, False): 9,  (chess.QUEEN, False):10, (chess.KING, False):  11,
}

def board_to_tensor(board: chess.Board, extras: bool = True) -> np.ndarray:
    """
    Convert python-chess Board -> 1D numpy float32 tensor.
    Default output shape: (768 + extras,) where extras ≈ 6 (turn, castling rights, ep file).
    Layout:
      - planes: 12 x 64 flattened (plane0 sq0..sq63, plane1 sq0..sq63, ...)
      - extras: [side_to_move, w_k_castle, w_q_castle, b_k_castle, b_q_castle, ep_file_normalized]
    Squares follow python-chess 0..63 indexing (a1=0 .. h8=63).
    """
    planes = np.zeros((12, 64), dtype=np.uint8)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        plane = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
        planes[plane, sq] = 1

    flat = planes.reshape(-1).astype(np.float32)  # 768

    if not extras:
        return flat

    ep = board.ep_square
    ep_norm = (ep / 63.0) if ep is not None else -1.0

    extras_arr = np.array([
        1.0 if board.turn == chess.WHITE else 0.0,
        1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0,
        1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0,
        1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0,
        1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0,
        ep_norm
    ], dtype=np.float32)

    return np.concatenate([flat, extras_arr], axis=0)  # shape (774,)
