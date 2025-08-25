# engine/search.py
# Negamax (alpha-beta) + quiescence search, simple move ordering.
# Returns centipawn scores (White positive). Public API: find_best_move(board, depth, eval_fn)

import chess
import time
from typing import Optional, Tuple
from engine import eval as engine_eval  # fallback if eval_fn not provided

# Simple piece values for MVV-LVA ordering (used for sorting captures)
MVV_LVA_VALUE = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   10000
}

# Quiescence search depth limit (in plies of capture sequence)
MAX_Q_DEPTH = 10

def _is_capture(board: chess.Board, move: chess.Move) -> bool:
    # python-chess provides board.is_capture(move)
    return board.is_capture(move)

def _capture_score(board: chess.Board, move: chess.Move) -> int:
    # Higher score if capturing more valuable piece with less valuable piece.
    # Try to find captured piece type. Handle en-passant separately.
    if board.is_en_passant(move):
        captured_value = MVV_LVA_VALUE[chess.PAWN]
    else:
        captured = board.piece_at(move.to_square)
        captured_value = MVV_LVA_VALUE[captured.piece_type] if captured else 0

    mover = board.piece_at(move.from_square)
    mover_value = MVV_LVA_VALUE[mover.piece_type] if mover else 0

    # MVV-LVA: reward high captured_value and lower mover_value
    return (captured_value * 100) - mover_value

def _order_moves(board: chess.Board):
    """
    Return moves ordered: captures (MVV-LVA) first (descending), then quiet moves.
    """
    moves = list(board.legal_moves)
    def key(m):
        if _is_capture(board, m):
            return (2, _capture_score(board, m))
        # prefer promotions too
        if m.promotion:
            return (1, MVV_LVA_VALUE.get(m.promotion, 0))
        return (0, 0)
    moves.sort(key=key, reverse=True)
    return moves

# -------------------------
# Quiescence search (captures-only)
# -------------------------
def quiescence(board: chess.Board, alpha: int, beta: int, eval_fn, depth=0) -> int:
    """
    Standalone quiescence: evaluate and then explore captures while improving alpha.
    Returns score in centipawns (White perspective).
    """
    # Stand pat
    stand_pat = eval_fn(board)
    if stand_pat >= beta:
        return stand_pat
    if alpha < stand_pat:
        alpha = stand_pat

    if depth >= MAX_Q_DEPTH:
        return stand_pat

    # Explore captures only
    for move in _order_moves(board):
        if not _is_capture(board, move):
            continue
        board.push(move)
        score = -quiescence(board, -beta, -alpha, eval_fn, depth + 1)
        board.pop()

        if score >= beta:
            return score
        if score > alpha:
            alpha = score

    return alpha

# -------------------------
# Negamax with alpha-beta
# -------------------------
def negamax(board: chess.Board, depth: int, alpha: int, beta: int, eval_fn) -> int:
    """
    Negamax returns value in centipawns from the current side-to-move perspective,
    but uses evaluate(board) which is White-centric. We'll convert by sign as needed:
    We will return score always in White's perspective for consistency with eval.evaluate.
    Implementation detail: we use standard negamax with sign flip.
    """
    # Terminal conditions
    if board.is_checkmate():
        # If side to move is checkmated, it's large negative from White perspective if White to move, etc.
        return -100000 if board.turn == chess.WHITE else 100000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    if depth <= 0:
        # call quiescence to avoid horizon effect
        return quiescence(board, alpha, beta, eval_fn)

    # Move ordering
    best_score = -999999
    for move in _order_moves(board):
        board.push(move)
        val = -negamax(board, depth - 1, -beta, -alpha, eval_fn)
        board.pop()

        if val > best_score:
            best_score = val
        if val > alpha:
            alpha = val
        if alpha >= beta:
            break  # beta cutoff

    # If no legal moves (should be handled by checkmate/stalemate above)
    if best_score == -999999:
        # no moves: either mate or stalemate, already handled but safe fallback
        return 0
    return best_score

# -------------------------
# Root search (returns best move)
# -------------------------
def find_best_move(board: chess.Board, depth: int = 3, eval_fn=None, time_limit: Optional[float]=None) -> Optional[chess.Move]:
    """
    Public routine. Search to fixed depth (plies). Returns best chess.Move or None.
    - board: current python-chess Board (will not be mutated)
    - depth: positive integer (ply)
    - eval_fn: function(board) -> score in centipawns (White positive). If None, uses engine.eval.evaluate.
    - time_limit: optional seconds; if set, best found so far is returned when time expires (simple timer).
    """
    if eval_fn is None:
        eval_fn = engine_eval.evaluate

    start_time = time.time()
    deadline = start_time + time_limit if time_limit else None

    best_move = None
    best_score = -999999

    # root move ordering initially
    moves = _order_moves(board)

    for move in moves:
        # time check
        if deadline and time.time() > deadline:
            break

        board.push(move)
        score = -negamax(board, depth - 1, -999999, 999999, eval_fn)
        board.pop()

        # score is in White-centric centipawns
        if score > best_score:
            best_score = score
            best_move = move

    return best_move
