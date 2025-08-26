# engine/eval_model.py
# Single fast evaluator (no TF): tapered classical eval with PSTs, mobility, pawn structure, king safety.
# API stays the same: evaluate(board: chess.Board) -> float (centipawns, White perspective)

import chess
import chess.polyglot
from collections import OrderedDict

# --------------------
# Tunables (centipawns)
# --------------------
P, N, B, R, Q, K = 100, 320, 330, 500, 900, 0

# Game-phase weights (piece phase values; used to blend MG/EG)
PHASE_P, PHASE_N, PHASE_B, PHASE_R, PHASE_Q = 0, 1, 1, 2, 4
MAX_PHASE = 16  # 2N + 2B + 2R*2 + Q*2 = 2 + 2 + 4 + 8 = 16

# Mobility (per available target square)
MOB_N_MG, MOB_N_EG = 4, 3
MOB_B_MG, MOB_B_EG = 5, 4
MOB_R_MG, MOB_R_EG = 2, 2
MOB_Q_MG, MOB_Q_EG = 1, 1

# Pawn structure
PAWN_DOUBLED   = -15
PAWN_ISOLATED  = -12
PAWN_BACKWARD  = -10
PAWN_PASSED_MG = [0, 10, 20, 40, 70, 110, 160, 0]  # rank 1..8
PAWN_PASSED_EG = [0, 15, 30, 60, 100, 160, 240, 0]

# King safety (simple pawn shield in front of king, MG only)
KING_SHIELD_FILE_BONUS = 6
KING_OPEN_FILE_PENALTY = -12

# Tempo bonus
TEMPO = 10

# --------------------
# PSTs (midgame/endgame) — a1=0..h8=63; for Black we mirror with sq^56 to reuse white tables.
# --------------------
# fmt: off
PAWN_PST_MG = [
     0,  0,  0,   0,   0,  0,  0,  0,
    12, 16, 12,  10,  10, 12, 16, 12,
     4,  6,  8,  16,  16,  8,  6,  4,
     2,  4,  6,  10,  10,  6,  4,  2,
     0,  2,  4,   6,   6,  4,  2,  0,
     0,  0,  2,   4,   4,  2,  0,  0,
     0,  0,  0,   0,   0,  0,  0,  0,
     0,  0,  0,   0,   0,  0,  0,  0,
]
PAWN_PST_EG = [
     0,  0,  0,   0,   0,  0,  0,  0,
     8, 10, 12,  14,  14, 12, 10,  8,
     6,  8, 10,  12,  12, 10,  8,  6,
     4,  6,  8,  10,  10,  8,  6,  4,
     2,  4,  6,   8,   8,  6,  4,  2,
     1,  2,  4,   6,   6,  4,  2,  1,
     0,  0,  0,   0,   0,  0,  0,  0,
     0,  0,  0,   0,   0,  0,  0,  0,
]
KNIGHT_PST_MG = [
   -50,-30,-20,-20,-20,-20,-30,-50,
   -30,-10,  0,  0,  0,  0,-10,-30,
   -20,  0, 10, 15, 15, 10,  0,-20,
   -20,  5, 15, 20, 20, 15,  5,-20,
   -20,  0, 15, 20, 20, 15,  0,-20,
   -20,  5, 10, 15, 15, 10,  5,-20,
   -30,-10,  0,  5,  5,  0,-10,-30,
   -50,-30,-20,-20,-20,-20,-30,-50,
]
KNIGHT_PST_EG = [
   -40,-25,-15,-15,-15,-15,-25,-40,
   -25, -8,  0,  0,  0,  0, -8,-25,
   -15,  0,  8, 12, 12,  8,  0,-15,
   -15,  4, 12, 16, 16, 12,  4,-15,
   -15,  0, 12, 16, 16, 12,  0,-15,
   -15,  4,  8, 12, 12,  8,  4,-15,
   -25, -8,  0,  4,  4,  0, -8,-25,
   -40,-25,-15,-15,-15,-15,-25,-40,
]
BISHOP_PST_MG = [
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  0,  0,  5,  5,  0,  0,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10,  5, 10, 15, 15, 10,  5,-10,
   -10,  0, 10, 15, 15, 10,  0,-10,
   -10,  5, 10, 10, 10, 10,  5,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10,-10,-10,-10,-10,-20,
]
BISHOP_PST_EG = [
   -18, -8, -8, -8, -8, -8, -8,-18,
    -8,  0,  0,  4,  4,  0,  0, -8,
    -8,  0,  8,  8,  8,  8,  0, -8,
    -8,  4,  8, 12, 12,  8,  4, -8,
    -8,  0,  8, 12, 12,  8,  0, -8,
    -8,  4,  8,  8,  8,  8,  4, -8,
    -8,  0,  0,  0,  0,  0,  0, -8,
   -18, -8, -8, -8, -8, -8, -8,-18,
]
ROOK_PST_MG = [
     0,  0,  2,  4,  4,  2,  0,  0,
     2,  4,  6,  8,  8,  6,  4,  2,
     0,  2,  4,  6,  6,  4,  2,  0,
     0,  0,  2,  4,  4,  2,  0,  0,
     0,  0,  2,  4,  4,  2,  0,  0,
     0,  2,  4,  6,  6,  4,  2,  0,
     2,  4,  6,  8,  8,  6,  4,  2,
     0,  0,  2,  4,  4,  2,  0,  0,
]
ROOK_PST_EG = [
     0,  0,  1,  2,  2,  1,  0,  0,
     1,  2,  3,  4,  4,  3,  2,  1,
     0,  1,  2,  3,  3,  2,  1,  0,
     0,  0,  1,  2,  2,  1,  0,  0,
     0,  0,  1,  2,  2,  1,  0,  0,
     0,  1,  2,  3,  3,  2,  1,  0,
     1,  2,  3,  4,  4,  3,  2,  1,
     0,  0,  1,  2,  2,  1,  0,  0,
]
QUEEN_PST_MG = [
     0,  0,  0,  2,  2,  0,  0,  0,
     0,  2,  2,  4,  4,  2,  2,  0,
     0,  2,  3,  4,  4,  3,  2,  0,
     2,  4,  4,  6,  6,  4,  4,  2,
     2,  4,  4,  6,  6,  4,  4,  2,
     0,  2,  3,  4,  4,  3,  2,  0,
     0,  2,  2,  4,  4,  2,  2,  0,
     0,  0,  0,  2,  2,  0,  0,  0,
]
QUEEN_PST_EG = [
     0,  0,  0,  1,  1,  0,  0,  0,
     0,  1,  1,  2,  2,  1,  1,  0,
     0,  1,  2,  2,  2,  2,  1,  0,
     1,  2,  2,  3,  3,  2,  2,  1,
     1,  2,  2,  3,  3,  2,  2,  1,
     0,  1,  2,  2,  2,  2,  1,  0,
     0,  1,  1,  2,  2,  1,  1,  0,
     0,  0,  0,  1,  1,  0,  0,  0,
]
KING_PST_MG = [
    20, 30, 10,  0,  0, 10, 30, 20,
    20, 20,  0,  0,  0,  0, 20, 20,
   -10,-10,-20,-20,-20,-20,-10,-10,
   -20,-20,-30,-30,-30,-30,-20,-20,
   -30,-30,-40,-40,-40,-40,-30,-30,
   -30,-30,-40,-40,-40,-40,-30,-30,
   -30,-30,-40,-40,-40,-40,-30,-30,
   -30,-30,-40,-40,-40,-40,-30,-30,
]
KING_PST_EG = [
   -60,-40,-30,-20,-20,-30,-40,-60,
   -40,-20,-10,  0,  0,-10,-20,-40,
   -30,-10,  0, 10, 10,  0,-10,-30,
   -20,  0, 10, 20, 20, 10,  0,-20,
   -20,  0, 10, 20, 20, 10,  0,-20,
   -30,-10,  0, 10, 10,  0,-10,-30,
   -40,-20,-10,  0,  0,-10,-20,-40,
   -60,-40,-30,-20,-20,-30,-40,-60,
]
# fmt: on

def mirror_sq_for_white(sq: int) -> int:
    return sq ^ 56  # mirror vertically so we can reuse the same PST table

def popcount(bb) -> int:
    # Works for both int and SquareSet
    return int(bb).__int__().bit_count() if hasattr(bb, "__int__") else int(bb).bit_count()

# File masks
FILE_MASKS = [chess.BB_FILES[i] for i in range(8)]
ADJ_FILE_MASKS = [0]*8
for f in range(8):
    m = 0
    if f-1 >= 0: m |= FILE_MASKS[f-1]
    if f+1 <= 7: m |= FILE_MASKS[f+1]
    ADJ_FILE_MASKS[f] = m

RANK_OF = [i//8 + 1 for i in range(64)]
FILE_OF = [i%8 for i in range(64)]

# --------------------
# Cache
# --------------------
_EVAL_CACHE: "OrderedDict[int, int]" = OrderedDict()
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

# --------------------------------
# Phase computation for tapered eval
# --------------------------------
def game_phase(board: chess.Board) -> int:
    wN = popcount(board.pieces_mask(chess.KNIGHT, chess.WHITE))
    bN = popcount(board.pieces_mask(chess.KNIGHT, chess.BLACK))
    wB = popcount(board.pieces_mask(chess.BISHOP, chess.WHITE))
    bB = popcount(board.pieces_mask(chess.BISHOP, chess.BLACK))
    wR = popcount(board.pieces_mask(chess.ROOK,   chess.WHITE))
    bR = popcount(board.pieces_mask(chess.ROOK,   chess.BLACK))
    wQ = popcount(board.pieces_mask(chess.QUEEN,  chess.WHITE))
    bQ = popcount(board.pieces_mask(chess.QUEEN,  chess.BLACK))
    phase = (wN+bN)*PHASE_N + (wB+bB)*PHASE_B + (wR+bR)*PHASE_R + (wQ+bQ)*PHASE_Q
    if phase > MAX_PHASE: phase = MAX_PHASE
    return phase

# --------------------------------
# Pawn structure terms
# --------------------------------
def eval_pawns(board: chess.Board, white: bool) -> tuple[int,int]:
    pawns_ss = board.pieces(chess.PAWN, white)          # SquareSet
    pawns = int(pawns_ss)                                # int bitboard
    score_mg = 0
    score_eg = 0
    their_pawns_ss = board.pieces(chess.PAWN, not white)
    their_pawns = int(their_pawns_ss)

    for sq in pawns_ss:
        f = FILE_OF[sq]
        r = RANK_OF[sq] if white else 9 - RANK_OF[sq]

        # doubled
        if popcount(pawns & FILE_MASKS[f]) > 1:
            score_mg += PAWN_DOUBLED
            score_eg += PAWN_DOUBLED

        # isolated (no friendly pawns on adjacent files)
        if (pawns & ADJ_FILE_MASKS[f]) == 0:
            score_mg += PAWN_ISOLATED
            score_eg += PAWN_ISOLATED

        # backward (very simplified): no friendly pawn ahead on same file, and enemy controls advance square
        ahead_sq = sq + (8 if white else -8)
        if 0 <= ahead_sq < 64 and (pawns & (1 << ahead_sq)) == 0:
            if white:
                enemy_attacks = ((their_pawns << 7) & ~int(chess.BB_FILE_H)) | ((their_pawns << 9) & ~int(chess.BB_FILE_A))
            else:
                enemy_attacks = ((their_pawns >> 9) & ~int(chess.BB_FILE_H)) | ((their_pawns >> 7) & ~int(chess.BB_FILE_A))
            if (enemy_attacks & (1 << ahead_sq)) != 0:
                score_mg += PAWN_BACKWARD
                score_eg += PAWN_BACKWARD

        # passed pawn: no enemy pawns ahead on same/adjacent files
        ahead_mask = 0
        if white:
            for rank in range(RANK_OF[sq]+1, 9):
                ahead_mask |= (FILE_MASKS[f] | ADJ_FILE_MASKS[f]) & int(chess.BB_RANKS[rank-1])
        else:
            for rank in range(RANK_OF[sq]-1, 0, -1):
                ahead_mask |= (FILE_MASKS[f] | ADJ_FILE_MASKS[f]) & int(chess.BB_RANKS[rank-1])
        if (their_pawns & ahead_mask) == 0:
            score_mg += PAWN_PASSED_MG[r]
            score_eg += PAWN_PASSED_EG[r]

    return score_mg, score_eg

# --------------------------------
# Mobility (pseudo: attack squares count)
# --------------------------------
def eval_mobility(board: chess.Board, white: bool) -> tuple[int,int]:
    mg = eg = 0
    color = chess.WHITE if white else chess.BLACK
    occ_own = int(board.occupied_co[color])

    # Knights
    for sq in board.pieces(chess.KNIGHT, color):
        moves = popcount(int(board.attacks(sq)) & ~occ_own)
        mg += MOB_N_MG * moves; eg += MOB_N_EG * moves
    # Bishops
    for sq in board.pieces(chess.BISHOP, color):
        moves = popcount(int(board.attacks(sq)) & ~occ_own)
        mg += MOB_B_MG * moves; eg += MOB_B_EG * moves
    # Rooks
    for sq in board.pieces(chess.ROOK, color):
        moves = popcount(int(board.attacks(sq)) & ~occ_own)
        mg += MOB_R_MG * moves; eg += MOB_R_EG * moves
    # Queens
    for sq in board.pieces(chess.QUEEN, color):
        moves = popcount(int(board.attacks(sq)) & ~occ_own)
        mg += MOB_Q_MG * moves; eg += MOB_Q_EG * moves
    return mg, eg

# --------------------------------
# King safety (very light & fast)
# --------------------------------
def eval_king_safety(board: chess.Board, white: bool) -> int:
    color = chess.WHITE if white else chess.BLACK
    k_sq = board.king(color)
    if k_sq is None:
        return 0
    f = FILE_OF[k_sq]

    pawns = int(board.pieces(chess.PAWN, color))
    bonus = 0

    # friendly pawns on same/adjacent files in front half (from color POV)
    files = [f]
    if f-1 >= 0: files.append(f-1)
    if f+1 <= 7: files.append(f+1)
    shield = 0
    for ff in files:
        mask = int(FILE_MASKS[ff])
        if white:
            mask &= int(chess.BB_RANK_2 | chess.BB_RANK_3 | chess.BB_RANK_4)
        else:
            mask &= int(chess.BB_RANK_7 | chess.BB_RANK_6 | chess.BB_RANK_5)
        shield += popcount(pawns & mask)
    bonus += shield * KING_SHIELD_FILE_BONUS

    # open file near king (no friendly pawns) is a bit hazardous
    for ff in files:
        if popcount(pawns & int(FILE_MASKS[ff])) == 0:
            bonus += KING_OPEN_FILE_PENALTY

    return bonus

# --------------------------------
# Core eval (MG/EG tapered)
# --------------------------------
def evaluate_white_cp(board: chess.Board) -> int:
    key = chess.polyglot.zobrist_hash(board)
    v = _cache_get(key)
    if v is not None:
        return v

    # Material
    mat_w = (
        popcount(board.pieces_mask(chess.PAWN,   chess.WHITE)) * P +
        popcount(board.pieces_mask(chess.KNIGHT, chess.WHITE)) * N +
        popcount(board.pieces_mask(chess.BISHOP, chess.WHITE)) * B +
        popcount(board.pieces_mask(chess.ROOK,   chess.WHITE)) * R +
        popcount(board.pieces_mask(chess.QUEEN,  chess.WHITE)) * Q
    )
    mat_b = (
        popcount(board.pieces_mask(chess.PAWN,   chess.BLACK)) * P +
        popcount(board.pieces_mask(chess.KNIGHT, chess.BLACK)) * N +
        popcount(board.pieces_mask(chess.BISHOP, chess.BLACK)) * B +
        popcount(board.pieces_mask(chess.ROOK,   chess.BLACK)) * R +
        popcount(board.pieces_mask(chess.QUEEN,  chess.BLACK)) * Q
    )
    mat_cp = mat_w - mat_b

    # PST and piece activity
    pst_mg_w = pst_eg_w = pst_mg_b = pst_eg_b = 0

    for sq in board.pieces(chess.PAWN, chess.WHITE):
        widx = mirror_sq_for_white(sq); pst_mg_w += PAWN_PST_MG[widx]; pst_eg_w += PAWN_PST_EG[widx]
    for sq in board.pieces(chess.PAWN, chess.BLACK):
        widx = mirror_sq_for_white(sq); pst_mg_b += PAWN_PST_MG[widx]; pst_eg_b += PAWN_PST_EG[widx]

    for sq in board.pieces(chess.KNIGHT, chess.WHITE):
        widx = mirror_sq_for_white(sq); pst_mg_w += KNIGHT_PST_MG[widx]; pst_eg_w += KNIGHT_PST_EG[widx]
    for sq in board.pieces(chess.KNIGHT, chess.BLACK):
        widx = mirror_sq_for_white(sq); pst_mg_b += KNIGHT_PST_MG[widx]; pst_eg_b += KNIGHT_PST_EG[widx]

    for sq in board.pieces(chess.BISHOP, chess.WHITE):
        widx = mirror_sq_for_white(sq); pst_mg_w += BISHOP_PST_MG[widx]; pst_eg_w += BISHOP_PST_EG[widx]
    for sq in board.pieces(chess.BISHOP, chess.BLACK):
        widx = mirror_sq_for_white(sq); pst_mg_b += BISHOP_PST_MG[widx]; pst_eg_b += BISHOP_PST_EG[widx]

    for sq in board.pieces(chess.ROOK, chess.WHITE):
        widx = mirror_sq_for_white(sq); pst_mg_w += ROOK_PST_MG[widx]; pst_eg_w += ROOK_PST_EG[widx]
    for sq in board.pieces(chess.ROOK, chess.BLACK):
        widx = mirror_sq_for_white(sq); pst_mg_b += ROOK_PST_MG[widx]; pst_eg_b += ROOK_PST_EG[widx]

    for sq in board.pieces(chess.QUEEN, chess.WHITE):
        widx = mirror_sq_for_white(sq); pst_mg_w += QUEEN_PST_MG[widx]; pst_eg_w += QUEEN_PST_EG[widx]
    for sq in board.pieces(chess.QUEEN, chess.BLACK):
        widx = mirror_sq_for_white(sq); pst_mg_b += QUEEN_PST_MG[widx]; pst_eg_b += QUEEN_PST_EG[widx]

    for sq in board.pieces(chess.KING, chess.WHITE):
        widx = mirror_sq_for_white(sq); pst_mg_w += KING_PST_MG[widx]; pst_eg_w += KING_PST_EG[widx]
    for sq in board.pieces(chess.KING, chess.BLACK):
        widx = mirror_sq_for_white(sq); pst_mg_b += KING_PST_MG[widx]; pst_eg_b += KING_PST_EG[widx]

    pst_mg = pst_mg_w - pst_mg_b
    pst_eg = pst_eg_w - pst_eg_b

    # Pawn structure
    pawns_mg_w, pawns_eg_w = eval_pawns(board, True)
    pawns_mg_b, pawns_eg_b = eval_pawns(board, False)
    pawn_mg = pawns_mg_w - pawns_mg_b
    pawn_eg = pawns_eg_w - pawns_eg_b

    # Mobility
    mob_mg_w, mob_eg_w = eval_mobility(board, True)
    mob_mg_b, mob_eg_b = eval_mobility(board, False)
    mob_mg = mob_mg_w - mob_mg_b
    mob_eg = mob_eg_w - mob_eg_b

    # King safety
    ks_mg = eval_king_safety(board, True) - eval_king_safety(board, False)
    ks_eg = 0

    # Tapered blend
    phase = game_phase(board)  # 0..MAX_PHASE
    mg_score = mat_cp + pst_mg + pawn_mg + mob_mg + ks_mg + TEMPO
    eg_score = mat_cp + pst_eg + pawn_eg + mob_eg + ks_eg + TEMPO
    score = (mg_score * phase + eg_score * (MAX_PHASE - phase)) // max(1, MAX_PHASE)

    _cache_put(key, int(score))
    return int(score)

def evaluate(board: chess.Board) -> float:
    """Public API: return evaluation in centipawns from White's perspective (float)."""
    return float(evaluate_white_cp(board))
