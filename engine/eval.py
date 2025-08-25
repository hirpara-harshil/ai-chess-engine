# engine/eval.py
# Evaluation function for python-chess Board.
# Returns centipawns from White's perspective (+ better for White).

import chess
from functools import lru_cache

# --------------------------
# Piece base values (cp)
# --------------------------
PIECE_VALUE = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   0,    # handled by PST/safety
}

# --------------------------
# Game phase (tapered eval)
# weights approximate impact on phase
# --------------------------
PHASE_WEIGHTS = {
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK:   2,
    chess.QUEEN:  4,
}
# initial total phase = 24 (4N*1 + 4B*1 + 4R*2 + 2Q*4)
MAX_PHASE = 24

# --------------------------
# Piece-Square Tables (mg/eg)
# Values are oriented for White; mirror for Black.
# Units: centipawns. Keep numbers modest; king PSTs are phase-blended with safety.
# --------------------------
# Pawn (mg/eg)
PAWN_MG = [
     0,   2,   2,  -6,  -6,   2,   2,   0,
     0,   6,  10,  12,  12,  10,   6,   0,
     4,   8,  14,  18,  18,  14,   8,   4,
     8,  12,  20,  26,  26,  20,  12,   8,
    10,  16,  24,  28,  28,  24,  16,  10,
    10,  18,  22,  24,  24,  22,  18,  10,
     8,  10,  12,  16,  16,  12,  10,   8,
     0,   0,   0,   0,   0,   0,   0,   0,
]
PAWN_EG = [
     0,   6,   8,  10,  10,   8,   6,   0,
     6,  12,  16,  18,  18,  16,  12,   6,
     8,  16,  20,  24,  24,  20,  16,   8,
    10,  18,  22,  26,  26,  22,  18,  10,
    10,  16,  20,  22,  22,  20,  16,  10,
     6,  10,  14,  16,  16,  14,  10,   6,
     4,   6,   8,  10,  10,   8,   6,   4,
     0,   0,   0,   0,   0,   0,   0,   0,
]

# Knight (mg/eg)
KNIGHT_MG = [
   -60, -40, -24, -20, -20, -24, -40, -60,
   -40, -16,  -8,  -4,  -4,  -8, -16, -40,
   -24,  -8,  10,  14,  14,  10,  -8, -24,
   -20,  -4,  14,  24,  24,  14,  -4, -20,
   -20,  -2,  14,  26,  26,  14,  -2, -20,
   -24,  -4,  10,  18,  18,  10,  -4, -24,
   -40, -16,  -4,   0,   0,  -4, -16, -40,
   -60, -40, -24, -18, -18, -24, -40, -60,
]
KNIGHT_EG = [
   -50, -30, -20, -12, -12, -20, -30, -50,
   -30, -10,  -4,   0,   0,  -4, -10, -30,
   -20,  -4,   8,  12,  12,   8,  -4, -20,
   -12,   0,  12,  18,  18,  12,   0, -12,
   -12,   0,  12,  18,  18,  12,   0, -12,
   -20,  -4,   8,  12,  12,   8,  -4, -20,
   -30, -10,  -4,   0,   0,  -4, -10, -30,
   -50, -30, -20, -12, -12, -20, -30, -50,
]

# Bishop (mg/eg)
BISHOP_MG = [
   -20, -12, -10,  -8,  -8, -10, -12, -20,
   -10,  -4,   0,   6,   6,   0,  -4, -10,
    -8,   0,  10,  14,  14,  10,   0,  -8,
    -6,   4,  14,  18,  18,  14,   4,  -6,
    -6,   4,  12,  18,  18,  12,   4,  -6,
    -8,   0,  10,  14,  14,  10,   0,  -8,
   -10,  -4,   0,   6,   6,   0,  -4, -10,
   -20, -12, -10,  -8,  -8, -10, -12, -20,
]
BISHOP_EG = [
   -10,  -6,  -4,  -2,  -2,  -4,  -6, -10,
    -6,   0,   2,   6,   6,   2,   0,  -6,
    -4,   2,   8,  12,  12,   8,   2,  -4,
    -2,   6,  12,  16,  16,  12,   6,  -2,
    -2,   6,  12,  16,  16,  12,   6,  -2,
    -4,   2,   8,  12,  12,   8,   2,  -4,
    -6,   0,   2,   6,   6,   2,   0,  -6,
   -10,  -6,  -4,  -2,  -2,  -4,  -6, -10,
]

# Rook (mg/eg)
ROOK_MG = [
     4,   6,   8,  10,  10,   8,   6,   4,
     6,   8,  10,  12,  12,  10,   8,   6,
     4,   6,   8,  10,  10,   8,   6,   4,
     2,   4,   6,   8,   8,   6,   4,   2,
     2,   4,   6,   8,   8,   6,   4,   2,
     0,   2,   4,   6,   6,   4,   2,   0,
    -2,   0,   2,   4,   4,   2,   0,  -2,
    -4,  -2,   0,   2,   2,   0,  -2,  -4,
]
ROOK_EG = [
     2,   4,   6,   8,   8,   6,   4,   2,
     4,   6,   8,  10,  10,   8,   6,   4,
     4,   6,   8,  10,  10,   8,   6,   4,
     2,   4,   6,   8,   8,   6,   4,   2,
     0,   2,   4,   6,   6,   4,   2,   0,
    -2,   0,   2,   4,   4,   2,   0,  -2,
    -4,  -2,   0,   2,   2,   0,  -2,  -4,
    -6,  -4,  -2,   0,   0,  -2,  -4,  -6,
]

# Queen (mg/eg)
QUEEN_MG = [
   -10,  -6,  -2,   0,   0,  -2,  -6, -10,
    -6,  -2,   2,   4,   4,   2,  -2,  -6,
    -2,   2,   6,   8,   8,   6,   2,  -2,
     0,   4,   8,  12,  12,   8,   4,   0,
     0,   4,   8,  12,  12,   8,   4,   0,
    -2,   2,   6,   8,   8,   6,   2,  -2,
    -6,  -2,   2,   4,   4,   2,  -2,  -6,
   -10,  -6,  -2,   0,   0,  -2,  -6, -10,
]
QUEEN_EG = [
   -10,  -8,  -6,  -4,  -4,  -6,  -8, -10,
    -8,  -6,  -2,   0,   0,  -2,  -6,  -8,
    -6,  -2,   2,   4,   4,   2,  -2,  -6,
    -4,   0,   4,   8,   8,   4,   0,  -4,
    -4,   0,   4,   8,   8,   4,   0,  -4,
    -6,  -2,   2,   4,   4,   2,  -2,  -6,
    -8,  -6,  -2,   0,   0,  -2,  -6,  -8,
   -10,  -8,  -6,  -4,  -4,  -6,  -8, -10,
]

# King (mg/eg): safer behind pawns (mg), centralized in eg.
KING_MG = [
    20,  24,  10,   0,   0,  10,  24,  20,
    18,  10,   0, -10, -10,   0,  10,  18,
     8,   0, -12, -18, -18, -12,   0,   8,
     4,  -8, -18, -24, -24, -18,  -8,   4,
     4,  -8, -18, -24, -24, -18,  -8,   4,
     8,   0, -12, -18, -18, -12,   0,   8,
    12,   6,  -2, -10, -10,  -2,   6,  12,
    24,  16,   8,  -2,  -2,   8,  16,  24,
]
KING_EG = [
   -20, -12,  -6,  -2,  -2,  -6, -12, -20,
   -12,  -6,   0,   6,   6,   0,  -6, -12,
    -6,   0,  10,  16,  16,  10,   0,  -6,
    -2,   6,  16,  22,  22,  16,   6,  -2,
    -2,   6,  16,  22,  22,  16,   6,  -2,
    -6,   0,  10,  16,  16,  10,   0,  -6,
   -12,  -6,   0,   6,   6,   0,  -6, -12,
   -20, -12,  -6,  -2,  -2,  -6, -12, -20,
]

PSQT_MG = {
    chess.PAWN:   PAWN_MG,
    chess.KNIGHT: KNIGHT_MG,
    chess.BISHOP: BISHOP_MG,
    chess.ROOK:   ROOK_MG,
    chess.QUEEN:  QUEEN_MG,
    chess.KING:   KING_MG,
}
PSQT_EG = {
    chess.PAWN:   PAWN_EG,
    chess.KNIGHT: KNIGHT_EG,
    chess.BISHOP: BISHOP_EG,
    chess.ROOK:   ROOK_EG,
    chess.QUEEN:  QUEEN_EG,
    chess.KING:   KING_EG,
}

# --------------------------
# Helper: game phase 0..1
# --------------------------
def game_phase(board: chess.Board) -> float:
    phase = 0
    for pt, w in PHASE_WEIGHTS.items():
        phase += w * (len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK)))
    # phase in [0..MAX_PHASE]; normalize to [0..1] where 1=opening, 0=endgame
    return phase / MAX_PHASE if MAX_PHASE else 0.0

# --------------------------
# Material score
# --------------------------
def material_score(board: chess.Board) -> int:
    score = 0
    for pt, val in PIECE_VALUE.items():
        score += val * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))
    return score

# --------------------------
# PST (tapered)
# --------------------------
def psqt_score(board: chess.Board, phase_w: float) -> int:
    mg_s, eg_s = 0, 0
    for color in [chess.WHITE, chess.BLACK]:
        sign = 1 if color == chess.WHITE else -1
        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
            table_mg = PSQT_MG[pt]
            table_eg = PSQT_EG[pt]
            for sq in board.pieces(pt, color):
                idx = sq if color == chess.WHITE else chess.square_mirror(sq)
                mg_s += sign * table_mg[idx]
                eg_s += sign * table_eg[idx]
    # Taper
    return int(round(phase_w * mg_s + (1 - phase_w) * eg_s))

# --------------------------
# Mobility (legal moves count per side)
# Clipped to avoid dominance; scaled modestly.
# --------------------------
def mobility_score(board: chess.Board) -> int:
    def count_moves(b: chess.Board, color: bool) -> int:
        c = b.copy(stack=False)
        c.turn = color
        return sum(1 for _ in c.legal_moves)

    w = count_moves(board, chess.WHITE)
    b = count_moves(board, chess.BLACK)
    w = min(w, 40)
    b = min(b, 40)
    return 2 * (w - b)  # 2 cp per move difference

# --------------------------
# Pawn structure: doubled, isolated, passed
# --------------------------
# FILES = [chess.FILE_A, chess.FILE_B, chess.FILE_C, chess.FILE_D,
        #  chess.FILE_E, chess.FILE_F, chess.FILE_G, chess.FILE_H]
FILES = range(8)  # files a..h as 0..7


def pawn_structure_score(board: chess.Board) -> int:
    score = 0

    for color in [chess.WHITE, chess.BLACK]:
        sign = 1 if color == chess.WHITE else -1
        pawns = list(board.pieces(chess.PAWN, color))
        pawn_files = {chess.square_file(sq) for sq in pawns}

        # Doubled pawns
        for f in FILES:
            cnt = sum(1 for sq in pawns if chess.square_file(sq) == f)
            if cnt > 1:
                score += sign * (-12 * (cnt - 1))

        # Isolated pawns (no friendly pawns on adjacent files)
        for sq in pawns:
            f = chess.square_file(sq)
            adj = {f-1, f+1}
            if not any((af in pawn_files) for af in adj if 0 <= af < 8):
                score += sign * (-10)

        # Passed pawns (no enemy pawns ahead on same/adj files)
        enemy_pawns = board.pieces(chess.PAWN, not color)
        for sq in pawns:
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            passed = True
            for df in (-1, 0, 1):
                nf = f + df
                if 0 <= nf < 8:
                    for er in (range(r+1, 8) if color == chess.WHITE else range(r-1, -1, -1)):
                        if chess.square(nf, er) in enemy_pawns:
                            passed = False
                            break
                    if not passed:
                        break
            if passed:
                # Stronger in endgame; we keep modest base here; king tropism will help too.
                rank_bonus = (r - 1) if color == chess.WHITE else (6 - r)
                score += sign * (20 + 4 * max(0, rank_bonus))

    return score

# --------------------------
# Bishop pair
# --------------------------
def bishop_pair_score(board: chess.Board) -> int:
    score = 0
    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        score += 30
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        score -= 30
    return score

# --------------------------
# Rooks on open/semi-open files
# --------------------------
def rook_file_score(board: chess.Board) -> int:
    score = 0
    white_pawns = board.pieces(chess.PAWN, chess.WHITE)
    black_pawns = board.pieces(chess.PAWN, chess.BLACK)

    for color in [chess.WHITE, chess.BLACK]:
        rooks = board.pieces(chess.ROOK, color)
        sign = 1 if color == chess.WHITE else -1
        for sq in rooks:
            f = chess.square_file(sq)
            file_sqs = [chess.square(f, r) for r in range(8)]
            own_pawn_on_file = any(s in (white_pawns if color == chess.WHITE else black_pawns) for s in file_sqs)
            opp_pawn_on_file = any(s in (black_pawns if color == chess.WHITE else white_pawns) for s in file_sqs)
            if not own_pawn_on_file and not opp_pawn_on_file:
                score += sign * 18   # open file
            elif not own_pawn_on_file and opp_pawn_on_file:
                score += sign * 10   # semi-open
    return score

# --------------------------
# Knight outposts (simple): knight on 4th-6th ranks, not attackable by enemy pawns
# --------------------------
def knight_outpost_score(board: chess.Board) -> int:
    score = 0
    for color in [chess.WHITE, chess.BLACK]:
        sign = 1 if color == chess.WHITE else -1
        enemy_pawns = board.pieces(chess.PAWN, not color)

        for sq in board.pieces(chess.KNIGHT, color):
            r = chess.square_rank(sq)
            if color == chess.WHITE and r < 3:
                continue
            if color == chess.BLACK and r > 4:
                continue
            # attacked by enemy pawn?
            attacked_by_enemy_pawn = False
            f = chess.square_file(sq)
            for df in (-1, 1):
                nf = f + df
                if 0 <= nf < 8:
                    target_rank = r - 1 if color == chess.WHITE else r + 1
                    if 0 <= target_rank < 8 and chess.square(nf, target_rank) in enemy_pawns:
                        attacked_by_enemy_pawn = True
                        break
            if not attacked_by_enemy_pawn:
                score += sign * 16
    return score

# --------------------------
# Center control (d4, e4, d5, e5 primary; extended center small)
# --------------------------
CENTER_SQS = [chess.D4, chess.E4, chess.D5, chess.E5]
EXT_CENTER = [
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.F4, chess.C5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6
]

def center_control_score(board: chess.Board) -> int:
    score = 0
    for sq in CENTER_SQS:
        w = len(board.attackers(chess.WHITE, sq))
        b = len(board.attackers(chess.BLACK, sq))
        score += 3 * (w - b)
    for sq in EXT_CENTER:
        w = len(board.attackers(chess.WHITE, sq))
        b = len(board.attackers(chess.BLACK, sq))
        score += 1 * (w - b)
    return score

# --------------------------
# King safety: (1) pawn shield; (2) tropism of enemy pieces toward king
# --------------------------
def king_safety_score(board: chess.Board) -> int:
    score = 0
    score += pawn_shield_term(board, chess.WHITE) - pawn_shield_term(board, chess.BLACK)
    score += king_tropism_term(board, chess.WHITE) - king_tropism_term(board, chess.BLACK)
    return score

def pawn_shield_term(board: chess.Board, color: bool) -> int:
    ksq = board.king(color)
    if ksq is None:
        return 0
    f = chess.square_file(ksq)
    r = chess.square_rank(ksq)

    # inspect files (f-1, f, f+1) 1–3 ranks in front of king
    score = 0
    for df in (-1, 0, 1):
        nf = f + df
        if not (0 <= nf < 8):
            continue
        # ranks in front depending on color
        ranks = range(r+1, min(r+4, 8)) if color == chess.WHITE else range(max(r-3, 0), r)
        shield_found = 0
        for rr in ranks:
            if chess.square(nf, rr) in board.pieces(chess.PAWN, color):
                shield_found = 1
                break
        score += 10 * shield_found  # 0/10 per file
    return score

def king_tropism_term(board: chess.Board, color: bool) -> int:
    """Attraction of enemy heavy/minor pieces to our king (penalty)."""
    my_king = board.king(color)
    if my_king is None:
        return 0
    my_kf, my_kr = chess.square_file(my_king), chess.square_rank(my_king)

    penalty = 0
    enemy = not color
    weights = {
        chess.QUEEN: 3,
        chess.ROOK:  2,
        chess.BISHOP:1,
        chess.KNIGHT:1,
    }
    for pt, w in weights.items():
        for sq in board.pieces(pt, enemy):
            f, r = chess.square_file(sq), chess.square_rank(sq)
            dist = abs(f - my_kf) + abs(r - my_kr)  # Manhattan
            penalty += max(0, (7 - dist)) * w  # closer → bigger
    # Convert to centipawn penalty (cap)
    return -min(penalty, 60)

# --------------------------
# Tempo (side to move bonus)
# --------------------------
def tempo_score(board: chess.Board) -> int:
    return 8 if board.turn == chess.WHITE else -8

# --------------------------
# Drawishness: if claimable draw soon, move eval toward zero slightly.
# (Keep small to avoid hiding winning lines.)
# --------------------------
def soft_draw_pull(board: chess.Board) -> int:
    pull = 0
    if board.can_claim_threefold_repetition() or board.can_claim_fifty_moves():
        pull = -10  # toward zero; applied symmetrically via sign later
    # from White POV, just return pull (already symmetric)
    return pull

# --------------------------
# Top-level evaluate(board) -> int (centipawns)
# --------------------------
def evaluate(board: chess.Board) -> int:
    # Terminal checks are typically handled in search. Keep eval pure.
    # But if you want safety:
    if board.is_checkmate():
        # If side to move is checkmated, it's bad for that side.
        return  -100000 if board.turn == chess.WHITE else 100000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    phase_w = game_phase(board)

    score  = 0
    score += material_score(board)
    score += psqt_score(board, phase_w)
    score += mobility_score(board)
    score += pawn_structure_score(board)
    score += bishop_pair_score(board)
    score += rook_file_score(board)
    score += knight_outpost_score(board)
    score += center_control_score(board)
    score += king_safety_score(board)
    score += tempo_score(board)
    score += soft_draw_pull(board)

    return int(score)
