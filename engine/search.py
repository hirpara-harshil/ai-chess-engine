# engine/search.py
# Stronger search: TT (zobrist, 4-way), PVS, LMR, Null-Move, Aspiration, killers/history.
# + strict time control checks in both full search and quiescence.
import time, math
import chess, chess.polyglot
from typing import Optional

MATE_SCORE = 100000
INF = 10**9

# --- Simple MVV-LVA for captures ---
PIECE_CP = {None:0, chess.PAWN:100, chess.KNIGHT:320, chess.BISHOP:330, chess.ROOK:500, chess.QUEEN:900, chess.KING:10000}
def mvv_lva(board: chess.Board, m: chess.Move) -> int:
    if board.is_en_passant(m):
        vic = chess.PAWN
    else:
        p = board.piece_type_at(m.to_square)
        vic = p if p else None
    atk = board.piece_type_at(m.from_square)
    return 100*PIECE_CP[vic] - PIECE_CP[atk]

def is_tactical(board: chess.Board, m: chess.Move) -> bool:
    return board.is_capture(m) or m.promotion is not None or board.gives_check(m)

# --- Transposition table ---
EXACT, LOWER, UPPER = 0, 1, 2

class TTEntry:
    __slots__ = ("key","depth","score","flag","move","age","static_eval")
    def __init__(self, key, depth, score, flag, move, age, static_eval):
        self.key, self.depth, self.score, self.flag = key, depth, score, flag
        self.move, self.age, self.static_eval = move, age, static_eval

class TranspositionTable:
    # fixed-size, 4-way set associative
    def __init__(self, mb: int = 64):
        n_entries = max(4096, (mb * 1024 * 1024) // 32)
        self.ways = 4
        self.size = n_entries // self.ways
        self.table = [[] for _ in range(self.size)]
        self.age = 0

    def new_age(self): self.age = (self.age + 1) & 255
    def _idx(self, k: int) -> int: return k % self.size

    def probe(self, k: int) -> Optional[TTEntry]:
        b = self.table[self._idx(k)]
        for e in b:
            if e.key == k:
                return e
        return None

    def store(self, k: int, depth: int, score: int, flag: int, move: Optional[chess.Move], static_eval: Optional[int]):
        b = self.table[self._idx(k)]
        e = TTEntry(k, depth, score, flag, move, self.age, static_eval)
        if len(b) < self.ways:
            b.append(e); return
        # Replace the shallowest; break ties by oldest
        r = min(range(self.ways), key=lambda i:(b[i].depth, b[i].age))
        b[r] = e

# --- Searcher ---
class Searcher:
    def __init__(self, eval_white_fn, tt_mb=128):
        """
        eval_white_fn(board)->cp from White's perspective (your current Keras eval).
        """
        self.eval_white = eval_white_fn
        self.tt = TranspositionTable(tt_mb)
        self.nodes = 0
        self.abort = False
        self.start = 0.0
        self.deadline = 0.0
        self.killers = [[None, None] for _ in range(256)]
        self.history = {}

    # ---- time control helper ----
    def _tick(self):
        if self.deadline and time.time() >= self.deadline:
            self.abort = True
            return True
        return False

    # side-to-move eval from White-centric eval
    def static_eval(self, board: chess.Board) -> int:
        s = self.eval_white(board)
        return int(s if board.turn == chess.WHITE else -s)

    def order_moves(self, board: chess.Board, moves, tt_move, ply):
        k1, k2 = self.killers[ply]
        def key(m: chess.Move):
            if tt_move and m == tt_move: return 10_000_000
            if board.is_capture(m):      return 1_000_000 + mvv_lva(board, m)
            if k1 and m == k1:           return 900_000
            if k2 and m == k2:           return 800_000
            return self.history.get(m.uci(), 0)
        return sorted(moves, key=key, reverse=True)

    def qsearch(self, board: chess.Board, alpha: int, beta: int) -> int:
        # frequent time checks in quiescence too
        if self._tick(): return 0

        self.nodes += 1
        # stand-pat
        stand = self.static_eval(board)
        if stand >= beta: return beta
        if alpha < stand: alpha = stand

        # only examine captures
        for m in self.order_moves(board, [m for m in board.legal_moves if board.is_capture(m)], None, 0):
            if self._tick(): return alpha
            board.push(m)
            score = -self.qsearch(board, -beta, -alpha)
            board.pop()
            if score >= beta: return beta
            if score > alpha: alpha = score
        return alpha

    def can_null(self, board: chess.Board) -> bool:
        if board.is_check(): return False
        # don't null with too little material
        minors_majors = 0
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            minors_majors += len(board.pieces(pt, True)) + len(board.pieces(pt, False))
        pawns = len(board.pieces(chess.PAWN, True)) + len(board.pieces(chess.PAWN, False))
        return (minors_majors + pawns) > 6

    def search(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int, in_pv: bool) -> int:
        if self._tick(): return 0

        self.nodes += 1
        orig_alpha = alpha
        k = chess.polyglot.zobrist_hash(board)

        # Draws
        if board.is_repetition(3) or board.can_claim_fifty_moves():
            return 0

        # TT probe
        tte = self.tt.probe(k)
        if tte and tte.depth >= depth:
            if tte.flag == EXACT: return tte.score
            if tte.flag == LOWER and tte.score >= beta:  return tte.score
            if tte.flag == UPPER and tte.score <= alpha: return tte.score

        # leaf / terminal
        if depth <= 0:
            val = self.qsearch(board, alpha, beta)
            # store static eval for move ordering at parent
            self.tt.store(k, 0, val, EXACT, tte.move if tte else None, static_eval=self.static_eval(board))
            return val

        if board.is_checkmate():
            return -MATE_SCORE + ply
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        # Null-move pruning
        if not in_pv and depth >= 3 and self.can_null(board):
            board.push(chess.Move.null())
            R = 2 + (depth // 6)
            score = -self.search(board, depth - 1 - R, -beta, -beta + 1, ply+1, in_pv=False)
            board.pop()
            if self.abort: return 0
            if score >= beta:
                # verification
                ver = -self.search(board, depth - 1, -beta, -beta + 1, ply+1, in_pv=False)
                if self.abort: return 0
                if ver >= beta:
                    self.tt.store(k, depth, ver, LOWER, chess.Move.null(), None)
                    return ver

        # Move ordering
        tt_move = (tte.move if tte else None)
        moves = self.order_moves(board, list(board.legal_moves), tt_move, ply)
        if not moves:
            return -MATE_SCORE + ply if board.is_check() else 0

        best = -INF
        best_move = None
        first = True

        for i, m in enumerate(moves):
            if self._tick(): return best if best != -INF else 0

            board.push(m)

            # LMR: reduce late, quiet moves outside PV, depth>=3
            do_lmr = (not in_pv) and depth >= 3 and not is_tactical(board, m) and not board.is_check()
            reduction = 0
            if do_lmr:
                # simple tunable reduction
                reduction = int(0.75 + math.log(depth+1, 2) * math.log(i+2, 2))
                reduction = min(reduction, depth-1)

            if first:
                score = -self.search(board, depth - 1, -beta, -alpha, ply+1, in_pv=in_pv)
                first = False
            else:
                # PVS: try narrow with reduction
                score = -self.search(board, depth - 1 - reduction, -alpha-1, -alpha, ply+1, in_pv=False)
                if self.abort:
                    board.pop(); return best if best != -INF else 0
                if score > alpha and reduction > 0:
                    score = -self.search(board, depth - 1, -alpha-1, -alpha, ply+1, in_pv=False)
                if score > alpha and score < beta:
                    score = -self.search(board, depth - 1, -beta, -alpha, ply+1, in_pv=True)

            board.pop()
            if self.abort: return best if best != -INF else 0

            if score > best:
                best = score
                best_move = m
                if score > alpha:
                    alpha = score
                    if not is_tactical(board, m):
                        self.history[m.uci()] = self.history.get(m.uci(), 0) + depth*depth
                if alpha >= beta:
                    # killer update
                    if not is_tactical(board, m):
                        k1, k2 = self.killers[ply]
                        if m != k1:
                            self.killers[ply] = [m, k1]
                    break

        flag = EXACT if best > orig_alpha and best < beta else (LOWER if best >= beta else UPPER)
        # try to keep a static_eval around for parents to use in ordering
        static_eval = None
        if tte and tte.static_eval is not None:
            static_eval = tte.static_eval
        else:
            # cheap: one call for current node (white-centric -> stm)
            static_eval = self.static_eval(board)
        self.tt.store(k, depth, best, flag, best_move, static_eval)
        return best

    def think(self, board: chess.Board, max_depth=64, time_limit: Optional[float]=None):
        """Iterative deepening with aspiration windows. time_limit in seconds (optional)."""
        self.nodes = 0
        self.abort = False
        self.tt.new_age()
        self.start = time.time()
        self.deadline = self.start + time_limit if time_limit else 0.0

        best_move = None
        last_score = 0
        window = 50  # cp
        for d in range(1, max_depth+1):
            if self._tick(): break
            alpha = last_score - window
            beta  = last_score + window
            while True:
                sc = self.search(board, d, alpha, beta, ply=0, in_pv=True)
                if self.abort: break
                if sc <= alpha:
                    alpha -= window * 2
                elif sc >= beta:
                    beta  += window * 2
                else:
                    last_score = sc
                    break
            if self.abort: break
            tte = self.tt.probe(chess.polyglot.zobrist_hash(board))
            if tte and tte.move:
                best_move = tte.move
        return best_move, last_score

# --- Public API (keeps your signature) ---
def find_best_move(board: chess.Board, depth: int = 3, eval_fn=None, time_limit: Optional[float]=None) -> Optional[chess.Move]:
    """
    Returns best move (or None). Depth=plies; time_limit (s) optional.
    eval_fn is White-centric cp; we flip for side-to-move internally.
    """
    if eval_fn is None:
        from engine import eval_model as engine_eval
        eval_fn = engine_eval.evaluate

    S = Searcher(eval_white_fn=eval_fn, tt_mb=128)
    move, _ = S.think(board, max_depth=depth, time_limit=time_limit)
    return move
