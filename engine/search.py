# engine/search.py
# TT + PVS + LMR + Null-Move + Aspiration + (NEW) Futility, LMP, Razoring.
# strict time control with safety margin & pre-eval checks.
import time, math, random
import chess, chess.polyglot
from typing import Optional, Callable, Dict

MATE_SCORE = 100000
INF = 10**9

PIECE_CP = {None:0, chess.PAWN:100, chess.KNIGHT:320, chess.BISHOP:330, chess.ROOK:500, chess.QUEEN:900, chess.KING:10000}
def mvv_lva(board: chess.Board, m: chess.Move) -> int:
    if board.is_en_passant(m):
        vic = chess.PAWN
    else:
        p = board.piece_type_at(m.to_square)
        vic = p if p else None
    atk = board.piece_type_at(m.from_square)
    return 100*PIECE_CP[vic] - PIECE_CP[atk]

def is_tactical_parent(board: chess.Board, m: chess.Move) -> bool:
    # tactical properties must be computed on the parent position (before push)
    return board.is_capture(m) or (m.promotion is not None) or board.gives_check(m)

EXACT, LOWER, UPPER = 0, 1, 2

class TTEntry:
    __slots__ = ("key","depth","score","flag","move","age","static_eval")
    def __init__(self, key, depth, score, flag, move, age, static_eval):
        self.key, self.depth, self.score, self.flag = key, depth, score, flag
        self.move, self.age, self.static_eval = move, age, static_eval

class TranspositionTable:
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
            if e.key == k: return e
        return None
    def store(self, k: int, depth: int, score: int, flag: int, move: Optional[chess.Move], static_eval: Optional[int]):
        b = self.table[self._idx(k)]
        e = TTEntry(k, depth, score, flag, move, self.age, static_eval)
        if len(b) < self.ways:
            b.append(e); return
        r = min(range(self.ways), key=lambda i:(b[i].depth, b[i].age))
        b[r] = e

InfoCallback = Optional[Callable[[Dict], None]]

# strength profiles (unchanged behavior if you pass strength="MAX")
PROFILE = {
    "FM":  {"depth_cap": 6,  "top_k": 3, "temperature": 1.2},
    "IM":  {"depth_cap": 10, "top_k": 2, "temperature": 0.9},
    "GM":  {"depth_cap": 64, "top_k": 1, "temperature": 0.6},
    "MAX": {"depth_cap": 64, "top_k": 1, "temperature": 0.0},
}

class Searcher:
    def __init__(self, eval_white_fn, tt_mb=128, safety_margin_s: float = 0.08):
        self.eval_white = eval_white_fn
        self.tt = TranspositionTable(tt_mb)
        self.nodes = 0
        self.abort = False
        self.start = 0.0
        self.deadline = 0.0
        self.margin = max(0.0, float(safety_margin_s))
        self.killers = [[None, None] for _ in range(256)]
        self.history = {}

    def _tick(self) -> bool:
        if self.deadline and time.time() >= self.deadline:
            self.abort = True
            return True
        return False

    def static_eval(self, board: chess.Board) -> int:
        if self._tick():  # don’t start a slow eval when out of time
            return 0
        s_white = self.eval_white(board)
        return int(s_white if board.turn == chess.WHITE else -s_white)

    def order_moves(self, board: chess.Board, moves, tt_move, ply):
        k1, k2 = self.killers[ply]
        def key(m: chess.Move):
            if tt_move and m == tt_move: return 10_000_000
            if board.is_capture(m):      return 1_000_000 + mvv_lva(board, m)
            if k1 and m == k1:           return 900_000
            if k2 and m == k2:           return 800_000
            return self.history.get(m.uci(), 0)
        return sorted(moves, key=key, reverse=True)

    def get_pv(self, board: chess.Board, max_len: int = 30):
        pv = []
        seen = set()
        temp = board.copy(stack=False)
        for _ in range(max_len):
            k = chess.polyglot.zobrist_hash(temp)
            tte = self.tt.probe(k)
            if not tte or not tte.move: break
            m = tte.move
            if m not in temp.legal_moves: break
            kp = (k, m.uci())
            if kp in seen: break
            seen.add(kp)
            pv.append(m); temp.push(m)
        return pv

    @staticmethod
    def clamp_mate_window(alpha: int, beta: int, ply: int):
        lo = -MATE_SCORE + ply
        hi =  MATE_SCORE - ply
        return (max(alpha, lo), min(beta, hi))

    def qsearch(self, board: chess.Board, alpha: int, beta: int) -> int:
        """Quiescence: if in check, search all evasions; otherwise stand-pat + captures."""
        if self._tick(): return 0
        self.nodes += 1

        # In-check qsearch: explore all legal evasions (not only captures)
        if board.is_check():
            best = -INF
            moves = self.order_moves(board, list(board.legal_moves), None, 0)
            for m in moves:
                if self._tick(): break
                board.push(m)
                score = -self.qsearch(board, -beta, -alpha)
                board.pop()
                if score >= beta: return beta
                if score > best: best = score
                if score > alpha: alpha = score
            # no legal move -> checkmated at qsearch node
            return best if best != -INF else -MATE_SCORE

        # Stand-pat
        stand = self.static_eval(board)
        if stand >= beta: return beta
        if alpha < stand: alpha = stand

        # Captures only
        caps = [m for m in board.legal_moves if board.is_capture(m)]
        if caps:
            caps = self.order_moves(board, caps, None, 0)
        for m in caps:
            if self._tick(): return alpha
            board.push(m)
            score = -self.qsearch(board, -beta, -alpha)
            board.pop()
            if score >= beta: return beta
            if score > alpha: alpha = score
        return alpha

    def can_null(self, board: chess.Board) -> bool:
        if board.is_check(): return False
        minors_majors = 0
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            minors_majors += len(board.pieces(pt, True)) + len(board.pieces(pt, False))
        pawns = len(board.pieces(chess.PAWN, True)) + len(board.pieces(chess.PAWN, False))
        return (minors_majors + pawns) > 6

    def search(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int, in_pv: bool) -> int:
        if self._tick(): return 0
        self.nodes += 1
        alpha, beta = self.clamp_mate_window(alpha, beta, ply)

        k = chess.polyglot.zobrist_hash(board)

        if board.is_repetition(3) or board.can_claim_fifty_moves():
            return 0

        if board.is_checkmate():
            return -MATE_SCORE + ply
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        in_check = board.is_check()
        if in_check:
            depth += 1  # check extension

        # TT
        tte = self.tt.probe(k)
        if tte and tte.depth >= depth:
            if tte.flag == EXACT: return tte.score
            if tte.flag == LOWER and tte.score >= beta:  return tte.score
            if tte.flag == UPPER and tte.score <= alpha: return tte.score

        # Leaf & frontier pruning
        if depth <= 0:
            return self.qsearch(board, alpha, beta)

        # --- Razoring (depth==1, not in check): try qsearch early if static eval way below alpha ---
        if depth == 1 and not in_check:
            st = self.static_eval(board)
            R = 200  # cp margin (tune)
            if st + R <= alpha:
                val = self.qsearch(board, alpha, beta)
                if val <= alpha:
                    return val

        # Null-move
        if not in_pv and depth >= 3 and self.can_null(board):
            board.push(chess.Move.null())
            Rn = 2 + (depth // 6)
            sc = -self.search(board, depth - 1 - Rn, -beta, -beta + 1, ply+1, in_pv=False)
            board.pop()
            if self.abort: return 0
            if sc >= beta:
                ver = -self.search(board, depth - 1, -beta, -beta + 1, ply+1, in_pv=False)
                if self.abort: return 0
                if ver >= beta:
                    self.tt.store(k, depth, ver, LOWER, chess.Move.null(), None)
                    return ver

        # Move ordering
        tt_move = tte.move if tte else None
        moves = list(board.legal_moves)
        if tt_move or moves:
            moves = self.order_moves(board, moves, tt_move, ply)
        if not moves:
            return 0

        best = -INF
        best_move = None
        first = True
        orig_alpha = alpha

        # Futility/LMP parameters
        fut_margin = 100 * (depth)      # cp margin; tune
        lmp_threshold = 8 + 2*depth     # after this many quiets at shallow depth, prune

        tried_quiets = 0
        for i, m in enumerate(moves):
            if self._tick(): return best if best != -INF else 0

            # --- Precompute tactical flags on the parent (correctly) ---
            is_cap = board.is_capture(m)
            tactical = is_tactical_parent(board, m)

            board.push(m)

            # --- Futility pruning (skip hopeless quiets at shallow depths) ---
            if not in_check and not is_cap and depth <= 3:
                st = self.static_eval(board)  # guarded by _tick()
                if st + fut_margin <= alpha:
                    board.pop()
                    tried_quiets += 1
                    # Late Move Pruning: skip very late quiets at shallow depth
                    if depth <= 2 and tried_quiets > lmp_threshold:
                        continue
                    continue

            # --- LMR (don't reduce checks/captures/promos; and don't reduce if child is check) ---
            child_in_check = board.is_check()
            do_lmr = (not in_pv) and depth >= 3 and (not tactical) and (not child_in_check)
            reduction = 0
            if do_lmr:
                reduction = int(0.75 + math.log(depth+1, 2) * math.log(i+2, 2))
                reduction = min(reduction, depth-1)

            if first:
                sc = -self.search(board, depth - 1, -beta, -alpha, ply+1, in_pv=in_pv)
                first = False
            else:
                sc = -self.search(board, depth - 1 - reduction, -alpha-1, -alpha, ply+1, in_pv=False)
                if self.abort:
                    board.pop(); return best if best != -INF else 0
                if sc > alpha and reduction > 0:
                    sc = -self.search(board, depth - 1, -alpha-1, -alpha, ply+1, in_pv=False)
                if sc > alpha and sc < beta:
                    sc = -self.search(board, depth - 1, -beta, -alpha, ply+1, in_pv=True)

            board.pop()
            if self.abort: return best if best != -INF else 0

            if sc > best:
                best = sc
                best_move = m
                if sc > alpha:
                    alpha = sc
                    if not tactical:
                        self.history[m.uci()] = self.history.get(m.uci(), 0) + depth*depth
                if alpha >= beta:
                    if not tactical:
                        k1, k2 = self.killers[ply]
                        if m != k1:
                            self.killers[ply] = [m, k1]
                    break
            if not is_cap:
                tried_quiets += 1

        flag = EXACT if best > orig_alpha and best < beta else (LOWER if best >= beta else UPPER)
        static_eval = tte.static_eval if (tte and tte.static_eval is not None) else self.static_eval(board)
        self.tt.store(k, depth, best, flag, best_move, static_eval)
        return best

    @staticmethod
    def score_to_mate(score: int) -> Optional[int]:
        s = abs(score)
        if s >= MATE_SCORE - 1000:
            return (MATE_SCORE - s) if score > 0 else -(MATE_SCORE - s)
        return None

    def _root_loop(self, board: chess.Board, depth: int, alpha: int, beta: int):
        root_scores = []
        k = chess.polyglot.zobrist_hash(board)
        tt_move = None
        tte = self.tt.probe(k)
        if tte and tte.move: tt_move = tte.move
        moves = self.order_moves(board, list(board.legal_moves), tt_move, ply=0)
        best = -INF; first = True
        for m in moves:
            if self._tick(): break
            # DEFENSIVE: ensure the move is still legal for this board state
            if m not in board.legal_moves:
                continue

            board.push(m)
            if first:
                sc = -self.search(board, depth-1, -beta, -alpha, ply=1, in_pv=True)
                first = False
            else:
                sc = -self.search(board, depth-1, -alpha-1, -alpha, ply=1, in_pv=False)
                if sc > alpha and sc < beta:
                    sc = -self.search(board, depth-1, -beta, -alpha, ply=1, in_pv=True)
            board.pop()
            root_scores.append((m, sc))
            if sc > best:
                best = sc
                if sc > alpha:
                    alpha = sc
        root_scores.sort(key=lambda t: t[1], reverse=True)
        return root_scores, best

    def think(self, board: chess.Board, max_depth=64, time_limit: Optional[float]=None, info_cb: InfoCallback=None):
        self.nodes = 0
        self.abort = False
        self.tt.new_age()
        self.start = time.time()
        self.deadline = self.start + max(0.0, (time_limit or 0.0) - self.margin) if time_limit else 0.0

        best_move = None
        last_score = 0
        last_root_scores = []
        window = 50

        for d in range(1, max_depth+1):
            if self._tick(): break
            alpha = last_score - window
            beta  = last_score + window
            while True:
                root_scores, best = self._root_loop(board, d, alpha, beta)
                if self.abort: break
                if best <= alpha:
                    alpha -= window * 2
                elif best >= beta:
                    beta  += window * 2
                else:
                    last_score = best
                    last_root_scores = root_scores
                    break
            if self.abort: break
            if last_root_scores:
                best_move = last_root_scores[0][0]

            if info_cb is not None:
                elapsed = max(1e-6, time.time() - self.start)
                mate = self.score_to_mate(last_score)
                pv_moves = self.get_pv(board, max_len=30)
                info = {
                    "depth": d,
                    "score_cp": None if mate is not None else last_score,
                    "mate": mate,
                    "pv": [m.uci() for m in pv_moves],
                    "nodes": self.nodes,
                    "nps": int(self.nodes / elapsed),
                    "time_ms": int(elapsed * 1000),
                }
                try: info_cb(info)
                except Exception: pass

        return best_move, last_score, last_root_scores

def find_best_move(
    board: chess.Board,
    depth: int = 3,
    eval_fn=None,
    time_limit: Optional[float]=None,
    strength: str = "MAX",
    info_cb: InfoCallback = None
) -> Optional[chess.Move]:
    if eval_fn is None:
        from engine import eval_model as engine_eval
        eval_fn = engine_eval.evaluate

    prof = PROFILE.get(strength.upper(), PROFILE["MAX"])
    max_depth = min(depth, prof["depth_cap"])

    S = Searcher(eval_white_fn=eval_fn, tt_mb=128, safety_margin_s=0.08)
    best_move, best_score, root_scores = S.think(board, max_depth=max_depth, time_limit=time_limit, info_cb=info_cb)

    if strength.upper() != "MAX" and root_scores:
        top = root_scores[:prof["top_k"]]
        mate_best = Searcher.score_to_mate(top[0][1])
        if mate_best is None and len(top) > 1 and prof["temperature"] > 0:
            temp = prof["temperature"]
            xs = [sc/100.0 for (_, sc) in top]
            m = max(xs)
            exps = [math.exp((x - m)/max(0.01, temp)) for x in xs]
            s = sum(exps)
            r = random.random() * s
            acc = 0.0
            choice = top[0][0]
            for (mv, sc), e in zip(top, exps):
                acc += e
                if r <= acc:
                    choice = mv; break
            return choice
    return best_move
