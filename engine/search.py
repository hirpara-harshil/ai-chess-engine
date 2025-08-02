# engine/search.py

from collections import namedtuple

Entry = namedtuple("Entry", "lower upper")

MATE_SCORE = 60000
MATE_THRESHOLD = MATE_SCORE - 1000
EVAL_ROUGHNESS = 15
QS = 40
QS_A = 140

class Searcher:
    def __init__(self):
        self.tp_score = {}
        self.tp_move = {}
        self.history = set()
        self.nodes = 0

    def bound(self, pos, gamma, depth, can_null=True):
        self.nodes += 1
        depth = max(0, depth)

        # Immediate mate detection
        if pos.score <= -MATE_THRESHOLD:
            return -MATE_SCORE

        # Transposition Table lookup
        entry = self.tp_score.get((pos, depth, can_null), Entry(-MATE_SCORE, MATE_SCORE))
        if entry.lower >= gamma:
            return entry.lower
        if entry.upper < gamma:
            return entry.upper

        if can_null and depth > 0 and pos in self.history:
            return 0  # Repetition detected

        def moves():
            # Null move pruning
            if depth > 2 and can_null and abs(pos.score) < 500:
                yield None, -self.bound(pos.rotate(True), 1 - gamma, depth - 3)

            # Quiescence stand-pat
            if depth == 0:
                yield None, pos.score

            # Killer move from TT
            killer = self.tp_move.get(pos)

            if not killer and depth > 2:
                self.bound(pos, gamma, depth - 3, can_null=False)
                killer = self.tp_move.get(pos)

            val_limit = QS - depth * QS_A

            if killer and pos.value(killer) >= val_limit:
                yield killer, -self.bound(pos.move(killer), 1 - gamma, depth - 1)

            # All moves sorted by static value
            for val, move in sorted(((pos.value(m), m) for m in pos.gen_moves()), reverse=True):
                if val < val_limit:
                    break
                if depth <= 1 and pos.score + val < gamma:
                    yield move, pos.score + val
                    break
                yield move, -self.bound(pos.move(move), 1 - gamma, depth - 1)

        best = -MATE_SCORE
        for move, score in moves():
            best = max(best, score)
            if best >= gamma:
                if move:
                    self.tp_move[pos] = move
                break

        if depth > 2 and best == -MATE_SCORE:
            flipped = pos.rotate(True)
            in_check = self.bound(flipped, MATE_SCORE, 0) == MATE_SCORE
            best = -MATE_THRESHOLD if in_check else 0

        if best >= gamma:
            self.tp_score[pos, depth, can_null] = Entry(best, entry.upper)
        else:
            self.tp_score[pos, depth, can_null] = Entry(entry.lower, best)

        return best

    def search(self, history):
        """Iterative deepening"""
        self.nodes = 0
        self.tp_score.clear()
        self.history = set(history)

        gamma = 0
        root = history[-1]

        for depth in range(1, 1000):
            lower, upper = -MATE_THRESHOLD, MATE_THRESHOLD
            while lower < upper - EVAL_ROUGHNESS:
                score = self.bound(root, gamma, depth, can_null=False)
                if score >= gamma:
                    lower = score
                else:
                    upper = score
                yield depth, gamma, score, self.tp_move.get(root)
                gamma = (lower + upper + 1) // 2
