# engine/board.py

from collections import namedtuple
from itertools import count

Move = namedtuple("Move", "i j prom")

# Directions (0x88 style)
N, E, S, W = -10, 1, 10, -1
A1, H1, A8, H8 = 91, 98, 21, 28

# Piece movement directions
DIRECTIONS = {
    "P": (N, N+N, N+W, N+E),
    "N": (N+N+E, E+N+E, E+S+E, S+S+E, S+S+W, W+S+W, W+N+W, N+N+W),
    "B": (N+E, S+E, S+W, N+W),
    "R": (N, E, S, W),
    "Q": (N, E, S, W, N+E, S+E, S+W, N+W),
    "K": (N, E, S, W, N+E, S+E, S+W, N+W)
}

# Initial board (120-length with padding)
INITIAL_BOARD = (
    "         \n"  #  0 -  9
    "         \n"  # 10 - 19
    " rnbqkbnr\n"  # 20 - 29
    " pppppppp\n"  # 30 - 39
    " ........\n"  # 40 - 49
    " ........\n"  # 50 - 59
    " ........\n"  # 60 - 69
    " ........\n"  # 70 - 79
    " PPPPPPPP\n"  # 80 - 89
    " RNBQKBNR\n"  # 90 - 99
    "         \n"  #100 -109
    "         \n"  #110 -119
)

class Position:
    def __init__(self, board, score, wc, bc, ep, kp):
        self.board = board
        self.score = score
        self.wc = wc  # white castling rights (q-side, k-side)
        self.bc = bc  # black castling rights
        self.ep = ep  # en-passant square
        self.kp = kp  # king-passed square (for castling check detection)

    def value(self, move):
        from engine.evaluate import pst, piece_values  # import here to avoid circular import

        i, j, prom = move
        board = self.board
        p = board[i]
        q = board[j]

        score = pst[p][j] - pst[p][i]  # PST difference

        if q.islower():
            score += pst[q.upper()][119 - j]  # Mirror capture bonus

        # Promotion bonus
        if p == "P" and prom:
            score += pst[prom][j] - pst["P"][j]

        return score

    def gen_moves(self):
        for i, p in enumerate(self.board):
            if not p.isupper():
                continue
            for d in DIRECTIONS.get(p, ()):
                for j in count(i + d, d):
                    q = self.board[j]
                    if q.isspace() or q.isupper():
                        break
                    if p == "P":
                        if d in (N, N + N) and q != '.':
                            break
                        if d == N + N and (i < A1 + N or self.board[i + N] != '.'):
                            break
                        if d in (N + W, N + E) and q == '.' and j not in (self.ep, self.kp, self.kp - 1, self.kp + 1):
                            break
                        if A8 <= j <= H8:
                            for prom in "NBRQ":
                                yield Move(i, j, prom)
                            break
                    yield Move(i, j, "")
                    if p in "PNK" or q.islower():
                        break
                    if i == A1 and self.board[j + E] == "K" and self.wc[0]:
                        yield Move(j + E, j + W, "")
                    if i == H1 and self.board[j + W] == "K" and self.wc[1]:
                        yield Move(j + W, j + E, "")

    def rotate(self, nullmove=False):
        return Position(
            self.board[::-1].swapcase(),
            -self.score,
            self.bc,
            self.wc,
            119 - self.ep if self.ep and not nullmove else 0,
            119 - self.kp if self.kp and not nullmove else 0
        )

    def move(self, move):
        i, j, prom = move
        board = self.board
        put = lambda b, idx, val: b[:idx] + val + b[idx+1:]
        p = board[i]
        score = self.score  # actual value will be added in evaluate()

        # Reset EP/KP, update castling rights
        ep, kp = 0, 0
        wc, bc = self.wc, self.bc

        board = put(board, j, board[i])
        board = put(board, i, ".")

        if i == A1: wc = (False, wc[1])
        if i == H1: wc = (wc[0], False)
        if j == A8: bc = (bc[0], False)
        if j == H8: bc = (False, bc[1])

        if p == "K":
            wc = (False, False)
            if abs(j - i) == 2:
                kp = (i + j) // 2
                board = put(board, A1 if j < i else H1, ".")
                board = put(board, kp, "R")

        if p == "P":
            if A8 <= j <= H8:
                board = put(board, j, prom)
            if j - i == 2 * N:
                ep = i + N
            if j == self.ep:
                board = put(board, j + S, ".")

        return Position(board, score, wc, bc, ep, kp).rotate()
        
