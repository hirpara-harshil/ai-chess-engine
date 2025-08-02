from engine.board import Position, INITIAL_BOARD, Move, A1
from engine.search import Searcher
from engine.evaluate import evaluate

def parse_move(move_str):
    """Convert algebraic (e2e4) to Move(i, j, prom) in 0x88"""
    def to_index(sq):
        file = ord(sq[0]) - ord('a')
        rank = int(sq[1]) - 1
        return A1 + file - 10 * rank

    if len(move_str) < 4:
        return None
    i = to_index(move_str[:2])
    j = to_index(move_str[2:4])
    prom = move_str[4].upper() if len(move_str) == 5 else ""
    return Move(i, j, prom)

def move_to_str(move, is_white):
    """Convert Move to algebraic notation based on perspective"""
    def from_index(i):
        rank, file = divmod(i - A1, 10)
        return chr(file + ord('a')) + str(-rank + 1)

    if not is_white:
        # If it's black's turn, we need to rotate the move
        move = Move(119 - move.i, 119 - move.j, move.prom)

    return from_index(move.i) + from_index(move.j) + move.prom.lower()

def print_board(board):
    print("  +-----------------+")
    for r in range(2, 10):
        row = board[r * 10 + 1 : r * 10 + 9]
        print(f"{10 - r} | {' '.join(row)} |")
    print("  +-----------------+")
    print("    a b c d e f g h")

def main():
    print("Welcome to your chess bot!")
    color = input("Play as (w/b): ").strip().lower()
    user_is_white = (color == "w")

    # Setup starting position
    board_str = INITIAL_BOARD
    pos = Position(board_str, 0, (True, True), (True, True), 0, 0)
    pos.score = evaluate(pos)
    history = [pos]
    searcher = Searcher()

    while True:
        current = history[-1]
        print_board(current.board)

        if current.board.count("K") < 1 or current.board.count("k") < 1:
            print("Game Over.")
            break

        if current.board.count('.') > 70:
            print("Draw by insufficient material or stalemate.")
            break

        is_users_turn = (len(history) % 2 == 1) == user_is_white

        if is_users_turn:
            move_str = input("Your move (e.g. e2e4): ").strip()
            move = parse_move(move_str)
            if move not in current.gen_moves():
                print("Invalid move.")
                continue
            next_pos = current.move(move)
            next_pos.score = evaluate(next_pos)
            history.append(next_pos)
        else:
            print("Engine thinking...")
            best_move = None
            for depth, gamma, score, move in searcher.search(history):
                if move:
                    best_move = move
                    print(f"Depth {depth}, score {score}, move {move_to_str(move, False)}")
                if depth >= 4:
                    break

            if best_move:
                print(f"Engine plays: {move_to_str(best_move, False)}")
                next_pos = current.move(best_move)
                next_pos.score = evaluate(next_pos)
                history.append(next_pos)
            else:
                print("Engine resigns or has no move.")
                break

if __name__ == "__main__":
    main()
