# scripts/generate_stockfish_data.py
"""
Generate a large number of semi-random legal chess positions, evaluate with Stockfish,
save as JSONL: {"fen": "...", "eval": float}
"""

import chess
import chess.engine
import random
import json
from tqdm import tqdm

def random_position(max_depth=20):
    """Play random moves from starting position to get a semi-random midgame board."""
    board = chess.Board()
    n_moves = random.randint(5, max_depth)
    for _ in range(n_moves):
        if board.is_game_over():
            break
        move = random.choice(list(board.legal_moves))
        board.push(move)
    return board

def main():
    stockfish_path = r"C:\ai-chess-engine\stockfish\stockfish.exe"
    output_file = "data/processed/stockfish_positions.jsonl"
    num_positions = 100_000  # target 100k positions
    depth = 12                # stockfish depth

    batch_size = 1000  # write in batches to avoid blocking
    positions_written = 0

    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine, \
         open(output_file, "w", encoding="utf-8") as out_f:

        pbar = tqdm(total=num_positions)
        while positions_written < num_positions:
            board = random_position()
            if board.is_game_over():
                continue

            try:
                info = engine.analyse(board, chess.engine.Limit(depth=depth))
                score = info["score"].pov(board.turn).score(mate_score=10000)
                if score is None:
                    continue  # skip unresolved mate positions
                rec = {"fen": board.fen(), "eval": score / 100.0}  # convert to pawns
                out_f.write(json.dumps(rec) + "\n")
                positions_written += 1
                pbar.update(1)
            except Exception as e:
                # skip if Stockfish fails on this position
                continue

        pbar.close()

    print(f"✅ Done. Saved {positions_written} Stockfish-evaluated positions to {output_file}")

if __name__ == "__main__":
    main()
