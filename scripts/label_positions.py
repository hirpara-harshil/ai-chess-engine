# scripts/label_positions.py
"""
Read positions JSONL (fen lines), evaluate with local Stockfish, convert to tensor
(using engine.utils.board_to_tensor) and save compressed NPZ shards X/Y for training.
"""
import argparse
import json
import os
import math
from tqdm import tqdm
import chess
import chess.engine
import numpy as np

# import your board->tensor util (must run from project root)
from engine.utils import board_to_tensor

def stockfish_eval(fen, sf_path, depth=None, movetime_ms=None):
    board = chess.Board(fen)
    with chess.engine.SimpleEngine.popen_uci(sf_path) as eng:
        if depth:
            info = eng.analyse(board, chess.engine.Limit(depth=depth))
        else:
            info = eng.analyse(board, chess.engine.Limit(time=movetime_ms/1000.0))
    score = info.get("score")
    # Convert to White-perspective centipawns
    if score is None:
        return None
    # use pov(WHITE) to make output consistently White-perspective
    sc = score.pov(chess.WHITE)
    if sc.is_mate():
        mate = sc.mate()
        if mate is None:
            return None
        # convert mate to large cp sign preserving
        return 100000.0 if mate > 0 else -100000.0
    else:
        return float(sc.score(mate_score=100000))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--positions", required=True, help="positions JSONL (output of extract_positions.py)")
    p.add_argument("--stockfish", required=True, help="path to stockfish binary")
    p.add_argument("--depth", type=int, default=12, help="Stockfish search depth (or use --movetime-ms)")
    p.add_argument("--movetime-ms", type=int, default=0, help="Stockfish movetime per position in ms (if >0 used instead of depth)")
    p.add_argument("--shard-size", type=int, default=5000, help="number of positions per .npz shard")
    p.add_argument("--out-dir", default="data/processed/shards", help="where to save .npz shards")
    p.add_argument("--max-positions", type=int, default=0, help="0=no limit")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    X_buffer = []
    Y_buffer = []
    shard_idx = 0
    processed = 0

    # open engine once and reuse (faster)
    eng = chess.engine.SimpleEngine.popen_uci(args.stockfish)

    with open(args.positions, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            if args.max_positions and processed >= args.max_positions:
                break
            rec = json.loads(line)
            fen = rec["fen"]
            board = chess.Board(fen)
            try:
                if args.movetime_ms and args.movetime_ms > 0:
                    info = eng.analyse(board, chess.engine.Limit(time=args.movetime_ms/1000.0))
                else:
                    info = eng.analyse(board, chess.engine.Limit(depth=args.depth))
            except Exception as e:
                # on rare engine errors, skip
                print("Engine error:", e)
                continue

            score = info.get("score")
            if score is None:
                continue

            sc = score.pov(chess.WHITE)
            if sc.is_mate():
                mate = sc.mate()
                if mate is None:
                    continue
                cp = 100000.0 if mate > 0 else -100000.0
            else:
                cp = float(sc.score(mate_score=100000))

            # convert board -> tensor (numpy)
            try:
                x = board_to_tensor(board)
            except Exception as e:
                print("board_to_tensor error:", e)
                continue

            X_buffer.append(x.astype(np.float32))
            Y_buffer.append(float(cp))
            processed += 1

            if len(X_buffer) >= args.shard_size:
                out = os.path.join(args.out_dir, f"shard_{shard_idx:04d}.npz")
                np.savez_compressed(out, X=np.stack(X_buffer), Y=np.array(Y_buffer, dtype=np.float32))
                print("Saved", out, " (positions:", len(X_buffer), ")")
                shard_idx += 1
                X_buffer = []
                Y_buffer = []

    # flush remainder
    if X_buffer:
        out = os.path.join(args.out_dir, f"shard_{shard_idx:04d}.npz")
        np.savez_compressed(out, X=np.stack(X_buffer), Y=np.array(Y_buffer, dtype=np.float32))
        print("Saved", out)

    eng.quit()
    print("Done. Total positions labeled:", processed)

if __name__ == "__main__":
    main()
