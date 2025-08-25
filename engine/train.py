# engine/train.py
import argparse
import os
import math
import time
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from engine.nn_model import EvalNet
from engine.utils import board_to_tensor
import chess.pgn
import chess
try:
    import chess.engine
except Exception:
    chess = None  # stockfish labeling not available

# ---------- simple dataset ----------
class ArrayDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs = xs.astype(np.float32)
        self.ys = ys.astype(np.float32)
    def __len__(self):
        return len(self.xs)
    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

# ---------- helper: label fen using stockfish ----------
def label_with_stockfish(fen, sf_path="stockfish", movetime_ms=200):
    if chess is None:
        raise RuntimeError("python-chess engine not available")
    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    board = chess.Board(fen)
    info = engine.analyse(board, chess.engine.Limit(time=movetime_ms/1000.0))
    # info['score'] can be mate or cp
    score = info.get('score')
    cp = None
    if score.is_mate():
        # convert mate to large cp with sign
        mate_in = score.white().mate()
        if mate_in is None:
            cp = 0.0
        else:
            cp = 100000.0 if mate_in > 0 else -100000.0
    else:
        cp = float(score.white().score(mate_score=100000))
    engine.close()
    return cp

# ---------- build dataset from PGN (optionally label with Stockfish) ----------
def build_dataset_from_pgn(pgn_paths, out_dir, max_positions_per_file=2000, use_stockfish=False, sf_path="stockfish", movetime_ms=200):
    os.makedirs(out_dir, exist_ok=True)
    shard_id = 0
    X_shard, Y_shard = [], []
    for pgn_path in pgn_paths:
        with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                board = game.board()
                for move in game.mainline_moves():
                    board.push(move)
                    # sample positions every ply or with some probability
                    fen = board.fen()
                    x = board_to_tensor(board)
                    if use_stockfish:
                        y = label_with_stockfish(fen, sf_path=sf_path, movetime_ms=movetime_ms)
                    else:
                        # fallback: use game result as weak label
                        result = game.headers.get("Result", "1/2-1/2")
                        if result == "1-0":
                            y = 300.0
                        elif result == "0-1":
                            y = -300.0
                        else:
                            y = 0.0
                    X_shard.append(x)
                    Y_shard.append(y)
                    if len(X_shard) >= max_positions_per_file:
                        out = os.path.join(out_dir, f"shard_{shard_id}.npz")
                        np.savez_compressed(out, X=np.stack(X_shard), Y=np.array(Y_shard, dtype=np.float32))
                        print("Wrote", out)
                        shard_id += 1
                        X_shard, Y_shard = [], []
    # flush remainder
    if X_shard:
        out = os.path.join(out_dir, f"shard_{shard_id}.npz")
        np.savez_compressed(out, X=np.stack(X_shard), Y=np.array(Y_shard, dtype=np.float32))
        print("Wrote", out)

# ---------- training loop ----------
def train_from_npz(npz_paths, out_model_path, epochs=6, batch_size=256, lr=1e-3, device="cpu"):
    # load all shards into arrays (for small datasets)
    Xs, Ys = [], []
    for p in npz_paths:
        d = np.load(p)
        Xs.append(d["X"])
        Ys.append(d["Y"])
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    # optional clipping & normalization
    # clip cp to +/- 10000 to avoid huge targets
    Y = np.clip(Y, -10000.0, 10000.0).astype(np.float32)
    # no scaling here: we train to predict cp directly (so no scale needed)
    ds = ArrayDataset(X, Y)
    val_n = max(1, int(0.02 * len(ds)))
    train_n = len(ds) - val_n
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_n, val_n])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    device = torch.device(device)
    model = EvalNet(input_dim=X.shape[1])
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    for ep in range(1, epochs+1):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb).squeeze(-1)
                val_loss += loss_fn(pred, yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {ep}: train_loss={train_loss:.3f}, val_loss={val_loss:.3f}, time={time.time()-t0:.1f}s")
        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'model_state': model.state_dict()}, out_model_path)
            print("Saved", out_model_path)

    # Save TorchScript for fast inference
    model_cpu = EvalNet(input_dim=X.shape[1])
    model_cpu.load_state_dict(torch.load(out_model_path)['model_state'])
    model_cpu.eval()
    script = torch.jit.script(model_cpu)
    ts_path = out_model_path.replace(".pt", "_ts.pt")
    script.save(ts_path)
    print("Saved TorchScript", ts_path)
    return out_model_path, ts_path

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["build", "train"], required=True)
    p.add_argument("--pgns", nargs="*", help="pgn files for build mode")
    p.add_argument("--out_dir", default="data/shards", help="where to write shards")
    p.add_argument("--use-stockfish", action="store_true", help="query stockfish for labels (local binary required)")
    p.add_argument("--stockfish-path", default="stockfish", help="path to stockfish binary")
    p.add_argument("--movetime-ms", default=200, type=int, help="stockfish movetime per position")
    p.add_argument("--npz", nargs="*", help="npz shards to train from")
    p.add_argument("--out_model", default="models/nn_eval.pt")
    p.add_argument("--epochs", default=6, type=int)
    p.add_argument("--batch_size", default=256, type=int)
    args = p.parse_args()

    if args.mode == "build":
        if not args.pgns:
            raise SystemExit("No PGN files given")
        build_dataset_from_pgn(args.pgns, args.out_dir, use_stockfish=args.use_stockfish,
                               sf_path=args.stockfish_path, movetime_ms=args.movetime_ms)
    elif args.mode == "train":
        if not args.npz:
            raise SystemExit("No npz shards provided")
        os.makedirs(os.path.dirname(args.out_model) or ".", exist_ok=True)
        train_from_npz(args.npz, args.out_model, epochs=args.epochs,
                       batch_size=args.batch_size)
    else:
        raise SystemExit("Invalid mode")

if __name__ == "__main__":
    main()
