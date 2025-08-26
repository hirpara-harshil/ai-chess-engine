# main.py
import tkinter as tk
import chess
import threading
import time
from gui import ChessGUI
from engine.search import find_best_move
from engine import eval_model as engine_eval

DEFAULT_DEPTH = 64  # depth cap; search will also be capped by strength profile

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess Engine Project - Play vs Bot")

        self.board = chess.Board()
        self.gui = ChessGUI(root, self.board)

        ctrl = tk.Frame(root)
        ctrl.grid(row=3, column=0, pady=(6,0))

        # Play vs bot checkbox
        self.play_vs_bot_var = tk.BooleanVar(value=False)
        tk.Checkbutton(ctrl, text="Play vs Bot", variable=self.play_vs_bot_var,
                       command=self._maybe_start_engine).grid(row=0, column=0, padx=4)

        # Choose side
        self.side_var = tk.StringVar(value="white")
        tk.Radiobutton(ctrl, text="You: White", variable=self.side_var, value="white").grid(row=0, column=1)
        tk.Radiobutton(ctrl, text="You: Black", variable=self.side_var, value="black").grid(row=0, column=2)

        # Strength selector
        tk.Label(ctrl, text="Strength:").grid(row=0, column=3, padx=(10,0))
        self.strength_var = tk.StringVar(value="MAX")
        strength_menu = tk.OptionMenu(ctrl, self.strength_var, "FM", "IM", "GM", "MAX")
        strength_menu.config(width=5)
        strength_menu.grid(row=0, column=4, padx=4)

        # Time per move (seconds)
        tk.Label(ctrl, text="Time (s):").grid(row=0, column=5, padx=(10,0))
        self.time_spin = tk.Spinbox(ctrl, from_=0.1, to=10.0, increment=0.1, width=5)
        self.time_spin.delete(0, "end")
        self.time_spin.insert(0, "1.0")  # default 1 second
        self.time_spin.grid(row=0, column=6, padx=4)

        # Depth (kept for analysis cap; engine also caps by strength)
        tk.Label(ctrl, text="Max Depth:").grid(row=0, column=7, padx=(10,0))
        self.depth_spin = tk.Spinbox(ctrl, from_=1, to=64, width=4)
        self.depth_spin.delete(0, "end")
        self.depth_spin.insert(0, str(DEFAULT_DEPTH))
        self.depth_spin.grid(row=0, column=8, padx=4)

        # New game
        tk.Button(ctrl, text="New Game", command=self.new_game).grid(row=0, column=9, padx=8)

        # Analysis info line
        self.info_label = tk.Label(root, text="Analysis: (idle)", anchor="w", justify="left")
        self.info_label.grid(row=4, column=0, sticky="w", padx=6, pady=(4,8))

        # Engine thinking flag
        self.engine_thinking = False

        # Start poller
        self._poll_engine()

    def new_game(self):
        self.board.reset()
        self.gui.new_game()
        self._maybe_start_engine()

    def _maybe_start_engine(self):
        if not self.play_vs_bot_var.get():
            return
        bot_color = chess.WHITE if self.side_var.get() == "black" else chess.BLACK
        if self.board.turn == bot_color and not self.engine_thinking:
            self._launch_engine_move()

    def _disable_clicks(self):
        try:
            self.gui.canvas.unbind("<Button-1>")
        except Exception:
            pass

    def _enable_clicks(self):
        self.gui.canvas.bind("<Button-1>", self.gui.on_click)

    def _launch_engine_move(self):
        self.engine_thinking = True
        self._disable_clicks()

        depth = int(self.depth_spin.get())
        strength = self.strength_var.get()
        try:
            time_limit = float(self.time_spin.get())
        except Exception:
            time_limit = 1.0

        # analysis callback to live-update PV each completed depth
        def info_cb(info):
            def _update():
                if info.get("mate") is not None:
                    s = f"Mate in {info['mate']}" if info['mate']>0 else f"Mated in {abs(info['mate'])}"
                else:
                    s = f"{info['score_cp']} cp"
                pv_str = " ".join(info.get("pv", [])[:10])
                self.info_label.config(
                    text=f"Analysis: depth {info['depth']} | {s} | nodes {info['nodes']} | nps {info['nps']} | pv {pv_str}"
                )
            self.root.after(0, _update)

        def worker():
            try:
                if not self.play_vs_bot_var.get():
                    return
                # ask engine; strict per-move time
                best = find_best_move(
                    self.board,
                    depth=depth,
                    eval_fn=engine_eval.evaluate,
                    time_limit=time_limit,
                    strength=strength,
                    info_cb=info_cb
                )
                if best is not None:
                    def push_and_render():
                        if best in self.board.legal_moves:
                            self.board.push(best)
                            self.gui.last_move = best
                            self.gui.render()
                        self.engine_thinking = False
                        self._enable_clicks()
                        self._maybe_start_engine()
                    self.root.after(1, push_and_render)
                else:
                    self.root.after(1, self._enable_clicks)
                    self.engine_thinking = False
            finally:
                self.root.after(1, lambda: setattr(self, "engine_thinking", False))
                self.root.after(1, self._enable_clicks)

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _poll_engine(self):
        try:
            if self.play_vs_bot_var.get() and not self.engine_thinking:
                bot_color = chess.WHITE if self.side_var.get() == "black" else chess.BLACK
                if self.board.turn == bot_color:
                    self._launch_engine_move()
        finally:
            self.root.after(200, self._poll_engine)

def main():
    root = tk.Tk()
    app = App(root)
    app._enable_clicks()
    root.mainloop()

if __name__ == "__main__":
    main()
