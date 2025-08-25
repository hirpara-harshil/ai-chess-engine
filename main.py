# main.py
import tkinter as tk
import chess
import threading
import time
from gui import ChessGUI
from engine.search import find_best_move
from engine import eval as engine_eval

# Default engine parameters
DEFAULT_DEPTH = 3

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess Engine Project - Play vs Bot")

        self.board = chess.Board()
        self.gui = ChessGUI(root, self.board)

        # Controls frame
        ctrl = tk.Frame(root)
        ctrl.grid(row=3, column=0, pady=(6,0))

        # Play vs bot checkbox
        self.play_vs_bot_var = tk.BooleanVar(value=False)
        tk.Checkbutton(ctrl, text="Play vs Bot", variable=self.play_vs_bot_var,
                       command=self._maybe_start_engine).grid(row=0, column=0, padx=4)

        # Choose side (radio buttons)
        self.side_var = tk.StringVar(value="white")
        tk.Radiobutton(ctrl, text="You: White", variable=self.side_var, value="white").grid(row=0, column=1)
        tk.Radiobutton(ctrl, text="You: Black", variable=self.side_var, value="black").grid(row=0, column=2)

        # Depth control
        tk.Label(ctrl, text="Depth (plies):").grid(row=0, column=3, padx=(10,0))
        self.depth_spin = tk.Spinbox(ctrl, from_=1, to=6, width=3)
        self.depth_spin.delete(0, "end")
        self.depth_spin.insert(0, str(DEFAULT_DEPTH))
        self.depth_spin.grid(row=0, column=4, padx=4)

        # New game button
        tk.Button(ctrl, text="New Game", command=self.new_game).grid(row=0, column=5, padx=8)

        # Engine thinking flag
        self.engine_thinking = False

        # Start polling loop for engine move
        self._poll_engine()

    def new_game(self):
        self.board.reset()
        self.gui.new_game()
        # If bot plays first (you chose black), let it move
        self._maybe_start_engine()

    def _maybe_start_engine(self):
        """
        Called when play_vs_bot toggles or when new game created.
        If it's bot's turn and Play vs Bot is enabled, schedule engine move.
        """
        if not self.play_vs_bot_var.get():
            return

        bot_color = chess.WHITE if self.side_var.get() == "black" else chess.BLACK
        # If current side to move == bot_color -> launch engine
        if self.board.turn == bot_color and not self.engine_thinking:
            self._launch_engine_move()

    def _disable_clicks(self):
        try:
            self.gui.canvas.unbind("<Button-1>")
        except Exception:
            pass

    def _enable_clicks(self):
        # rebind the canvas click back to gui.on_click
        self.gui.canvas.bind("<Button-1>", self.gui.on_click)

    def _launch_engine_move(self):
        # run the engine in a thread to avoid blocking Tkinter mainloop
        self.engine_thinking = True
        self._disable_clicks()
        depth = int(self.depth_spin.get())
        bot_color = chess.WHITE if self.side_var.get() == "black" else chess.BLACK

        def worker():
            try:
                # small safety: if user disabled play_vs_bot while thread started -> stop
                if not self.play_vs_bot_var.get():
                    return
                # time limit optional: None means full search to depth
                best = find_best_move(self.board, depth=depth, eval_fn=engine_eval.evaluate, time_limit=None)
                if best is not None:
                    # push move and update GUI (must schedule on main thread)
                    def push_and_render():
                        # double-check legality
                        if best in self.board.legal_moves:
                            self.board.push(best)
                            self.gui.last_move = best
                            self.gui.render()
                        # after engine move, maybe engine still to move (unlikely) -> check again
                        self.engine_thinking = False
                        self._enable_clicks()
                        # If opponent is the bot too and it's their turn, start again
                        self._maybe_start_engine()

                    self.root.after(1, push_and_render)
                else:
                    # no move (shouldn't happen)
                    self.root.after(1, lambda: self._enable_clicks())
                    self.engine_thinking = False
            finally:
                # ensure we re-enable clicks if something goes wrong
                self.root.after(1, lambda: setattr(self, "engine_thinking", False))
                self.root.after(1, self._enable_clicks)

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _poll_engine(self):
        """
        Poll loop: every 200ms check if it's bot's turn and auto-move if needed.
        This helps when human just moved (board updated via GUI).
        """
        try:
            if self.play_vs_bot_var.get() and not self.engine_thinking:
                # determine bot color
                bot_color = chess.WHITE if self.side_var.get() == "black" else chess.BLACK
                if self.board.turn == bot_color:
                    self._launch_engine_move()
        finally:
            self.root.after(200, self._poll_engine)

def main():
    root = tk.Tk()
    app = App(root)
    # Ensure GUI click binds exist
    app._enable_clicks()
    root.mainloop()

if __name__ == "__main__":
    main()
