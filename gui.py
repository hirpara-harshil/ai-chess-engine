# gui.py

import tkinter as tk
from tkinter import messagebox
import chess

class ChessGUI:
    LIGHT = "#EEEED2"
    DARK  = "#769656"
    HIGHLIGHT = "#F6F669"
    MOVE_FROM = "#BACA2B"
    MOVE_TO   = "#EBEC6B"

    def __init__(self, root, board: chess.Board):
        self.root = root
        self.board = board

        self.square_size = 72
        self.orientation_white_bottom = True  # toggle via UI
        self.selected_square = None
        self.legal_targets_for_selected = set()
        self.last_move = None  # chess.Move or None

        self.canvas = tk.Canvas(
            root, width=8*self.square_size, height=8*self.square_size, highlightthickness=0
        )
        self.canvas.grid(row=0, column=0, columnspan=6)

        # Controls
        self.status_var = tk.StringVar()
        self.status_var.set("White to move")
        tk.Label(root, textvariable=self.status_var, anchor="w").grid(row=1, column=0, columnspan=6, sticky="we", padx=4, pady=(6,0))

        tk.Button(root, text="New Game", command=self.new_game).grid(row=2, column=0, sticky="we", padx=2, pady=4)
        tk.Button(root, text="Undo", command=self.undo).grid(row=2, column=1, sticky="we", padx=2, pady=4)
        tk.Button(root, text="Flip", command=self.flip_board).grid(row=2, column=2, sticky="we", padx=2, pady=4)
        tk.Button(root, text="Claim Draw", command=self.claim_draw).grid(row=2, column=3, sticky="we", padx=2, pady=4)
        tk.Button(root, text="Copy FEN", command=self.copy_fen).grid(row=2, column=4, sticky="we", padx=2, pady=4)

        self.canvas.bind("<Button-1>", self.on_click)

        self.piece_symbols = {
            "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
            "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚"
        }

        self.render()

    # ---------- Drawing ----------
    def render(self):
        self.canvas.delete("all")
        self._draw_board()
        self._highlight_context()
        self._draw_pieces()
        self._update_status()

    def _draw_board(self):
        for board_row in range(8):
            for board_col in range(8):
                x1, y1 = board_col * self.square_size, board_row * self.square_size
                x2, y2 = x1 + self.square_size, y1 + self.square_size
                color = self.LIGHT if (board_row + board_col) % 2 == 0 else self.DARK
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=color)

        # files/ranks coordinates
        files = "abcdefgh"
        ranks = "12345678"
        disp_files = files if self.orientation_white_bottom else files[::-1]
        disp_ranks = ranks if self.orientation_white_bottom else ranks[::-1]
        for i, f in enumerate(disp_files):
            x = i*self.square_size + 5
            y = 8*self.square_size - 16
            self.canvas.create_text(x, y, text=f, anchor="w")
        for i, r in enumerate(disp_ranks[::-1]):
            x = 5
            y = i*self.square_size + 16
            self.canvas.create_text(x, y, text=r, anchor="w")

    def _highlight_context(self):
        # Selected square + its legal targets
        if self.selected_square is not None:
            sx, sy = self._square_to_canvas_coords(self.selected_square)
            self._draw_square_overlay(sx, sy, self.MOVE_FROM)

            for tgt in self.legal_targets_for_selected:
                tx, ty = self._square_to_canvas_coords(tgt)
                self._draw_square_overlay(tx, ty, self.HIGHLIGHT)

        # Last move from/to
        if self.last_move:
            fx, fy = self._square_to_canvas_coords(self.last_move.from_square)
            tx, ty = self._square_to_canvas_coords(self.last_move.to_square)
            self._draw_square_overlay(fx, fy, self.MOVE_FROM, alpha=0.35)
            self._draw_square_overlay(tx, ty, self.MOVE_TO, alpha=0.35)

    def _draw_pieces(self):
        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if not piece:
                continue
            col = chess.square_file(sq)
            row = chess.square_rank(sq)
            cx, cy = self._board_to_canvas(col, row)
            self.canvas.create_text(
                cx, cy, text=self.piece_symbols[piece.symbol()],
                font=("Arial", int(self.square_size*0.6)), fill="black"
            )

    def _draw_square_overlay(self, col, row, color, alpha=0.55):
        # Semi-transparent overlay using a stipple rectangle
        x1 = col * self.square_size
        y1 = row * self.square_size
        x2 = x1 + self.square_size
        y2 = y1 + self.square_size
        # Using stipple to simulate transparency (Tkinter limitation)
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, stipple="gray50", outline="")

    # ---------- Coordinates helpers ----------
    def _board_to_canvas(self, file_idx, rank_idx):
        # rank_idx: 0(bottom for White) .. 7(top for White) in chess library terms
        if self.orientation_white_bottom:
            draw_row = 7 - rank_idx
            draw_col = file_idx
        else:
            draw_row = rank_idx
            draw_col = 7 - file_idx
        x = draw_col * self.square_size + self.square_size//2
        y = draw_row * self.square_size + self.square_size//2
        return x, y

    def _canvas_to_square(self, x, y):
        col = int(x // self.square_size)
        row = int(y // self.square_size)
        if not (0 <= col < 8 and 0 <= row < 8):
            return None
        if self.orientation_white_bottom:
            file_idx = col
            rank_idx = 7 - row
        else:
            file_idx = 7 - col
            rank_idx = row
        return chess.square(file_idx, rank_idx)

    def _square_to_canvas_coords(self, sq):
        file_idx = chess.square_file(sq)
        rank_idx = chess.square_rank(sq)
        if self.orientation_white_bottom:
            draw_row = 7 - rank_idx
            draw_col = file_idx
        else:
            draw_row = rank_idx
            draw_col = 7 - file_idx
        return draw_col, draw_row

    # ---------- Interaction ----------
    def on_click(self, event):
        sq = self._canvas_to_square(event.x, event.y)
        if sq is None:
            return

        # If selecting a piece or reselecting
        if self.selected_square is None:
            self._try_select_square(sq)
        else:
            # If clicking same color piece again => reselect
            piece = self.board.piece_at(sq)
            if piece and (piece.color == self.board.turn):
                self._try_select_square(sq)
            else:
                # Attempt move from selected to this square
                self._try_make_move(self.selected_square, sq)

        self.render()

    def _try_select_square(self, sq):
        piece = self.board.piece_at(sq)
        if piece and (piece.color == self.board.turn):
            self.selected_square = sq
            # collect legal targets from this square
            self.legal_targets_for_selected = {
                m.to_square for m in self.board.legal_moves if m.from_square == sq
            }
        else:
            # empty or opponent piece (not selectable as origin)
            self.selected_square = None
            self.legal_targets_for_selected = set()

    def _try_make_move(self, from_sq, to_sq):
        # Determine if promotion is required
        move = chess.Move(from_sq, to_sq)
        if move not in self.board.legal_moves:
            # If a promotion might be needed, request piece type
            if self._is_promotion_move(from_sq, to_sq):
                promo_piece = self._ask_promotion_piece()
                if promo_piece is None:
                    # canceled
                    self._clear_selection()
                    return
                move = chess.Move(from_sq, to_sq, promotion=promo_piece)
            # If still illegal, abort
            if move not in self.board.legal_moves:
                self._clear_selection()
                return

        # Push move (handles castling, en passant, all legality)
        self.board.push(move)
        self.last_move = move
        self._clear_selection()

        # Check game end states (includes 50-move, repetitions when claimable)
        self.post_move_checks()

    def post_move_checks(self):
        # Automatic end if forced (checkmate/stalemate, insufficient material, fivefold, 75-move)
        # Note: For 50-move and threefold, they are claimable; we respect claims via button and
        # also show status. But is_game_over(claim_draw=True) will treat claimable as game over.
        if self.board.is_game_over(claim_draw=True):
            result = self.board.result(claim_draw=True)
            reason = self.board.outcome(claim_draw=True)
            msg = f"Game over: {result}"
            if reason:
                msg += f"\nReason: {reason.termination.name.replace('_', ' ').title()}"
            messagebox.showinfo("Game Over", msg)

    def _is_promotion_move(self, from_sq, to_sq):
        piece = self.board.piece_at(from_sq)
        if not piece or piece.piece_type != chess.PAWN:
            return False
        to_rank = chess.square_rank(to_sq)
        return (to_rank == 7 and self.board.turn is chess.WHITE) or (to_rank == 0 and self.board.turn is chess.BLACK)

    def _ask_promotion_piece(self):
        # Simple modal with Q/R/B/N buttons
        top = tk.Toplevel(self.root)
        top.title("Promote to")
        top.grab_set()
        choice = {"v": None}

        def set_choice(pt):
            choice["v"] = pt
            top.destroy()

        for (label, pt) in [("Queen", chess.QUEEN),
                            ("Rook", chess.ROOK),
                            ("Bishop", chess.BISHOP),
                            ("Knight", chess.KNIGHT)]:
            tk.Button(top, text=label, width=12, command=lambda p=pt: set_choice(p)).pack(padx=10, pady=6)

        top.wait_window()
        return choice["v"]

    def _clear_selection(self):
        self.selected_square = None
        self.legal_targets_for_selected = set()

    # ---------- Buttons ----------
    def new_game(self):
        self.board.reset()
        self.last_move = None
        self._clear_selection()
        self.render()

    def undo(self):
        # pop once (undo last side's move). Pop twice to undo a full pair if desired.
        if len(self.board.move_stack) > 0:
            self.board.pop()
            self.last_move = self.board.move_stack[-1] if self.board.move_stack else None
            self.render()

    def flip_board(self):
        self.orientation_white_bottom = not self.orientation_white_bottom
        self.render()

    def claim_draw(self):
        # Claim 50-move or threefold if allowed
        msg = []
        if self.board.can_claim_fifty_moves():
            msg.append("50-move rule draw claimed.")
        if self.board.can_claim_threefold_repetition():
            msg.append("Threefold repetition draw claimed.")
        if not msg:
            messagebox.showinfo("Claim Draw", "No claimable draw right now.")
            return
        # To apply the claim, we terminate the game logically. Easiest: inform user and block further moves.
        # A simple way is to show a message and disable clicks by unbinding (optional). Here we just show info.
        messagebox.showinfo("Draw", "\n".join(msg))
        # If you prefer to hard-stop, you can set a flag and ignore further clicks.

    def copy_fen(self):
        fen = self.board.board_fen() + " " + ("w" if self.board.turn else "b") + " " + self.board.castling_xfen() + " " + (self.board.ep_square_name() if self.board.ep_square else "-") + f" {self.board.halfmove_clock} {self.board.fullmove_number}"
        # Simpler: fen = self.board.fen()
        fen = self.board.fen()
        self.root.clipboard_clear()
        self.root.clipboard_append(fen)
        messagebox.showinfo("FEN", "Position FEN copied to clipboard.")

    # ---------- Status ----------
    def _update_status(self):
        turn = "White" if self.board.turn else "Black"
        parts = [f"{turn} to move"]

        if self.board.is_check():
            parts.append("(check)")

        # Inform about claimable draws
        claims = []
        if self.board.can_claim_fifty_moves():
            claims.append("50-move claim available")
        if self.board.can_claim_threefold_repetition():
            claims.append("3-fold claim available")
        if claims:
            parts.append(" | " + ", ".join(claims))

        self.status_var.set(" ".join(parts))
