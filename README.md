🌐 run command : 
.\.venv\Scripts\activate
python main.py

📂 folder structure information:
ai-chess-engine/
├── main.py                 # Launches UI/tournaments
├── engine/                 # Core engine logic
│   ├── board.py            # Bitboard representation
│   ├── movegen.py          # Magic bitboard move generation
│   ├── search.py           # Parallel MCTS+Negascout
│   ├── eval.py             # NNUE evaluation
│   ├── book.py            # Polyglot opening book
│   └── protocols.py       # UCI communication
├── tournaments/
│   ├── manager.py         # Engine vs Engine testing
│   └── elo.py            # Stockfish comparison
└── ui/
    ├── board_gui.py       # Pygame interface
    └── assets/            # Piece images



    sudo apt-get install cutechess