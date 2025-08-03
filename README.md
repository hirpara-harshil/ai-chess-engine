🌐 run command : 
.venv\Scripts\activate
python main.py

📂 folder structure information:
ai-chess-engine/
│
├── main.py                # Entry point for the application
├── engine/                # Folder for the chess engine code
│   ├── __init__.py        # Makes the engine folder a package
│   ├── board.py           # Class for the chess board representation
│   ├── move.py            # Class for handling moves
│   ├── engine.py          # Main engine logic
│   ├── evaluation.py      # Evaluation functions
│   ├── search.py          # Search algorithms (e.g., MCTS, Negamax)
│   ├── opening_book.py     # Opening book for fast moves
│   └── utils.py           # Utility functions
│
├── tests/                 # Folder for testing
│   ├── __init__.py
│   ├── test_engine.py      # Tests for the engine
│   └── test_performance.py  # Performance tests
│
└── ui/                    # Folder for the user interface
    ├── __init__.py
    ├── gui.py             # GUI for playing against the engine
    └── assets/            # Assets for the GUI (images, etc.)