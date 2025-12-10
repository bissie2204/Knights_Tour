"""
Abigail Tamburello
Languages & Paradigms Final Project

knight_tour_warnsdorff.py

Solves the Knight's Tour problem on an 8x8 board using Warnsdorff's heuristic,
visualizes the path using matplotlib, accepts user-selected starting positions, 
and prints statistics (steps, runtime).
"""
import argparse
import math
import time
import random
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Constants
# Size of the chessboard
BOARD_SIZE = 8
Square = Tuple[int, int]  # (row, col), 0-indexed

# Possible knight moves
KNIGHT_MOVES = [
    (-2, -1), (-1, -2), (1, -2), (2, -1),
    (2, 1), (1, 2), (-1, 2), (-2, 1),
]

# Utility Functions
# Check if a square is within the bounds of the chessboard
def in_bounds(square: Square) -> bool:
    r, c = square
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

# Convert algebraic notation to square
def algebraic_to_square(s: str) -> Square:
    """
    Convert algebraic notation like 'e4' to 0-indexed (row, col).
    Rank '1' is bottom row -> row index 7, so 'a1' is (7,0).
    """
    if len(s) != 2:
        raise ValueError("Algebraic square must be length 2, like 'e4'.")
    file_char = s[0].lower()
    rank_char = s[1]
    col = ord(file_char) - ord('a')
    rank = int(rank_char)
    row = BOARD_SIZE - rank
    if not in_bounds((row, col)):
        raise ValueError(f"Algebraic square out of bounds: {s}")
    return (row, col)

# Convert square to algebraic notation
def square_to_algebraic(square: Square) -> str:
    r, c = square
    file_char = chr(ord('a') + c)
    rank = BOARD_SIZE - r
    return f"{file_char}{rank}"

# Get valid knight-move neighbors
def neighbors(square: Square) -> List[Square]:
    r, c = square
    result = []
    for dr, dc in KNIGHT_MOVES:
        nxt = (r + dr, c + dc)
        if in_bounds(nxt):
            result.append(nxt)
    return result

# Warnsdorff's heuristic to find the next move
def warnsdorff_tour(start: Square, board_size: int = BOARD_SIZE, randomize_ties: bool = True) -> Optional[List[Square]]:
    """
    Attempt to produce a full knight's tour starting from `start` using Warnsdorff's heuristic:
      - Move to square with the minimum number of onward moves (degree).
      - Tie-breaking: deterministic (first) or randomized if randomize_ties True.
    Returns the path as list of squares if successful, or None if failed.
    """

    N = board_size
    total_squares = N * N

    # Initialize visited matrix and path
    visited = [[False] * N for _ in range(N)]
    path: List[Square] = [start]
    visited[start[0]][start[1]] = True

    # Function to compute the degree of a square
    def degree(sq: Square) -> int:
        cnt = 0
        for nb in neighbors(sq):
            if not visited[nb[0]][nb[1]]:
                cnt += 1
        return cnt

    # Main loop to build the tour
    for step in range(1, total_squares):
        current = path[-1]
        next_moves = []
        for nb in neighbors(current):
            if not visited[nb[0]][nb[1]]:
                next_moves.append((degree(nb), nb))

        if not next_moves:
            return None

        # Sort moves by degree
        next_moves.sort(key=lambda x: x[0])
        # Find minimal degree
        min_degree = next_moves[0][0]
        # Select all candidates with minimal degree
        candidates = [mv for deg, mv in next_moves if deg == min_degree]

        # Tie-breaking
        if randomize_ties and len(candidates) > 1:
            chosen = random.choice(candidates)
        else:
            chosen = candidates[0]

        path.append(chosen)
        visited[chosen[0]][chosen[1]] = True

    return path

# Retry mechanism for Warnsdorff's heuristic
def find_tour_with_retries(start: Square, max_retries: int = 2000) -> Tuple[Optional[List[Square]], int]:
    """
    Each trial uses a different RNG seed/state. If success, return path and the number of attempts used. 
    If not found, return (None, attempts).
    """
    attempts = 0
    while attempts < max_retries:
        attempts += 1
        # Randomize global state per attempt for different tie choices
        # Keep deterministic reproducibility optionally by seeding here if desired.
        path = warnsdorff_tour(start, randomize_ties=True)
        if path is not None:
            return path, attempts
    return None, attempts

# Visualization function
def draw_board(ax):
    #Draw 8x8 checkerboard on matplotlib Axes
    ax.set_xlim(0, BOARD_SIZE)
    ax.set_ylim(0, BOARD_SIZE)
    ax.set_xticks(np.arange(0, BOARD_SIZE + 1))
    ax.set_yticks(np.arange(0, BOARD_SIZE + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    # Draw squares
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            color = '#f0d9b5' if (r + c) % 2 == 0 else '#b58863'
            # Draw from bottom-left corner
            rect = patches.Rectangle((c, BOARD_SIZE - r - 1), 1, 1, facecolor=color)
            ax.add_patch(rect)
    # Grid lines
    for x in range(BOARD_SIZE + 1):
        ax.plot([x, x], [0, BOARD_SIZE], color='k', linewidth=0.5)
    for y in range(BOARD_SIZE + 1):
        ax.plot([0, BOARD_SIZE], [y, y], color='k', linewidth=0.5)
        
# Visualization function for the knight's tour path
def visualize_path(path: List[Square], title: str = "Knight's Tour") -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_board(ax)

    # Coordinates for plotting: convert board (row,col) to (x,y) 
    x_coords = []
    y_coords = []
    for (r, c) in path:
        x = c + 0.5
        y = (BOARD_SIZE - r - 1) + 0.5
        x_coords.append(x)
        y_coords.append(y)

    # Draw connecting path
    ax.plot(x_coords, y_coords, linewidth=2, marker='o')

    # Annotate step numbers inside the squares
    for step, (x, y) in enumerate(zip(x_coords, y_coords), start=1):
        ax.text(x, y, str(step), va='center', ha='center', fontsize=8, color='white', weight='bold')

    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# Argument parsing
# Parse command-line arguments and user-ineraction for starting square
def parse_args():
    parser = argparse.ArgumentParser(description="Knight's Tour (Warnsdorff) - 8x8")
    parser.add_argument('--start', nargs='+', help="Starting square: algebraic like e4 OR two ints row col (0-indexed). If omitted, choose interactively by clicking.")
    parser.add_argument('--retries', type=int, default=2000, help="Max retries with randomized tie breaking (default 2000).")
    parser.add_argument('--no-visual', action='store_true', help="Do not show visualization.")
    return parser.parse_args()

def interactive_pick_start() -> Square:
    # Show a static board and let user click to choose start square. 
    picked = {}

    fig, ax = plt.subplots(figsize=(6, 6))
    draw_board(ax)
    ax.set_title("Click a square to select starting position for the knight")
    plt.tight_layout()

    # Click event handler
    def onclick(event):
        # Map click (x,y) in figure to board coordinates
        if event.xdata is None or event.ydata is None:
            return
        col = int(event.xdata)  # 0..7
        y = event.ydata
        # Convert y back to row: y is 0..8 bottom->top; our row 0 is top
        row = BOARD_SIZE - int(y) - 1
        if in_bounds((row, col)):
            picked['square'] = (row, col)
            plt.close(fig)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    if 'square' not in picked:
        raise RuntimeError("No square picked.")
    return picked['square']

# Main function
def main():
    args = parse_args()

    # Determine start square
    if args.start is None:
        print("Please click a square in the board.")
        start = interactive_pick_start()
    else:
        tokens = args.start
        if len(tokens) == 1:
            token = tokens[0]
            try:
                start = algebraic_to_square(token)
            except Exception as e:
                raise ValueError(f"Invalid start token '{token}'. Use algebraic (e4) or two integers row col.") from e
        elif len(tokens) == 2:
            try:
                r = int(tokens[0])
                c = int(tokens[1])
                if not in_bounds((r, c)):
                    raise ValueError("Row/col out of bounds (0-7).")
                start = (r, c)
            except ValueError as e:
                raise ValueError("If specifying two values, they must be integers row col (0-indexed).") from e
        else:
            raise ValueError("start parameter must be algebraic 'e4' or two integers 'row col'.")

    print(f"Starting square: {square_to_algebraic(start)} (row={start[0]}, col={start[1]})")
    t0 = time.time()
    path, attempts = find_tour_with_retries(start, max_retries=args.retries)
    t1 = time.time()
    elapsed = t1 - t0

    if path is None:
        print(f"Failed to find a full tour after {attempts} attempts.")
        print(f"Elapsed time: {elapsed:.4f} seconds.")
        return

    # Statistics
    steps = len(path)
    is_full = (steps == BOARD_SIZE * BOARD_SIZE)
    print("=== Result statistics ===")
    print(f"Full tour found: {is_full}")
    print(f"Steps in path: {steps}")
    print(f"Attempts (randomized tie-break trials): {attempts}")
    print(f"Elapsed time: {elapsed:.6f} seconds")

    # Print path in algebraic notation, line-wrapped
    alg_path = [square_to_algebraic(sq) for sq in path]
    print("Path (algebraic):")
    # wrap 16 per line
    for i in range(0, len(alg_path), 16):
        print("  " + " ".join(alg_path[i:i + 16]))

    # Visualize unless suppressed
    if not args.no_visual:
        visualize_path(path, title=f"Knight's Tour starting {square_to_algebraic(start)} (attempts={attempts}, time={elapsed:.3f}s)")


if __name__ == "__main__":
    main()