#!/usr/bin/env python3
import numpy as np
import cv2
import argparse

def generate_chessboard(inner_rows: int,
                        inner_cols: int,
                        square_size_px: int,
                        output_path: str):
    """
    Generates a chessboard calibration pattern.

    Args:
        inner_rows:     Number of inner corners per column (e.g. 6).
        inner_cols:     Number of inner corners per row    (e.g. 9).
        square_size_px: Size of each square in pixels.
        output_path:    Where to save the pattern (e.g. 'chessboard.png').
    """
    # The board has one more square than inner corners
    rows = inner_rows + 1
    cols = inner_cols + 1

    # Create a blank image
    height = rows * square_size_px
    width  = cols * square_size_px
    board = np.zeros((height, width), dtype=np.uint8)

    # Fill squares
    for r in range(rows):
        for c in range(cols):
            # alternate black/white
            if (r + c) % 2 == 0:
                cv2.rectangle(
                    board,
                    (c * square_size_px, r * square_size_px),
                    ((c + 1) * square_size_px, (r + 1) * square_size_px),
                    255,  # white square
                    thickness=cv2.FILLED
                )

    # Save as PNG (or .pdf, .jpg, etc.)
    cv2.imwrite(output_path, board)
    print(f"✅ Saved chessboard to {output_path} ({width}×{height}px)")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate a chessboard pattern for camera calibration"
    )
    p.add_argument('--rows',   '-r', type=int, required=True,
                   help="Inner corners per column (e.g. 6)")
    p.add_argument('--cols',   '-c', type=int, required=True,
                   help="Inner corners per row    (e.g. 9)")
    p.add_argument('--size',   '-s', type=int, default=100,
                   help="Square size in pixels (default: 100)")
    p.add_argument('--output','-o', default='chessboard.png',
                   help="Output file (e.g. chessboard.png)")
    args = p.parse_args()

    generate_chessboard(
        inner_rows   = args.rows,
        inner_cols   = args.cols,
        square_size_px = args.size,
        output_path  = args.output
    )
