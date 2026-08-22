import chess
import numpy as np
import torch

def to_tensor(board, candidate_moves, evals, black):

    board_planes = np.zeros((12, 8, 8), dtype=np.float32)

    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = chess.square_rank(square)
            col = chess.square_file(square)
            plane = (piece_map[piece.symbol()] + 6 * black) % 12
            board_planes[plane, row, col] = 1.0

    board_tensor = torch.from_numpy(board_planes)

    move_features = []

    while len(candidate_moves) < 13:
            candidate_moves.append("0000")
            evals.append(0.0)



    for i, move in enumerate(candidate_moves):

        is_padding = float(move == chess.Move.from_uci("0000"))
        is_mate = float(evals[i] == None)

        is_capture = float(board.is_capture(move))
        # board.push(move)
        # is_check = float(board.is_check())
        # board.pop()
        is_check = float(board.gives_check(move))

        moving_piece = board.piece_at(move.from_square)

        moving_onehot = np.zeros(6, dtype=np.float32)
        if moving_piece:
            moving_onehot[moving_piece.piece_type - 1] = 1.0

        if board.is_en_passant(move):
            captured_type = chess.PAWN
        else:
            captured = board.piece_at(move.to_square)
            captured_type = captured.piece_type if captured else None

        capture_onehot = np.zeros(7, dtype=np.float32)
        capture_onehot[0 if captured_type is None else captured_type] = 1.0

        promotion_onehot = np.zeros(5, dtype=np.float32)
        promotion_onehot[0 if move.promotion is None else {
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4
        }[move.promotion]] = 1.0

        features = np.concatenate([
            np.array([
                chess.square_rank(move.from_square),
                chess.square_file(move.from_square),
                chess.square_rank(move.to_square),
                chess.square_file(move.to_square),
                evals[i],
                is_padding,
                is_mate,
                is_capture,
                is_check
            ], dtype=np.float32),
            moving_onehot,
            capture_onehot,
            promotion_onehot
        ])

        move_features.append(features)

    move_tensor = torch.tensor(
        np.asarray(move_features),
        dtype=torch.float32
    )

    return board_tensor, move_tensor