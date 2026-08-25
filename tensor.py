import chess
import numpy as np
import torch

NUM_CANDIDATES = 13
EVAL_SCALE = 400.0
DUMMY_MOVE = chess.Move.null()

def normalize_eval(x):
    if x is None:
        return 0.0
    return float(np.tanh(x / EVAL_SCALE))

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

    candidate_moves = list(candidate_moves)
    evals = list(evals)

    if len(evals) < len(candidate_moves):
        evals.extend([None] * (len(candidate_moves) - len(evals)))
    elif len(evals) > len(candidate_moves):
        evals = evals[:len(candidate_moves)]

    candidate_moves = candidate_moves[:NUM_CANDIDATES]
    evals = evals[:NUM_CANDIDATES]

    while len(candidate_moves) < NUM_CANDIDATES:
        candidate_moves.append(DUMMY_MOVE)
        evals.append(None)

    move_features = []

    for move, raw_eval in zip(candidate_moves, evals):
        is_padding = float(move == DUMMY_MOVE)
        is_mate = float(raw_eval is None and not is_padding)

        evaluation = normalize_eval(raw_eval)
        if is_padding:
            evaluation = 0.0

        if is_padding:
            from_rank = from_file = to_rank = to_file = 0
            is_capture = 0.0
            is_check = 0.0
            moving_piece = None
        else:
            from_rank = chess.square_rank(move.from_square)
            from_file = chess.square_file(move.from_square)
            to_rank = chess.square_rank(move.to_square)
            to_file = chess.square_file(move.to_square)
            is_capture = float(board.is_capture(move))
            is_check = float(board.gives_check(move))
            moving_piece = board.piece_at(move.from_square)

        moving_onehot = np.zeros(6, dtype=np.float32)
        if moving_piece:
            moving_onehot[moving_piece.piece_type - 1] = 1.0

        if is_padding:
            captured_type = None
        elif board.is_en_passant(move):
            captured_type = chess.PAWN
        else:
            captured = board.piece_at(move.to_square)
            captured_type = captured.piece_type if captured else None

        capture_onehot = np.zeros(7, dtype=np.float32)
        capture_onehot[0 if captured_type is None else captured_type] = 1.0

        promotion_onehot = np.zeros(5, dtype=np.float32)
        promotion_index = 0 if move.promotion is None else {
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4
        }[move.promotion]
        promotion_onehot[promotion_index] = 1.0

        features = np.concatenate([
            np.array([from_rank, from_file, to_rank, to_file, evaluation, is_padding, is_mate, is_capture, is_check], dtype=np.float32),
            moving_onehot,
            capture_onehot,
            promotion_onehot
        ])

        move_features.append(features)

    move_tensor = torch.from_numpy(np.asarray(move_features, dtype=np.float32))

    return board_tensor, move_tensor