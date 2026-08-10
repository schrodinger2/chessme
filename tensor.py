import chess
import numpy as np
import torch

def to_tensor(board: chess.Board,candidate_moves: list[chess.Move], eval, black: bool):
    """
    Returns
    board_tensor : (12, 8, 8)
    move_tensor : (NUM_CANDIDATE_MOVES, 22)

    Move features: [from_rank, from_file, to_rank, to_file, eval_diff, moving_piece(6 one-hot), captured_piece(7 one-hot), promotion(5 one-hot)]
    """
    while len(candidate_moves) < 13:
        candidate_moves.append(candidate_moves[-1])
        eval.append(-10000)
    # BOARD ENCODING

    board_planes = np.zeros((12, 8, 8), dtype=np.float32)

    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        row = chess.square_rank(square)
        col = chess.square_file(square)
        plane = (piece_map[piece.symbol()] + 6 * black) % 12

        board_planes[plane, row, col] = 1.0

    board_tensor = torch.from_numpy(board_planes)

    # MOVE ENCODING\

    move_features = []

    for idx ,move in enumerate(candidate_moves):

        from_rank = chess.square_rank(move.from_square)
        from_file = chess.square_file(move.from_square)

        to_rank = chess.square_rank(move.to_square)
        to_file = chess.square_file(move.to_square)

        # moving piece

        moving_piece = board.piece_at(move.from_square)

        moving_onehot = np.zeros(6, dtype=np.float32)

        if moving_piece is not None:
            moving_onehot[moving_piece.piece_type - 1] = 1.0

        # captured piece

        if board.is_en_passant(move):
            captured_type = chess.PAWN
        else:
            captured = board.piece_at(move.to_square)
            captured_type = captured.piece_type if captured else None

        capture_onehot = np.zeros(7, dtype=np.float32)

        if captured_type is None:
            capture_onehot[0] = 1.0
        else:
            capture_onehot[captured_type] = 1.0

        # promotion

        promotion_onehot = np.zeros(5, dtype=np.float32)

        if move.promotion is None:
            promotion_onehot[0] = 1.0
        else:
            promotion_onehot[
                {
                    chess.KNIGHT: 1,
                    chess.BISHOP: 2,
                    chess.ROOK: 3,
                    chess.QUEEN: 4
                }[move.promotion]
            ] = 1.0
        # engine eval
        if eval[idx] and eval[0]:
            eval_diff = eval[idx] - eval[0]
        else:
            eval_diff = 10000 #for mate attacks the eval is none
        #############################

        feature = np.concatenate([
            np.array(
                [from_rank,
                 from_file,
                 to_rank,
                 to_file,
                 eval_diff],
                dtype=np.float32
            ),
            moving_onehot,
            capture_onehot,
            promotion_onehot
        ])

        move_features.append(feature)

    move_tensor = torch.tensor(move_features, dtype=torch.float32)

    return board_tensor, move_tensor