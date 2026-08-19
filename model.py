import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessPolicyNet(nn.Module):

    def __init__(self):
        super().__init__()

        # board encoder
        self.conv1 = nn.Conv2d(12, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)

        self.board_fc = nn.Linear(64 * 8 * 8, 256)

        # move encoder
        self.move_fc = nn.Linear(25, 48)

        # combined decision head
        self.head = nn.Sequential(
            nn.Linear(256 + 48, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, board, moves):
        """
        board: (batch, 12, 8, 8)
        moves: (batch, 13, 25)
        """

        batch = board.shape[0]

        # board encoding
        x = F.relu(self.conv1(board))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(batch, -1)
        board_feat = F.relu(self.board_fc(x))  # (batch, 256)

        scores = []

        for i in range(13):

            move = moves[:, i, :]  # (batch, 25)

            move_feat = F.relu(self.move_fc(move))

            combined = torch.cat([board_feat, move_feat], dim=1)

            score = self.head(combined)

            scores.append(score)

        scores = torch.cat(scores, dim=1)  # (batch, 13)

        return scores