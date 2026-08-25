# Chess AI — Playing in My Style

A chess-playing system trained to imitate **my personal playing style**, rather than simply playing the strongest move possible.

The system combines a **neural network trained on my games** with a separate **opening book**, allowing it to reproduce both my opening preferences and my decision-making tendencies in later positions.

> 🎥 **[Watch me play against my AI](https://youtu.be/JJwmXE_iy6k)**

> 📓 **[Training & Evaluation Colab](https://colab.research.google.com/drive/1M6mISFVJWEt-l2JkkvvgS065e8WuqtjI#scrollTo=tjiYicMMTXAGOkay)**

---

## Why Is This Difficult?

Playing strong chess and playing *like a specific person* are two very different problems.

Even commercial chess bots trained on **200,000+ games from individual Grandmasters** struggle to convincingly reproduce their player's style. This project attempts the same problem with only **~5,000 of my own games**, spanning roughly five years of changing playing strength and style.

The goal is therefore not simply:

> *"What is the best move?"*

but rather:

> *"What move would I be likely to play?"*

This makes the problem much closer to **behavioral imitation** than conventional chess engine development.

---

## Model

The neural network receives:

- A 12-plane representation of the chess board
- Up to 13 Stockfish candidate moves
- Move coordinates
- Moving and captured piece information
- Promotion information
- Stockfish evaluation as an additional feature
- Padding/mate indicators for variable candidate sets
- check/capture indicators 

Stockfish is used to generate candidate moves, while the neural network learns which candidate most resembles my decisions.

An **opening book** is used separately to reproduce my opening repertoire and help guide the neural network into positions similar to those present in my training data.

---

## Results

Evaluation was performed on **10,055 positions** using games held out from training.

| Metric | Result |
|---|---:|
| Top-1 accuracy | **40.55%** |
| Mean P(actual move) | **0.3112** |
| Median P(actual move) | **0.2128** |
| Mean Reciprocal Rank | **0.6137** |

Top-1 accuracy measures how often my actual move was the model's first choice among the Stockfish candidate moves.

The full evaluation is available here:

> 📊 **[Validation Results](val/results.txt)**

---

## Technical Challenges

Several aspects of the problem required special handling:

- My games span five years, during which my playing style changed substantially.
- Training/validation splits had to be performed **by game rather than by position** to avoid highly correlated positions leaking between sets.
- The number of legal/candidate moves is variable, requiring explicit handling of padding and forced moves.
- Data augmentation can substantially increase the training set through board transformations.
- An opening book introduces additional problems around transpositions and position history.
- Chess style is difficult to measure directly, so evaluation focuses on whether my actual moves appear among the model's top candidate predictions.
- Blunders are relatively rare in my games, making deliberate imitation of my mistakes surprisingly difficult.

More details and experiments are documented separately.

> 📄 **[Technical Report](tech_report.pdf)**

---
├── openingBook.py      # Opening book implementation
├── custom_opening_book.pkl
└── tech report.txt     # Detailed technical report
