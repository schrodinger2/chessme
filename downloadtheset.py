import os
import urllib.request

output_dir = "chess_pieces"
os.makedirs(output_dir, exist_ok=True)

# Direct GitHub raw URLs for Lichess 'cburnett' SVGs
base_url = "https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/cburnett"

# Lichess uses lowercase letters: wP, wN, wB, wR, wQ, wK / bP, bN, bB, bR, bQ, bK
pieces = ["wP", "wN", "wB", "wR", "wQ", "wK", "bP", "bN", "bB", "bR", "bQ", "bK"]

headers = {"User-Agent": "Mozilla/5.0"}

print("Downloading piece SVG images from Lichess repo...")
for piece in pieces:
    url = f"{base_url}/{piece}.svg"
    save_path = os.path.join(output_dir, f"{piece}.svg")

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response, open(save_path, "wb") as out_file:
        out_file.write(response.read())

print(f"Done! Saved 12 SVG pieces to '{output_dir}/'")