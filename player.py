import chess
import random
import re
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from chess_tournament.players import Player

class TransformerPlayer(Player):
    """
        REQUIRED:
        Subclasses chess_tournament.players.Player
    """
    def __init__(self, name: str, model_path: str = "we0rr9u89q/chess_gpt2_results", tokenizer=None):
        super().__init__(name)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
        self.uci_re = re.compile(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b")
        self.retries = 5
        self.temperature = 0.1
        self.max_new_tokens = 100

    def _build_prompt(self, fen: str) -> str:
      return f"""You are a chess engine.\n\nYour task is to output the BEST LEGAL MOVE for the given chess position.\n\nSTRICT OUTPUT RULES:\n- Output EXACTLY ONE move\n- UCI format ONLY (examples: e2e4, g1f3, e7e8q)\n- NO explanations:\n- NO punctuation:\n- NO extra text:\n\nExamples:\n\nFEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1\nMove: e2e4\n
FEN: r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3\nMove: f1b5\n
FEN: rnbqkb1r/pppp1ppp/5n2/4p3/1P6/5NP1/P1PPPP1P/RNBQKB1R b KQkq - 0 3\nMove: e5e4\n
Now evaluate this position:\n\nFEN: {fen}\nMove:"""

    def _extract_move(self, text: str) -> Optional[str]:
        print("Extracted move:", text[:4])
        text = text[:4]
        match = self.uci_re.search(text)
        return match.group(0) if match else None

    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        prompt = self._build_prompt(fen)
        chosen_move = None

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=True,
                temperature=0.7,
                num_return_sequences=30, 
                pad_token_id=self.tokenizer.pad_token_id
            )

        for output in outputs:
            decoded = self.tokenizer.decode(output, skip_special_tokens=True)
            move_candidate = decoded.replace(prompt, "").strip().split()[0]

            move_candidate = self._extract_move(move_candidate)

            if move_candidate:
                try:
                    move_obj = chess.Move.from_uci(move_candidate)
                    if move_obj in board.legal_moves:
                        chosen_move = move_candidate
                        break
                except:
                    continue

        if chosen_move is None:
            print(f"TransformerPlayer {self.name} returned None.")
            return None
        else:
            print(f"TransformerPlayer {self.name} generated move: {chosen_move}")
            return chosen_move






