import re
from string import ascii_lowercase

import torch
import os
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, build_ctcdecoder

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, vocab_path=None, lm_path=None, beam_size=5, beam_search=False, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """
        
        lm_lowercase_path = 'lowercase_3-gram.pruned.1e-7.arpa'
        if not os.path.exists(lm_lowercase_path):
            with open(lm_path, 'r') as f_upper:
                with open(lm_lowercase_path, 'w') as f_lower:
                    for line in f_upper:
                        f_lower.write(line.lower())

        if vocab_path:
            with open(vocab_path) as file:
                unigrams = [c.lower() for c in file.read().strip().split("\n")]

        self.beam_width = beam_size
        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        if lm_path:
            self.decoder = build_ctcdecoder(labels=self.vocab, kenlm_model_path=lm_lowercase_path, unigrams=unigrams)
        elif beam_search:
            self.decoder = BeamSearchDecoderCTC(Alphabet(labels=self.vocab, is_bpe=False), language_model=None)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        decoded = []
        last_char_ind = self.char2ind[self.EMPTY_TOK]
        for ind in inds:
            if last_char_ind == ind:
                continue
            if ind != self.char2ind[self.EMPTY_TOK]:
                decoded.append(self.ind2char[ind])
            last_char_ind = ind

        return "".join(decoded)
    
    def ctc_beam_search_decode_vanila(self, probs, beam_size=5):
        probs = torch.exp(probs)
        dp = {
            ("", self.EMPTY_TOK): 1.0,
        }
        for prob in probs:
            dp = self._expand_and_merge_path(dp, prob)
            dp = self._truncate_paths(dp, beam_size)
        dp = [
            (prefix, proba)
            for (prefix, _), proba in sorted(dp.items(), key=lambda x: -x[1])
        ]
        return dp[0][0]
    
    def _expand_and_merge_path(self, dp, next_token_probs):
        new_dp = defaultdict(float)
        for ind, next_token_prob in enumerate(next_token_probs):
            cur_char = self.ind2char[ind]
            for (prefix, last_char), v in dp.items():
                if last_char == cur_char:
                    new_prefix = prefix
                else:
                    if cur_char != self.EMPTY_TOK:
                        new_prefix = prefix + cur_char
                    else:
                        new_prefix = prefix
                new_dp[(new_prefix, cur_char)] += v * next_token_prob
        return new_dp
    
    def _truncate_paths(self, dp, beam_size):
        return dict(sorted(list(dp.items()), key=lambda x: -x[1])[:beam_size])

    def ctc_beam_search_decode(self, inds) -> str:
        return self.decoder.decode(inds, beam_width=self.beam_width)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
