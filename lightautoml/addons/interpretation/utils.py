from typing import List, Tuple, Union, Any

import numpy as np

from collections import defaultdict
import itertools

import matplotlib.pyplot as plt


T_untokenized = Union[List[str], Tuple[List[str], List[Any]]]

def untokenize(raw: str, tokens: List[str],
               return_mask: bool = False,
               token_sym: Any = True,
               untoken_sym: Any = False) -> T_untokenized:
    """Get between tokens symbols.
    
    Args:
        raw: Raw string.
        tokens: List of tokens from raw string.
        return_mask: Flag to return mask 
            for each new token. Format: list of
            `token_sym`, `untoken_sym`.
        token_sym: Object, denote token symbol.
        untoken_sym: Object, denote untoken symbol.
    
    Returns:
        Tuple (full_tokens, tokens_mask) if `return_mask=True`,
            else just list full_tokens.
        
    """
    mask = []
    untokenized = []
    pos = raw.find(tokens[0])
    
    if pos != 0:
        untokenized.append(raw[:pos])
        mask.append(untoken_sym)
        raw = raw[pos:]
    
    prev_token = tokens[0]
    for token in tokens[1:]:
        raw = raw[len(prev_token):]
        pos = raw.find(token)
        untokenized.append(prev_token)
        mask.append(token_sym)
        if pos:
            mask.append(untoken_sym)
            untokenized.append(raw[:pos])
        prev_token = token
        raw = raw[pos:]
        
    untokenized.append(prev_token)
    mask.append(token_sym)
    
    cur = len(prev_token)
    if cur != len(raw):
        untokenized.append(raw[cur:])
        mask.append(untoken_sym)
    
    if return_mask:
        return untokenized, mask
    
    return untokenized


def find_positions(arr: List[str],
                   mask: List[bool]) -> List[int]:
    """Set positions and tokens.
    
    Args:
        tokens: List of tokens and untokens.
        mask: Mask for tokens. 
        
    Returns:
        List of positions of tokens.
        
    """
    pos = []
    for i, (token, istoken) in enumerate(zip(arr, mask)):
        if istoken:
            pos.append(i)
    return pos


class IndexedString:
    """Indexed string."""
    
    def __init__(self, raw_string: str, tokenizer: Any, force_order: bool = True):
        """
        Args:
            raw_string: Raw string.
            tokenizer: Tokenizer class.
            force_order: Save order, or use features as
                bag-of-words. 
        
        """
        self.raw = raw_string
        self.tokenizer = tokenizer
        self.force_order = force_order
        
        self.toks_ = self._tokenize(raw_string)
        self.toks = [token.lower() for token in self.toks_]
        self.as_list_, self.mask = untokenize(
            self.raw, self.toks_, return_mask=True)
        
        self.pos = find_positions(self.as_list_, self.mask)
        self.as_np_ = np.array(self.as_list_)
        self.inv = []
        if not force_order:
            pos = defaultdict(list)
            self.vocab = {}
            for token, cur in zip(self.toks, self.pos):
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
                    self.inv.append(token)
                idx = self.vocab[token]
                pos[idx].append(cur)
            self.pos = pos
        else:
            self.inv = self.toks_
        
        
    def _tokenize(self, text: str) -> List[str]:
        prep_text = self.tokenizer._tokenize(text)
        tokens = self.tokenizer.tokenize_sentence(text)
        
        return tokens
    
    def word(self, idx: int) -> str:
        """Token by its index.
        
        Args:
            idx: Index of token.
            
        Returns:
            Token.
            
        """
        return self.inv[idx]
    
    def inverse_removing(self, to_del: Union[List[str], List[int]],
                         by_tokens=False) -> str:
        """Remove tokens.
        
        Args:
            to_del: Tokens (text of int) to del.
            by_tokens: Flag if tokens are text or indexes.
            
        Returns:
            String without removed tokens.
        
        """
        
        
        # todo: this type of mapping will be not use order,
        # in case when we have not unique tokens.
        assert (not self.force_order) or \
            (self.force_order and not by_tokens)
        
        if not self.force_order:
            if by_tokens:
                to_del = [self.t_i[token.lower()] for token in to_del]
                to_del = np.array(to_del)
            to_del = list(itertools.chain.from_iterable(
                [self.pos[i] for i in to_del]))
        else:
            to_del = [self.pos[i] for i in to_del]    
        mask = np.ones_like(self.as_np_, dtype=bool)
        mask[to_del] = False
        new_str = ''.join(self.as_np_[mask])
        
        return new_str
            
    @property
    def n_words(self) -> int:
        """Number of unique words."""
        return len(self.pos)
    



def draw_html(tokens_and_weights: List[Tuple[str, float]],
              cmap: Any = plt.get_cmap("bwr"),
              token_template: str = """<span style="background-color: {color_hex}">{token}</span>""",
              font_style: str = "font-size:14px;"
             ) -> str:
    """Get colored text in html format.
    
    For color used gradient from cmap.
    To normalize weights sigmoid is used.
    
    Args:
        tokens_and_weights: List of tokens. 
        cmap: ```matplotlib.colors.Colormap``` object.
        token_template: Template for coloring the token.
        font_style: Styling properties of html.
        
    Returns:
        HTML like string.
    
    """
    def get_color_hex(weight):
        rgba = cmap(1. / (1 + np.exp(weight)), bytes=True)
        return '#%02X%02X%02X' % rgba[:3]
    
    tokens_html = [
        token_template.format(token=token, color_hex=get_color_hex(weight))
        for token, weight in tokens_and_weights
    ]
    raw_html = """<p style="{}">{}</p>""".format(font_style, ' '.join(tokens_html))
    
    return raw_html
    
        
    