# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/7/26 10:29
import typing
from typing import Callable
from transformers import TextStreamer


class GenTextStreamer(TextStreamer):
    def __init__(self,
                 process_token_fn: Callable,
                 fn_args,
                 tokenizer,
                 skip_word_list=None,
                 skip_prompt: bool = False,
                 on_filter_fn: typing.Optional[Callable]=None,
                 **decode_kwargs):
        super().__init__(tokenizer,skip_prompt,**decode_kwargs)
        self.process_token_fn = process_token_fn
        self.fn_args = fn_args
        self.on_filter_fn = on_filter_fn
        if skip_word_list is not None:
            skip_word_list = list(set(skip_word_list))
        self.skip_word_list = skip_word_list
        self.all_ids = []



    def put(self, value):
        """
        Recives tokens, decodes them, and prints them to stdout as soon as they form entire words.
       """
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        value_ids = value.tolist()
        if self.on_filter_fn is not None and self.on_filter_fn(self,value_ids):
            return

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        if self.skip_word_list is not None:
            if value_ids in self.skip_word_list:
                return
            if isinstance(value_ids,list):
                value_ids = [v for v in value_ids if v not in self.skip_word_list]
                if len(value_ids) == 0:
                    return
            else:
                if value_ids in self.skip_word_list:
                    return

        self.all_ids.append(value_ids)
        # Add the new token to the cache and decodes the entire thing.
        self.token_cache.extend(value_ids)
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)

        # After the symbol for a new line, we flush the cache.
        if text.endswith("\n"):
            printable_text = text[self.print_len:]
            self.token_cache = []
            self.print_len = 0
        # If the last token is a CJK character, we print the characters.
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len:]
            self.print_len += len(printable_text)
        # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
        # which may change with the subsequent token -- there are probably smarter ways to do this!)
        else:
            printable_text = text[self.print_len: text.rfind(" ") + 1]
            self.print_len += len(printable_text)

        self.on_finalized_text(printable_text)



    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Prints the new text to stdout. If the stream is ending, also prints a newline."""
        # print(text, flush=True, end="" if not stream_end else None)
        self.process_token_fn(text,stream_end,self.fn_args)