# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/18 12:26
from typing import Optional, List, Tuple, Generator, Any
import torch
from transformers import PreTrainedTokenizer, LogitsProcessorList
from .generator_base import GeneratorBase
from ..model_zoo.qwen.qwen_generation_utils import HistoryType, make_context, get_stop_words_ids, decode_tokens, \
    StopWordsLogitsProcessor


class Generate(GeneratorBase):
    @torch.no_grad()
    def chat(self,
        query: str,
        history: Optional[HistoryType],
        system: str = "You are a helpful assistant.",
        append_history: bool = True,
        stop_words_ids: Optional[List[List[int]]] = None,
        **kwargs
    ) -> Tuple[str, HistoryType]:

        assert self.generation_config.chat_format == 'chatml'

        if history is None:
            history = []

        if stop_words_ids is None:
            stop_words_ids = []

        max_window_size = kwargs.pop('max_window_size', None)
        if max_window_size is None:
            max_window_size = self.generation_config.max_window_size
        raw_text, context_tokens = make_context(
            self.tokenizer,
            query,
            history=history,
            system=system,
            max_window_size=max_window_size,
            chat_format=self.generation_config.chat_format,
        )

        stop_words_ids.extend(get_stop_words_ids(
            self.generation_config.chat_format, self.tokenizer
        ))
        input_ids = torch.tensor([context_tokens]).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            stop_words_ids=stop_words_ids,
            return_dict_in_generate=False,
            **kwargs,
        )

        response = decode_tokens(
            outputs[0],
            self.tokenizer,
            raw_text_len=len(raw_text),
            context_length=len(context_tokens),
            chat_format=self.generation_config.chat_format,
            verbose=False,
            errors='replace',
        )

        if append_history:
            history.append((query, response))

        return response, history

    @torch.no_grad()
    def chat_stream(
            self,
            tokenizer: PreTrainedTokenizer,
            query: str,
            history: Optional[HistoryType],
            system: str = "You are a helpful assistant.",
            **kwargs
    ) -> Generator[str, Any, None]:

        assert self.generation_config.chat_format == 'chatml'
        if history is None:
            history = []

        stop_words_ids = kwargs.pop('stop_words_ids',[])
        if not isinstance(stop_words_ids,list):
            stop_words_ids = [stop_words_ids]



        max_window_size = kwargs.get('max_window_size', None)
        if max_window_size is None:
            max_window_size = self.generation_config.max_window_size
        logits_processor = kwargs.pop('logits_processor',None)
        raw_text, context_tokens = make_context(
            tokenizer,
            query,
            history=history,
            system=system,
            max_window_size=max_window_size,
            chat_format=self.generation_config.chat_format,
        )

        stop_words_ids.extend(get_stop_words_ids(
            self.generation_config.chat_format, tokenizer
        ))
        if stop_words_ids is not None:
            stop_words_logits_processor = StopWordsLogitsProcessor(
                stop_words_ids=stop_words_ids,
                eos_token_id=self.generation_config.eos_token_id,
            )
            if logits_processor is None:
                logits_processor = LogitsProcessorList([stop_words_logits_processor])
            else:
                logits_processor.append(stop_words_logits_processor)
        input_ids = torch.tensor([context_tokens]).to(self.model.device)

        from transformers_stream_generator.main import NewGenerationMixin, StreamGenerationConfig
        self.__class__.generate_stream = NewGenerationMixin.generate
        self.__class__.sample_stream = NewGenerationMixin.sample_stream
        stream_config = StreamGenerationConfig(**self.generation_config.to_dict(), do_stream=True)

        def stream_generator():
            outputs = []
            for token in self.model.generate_stream(
                    input_ids,
                    return_dict_in_generate=False,
                    generation_config=stream_config,
                    logits_processor=logits_processor,
                    **kwargs):
                outputs.append(token.item())
                yield tokenizer.decode(outputs, skip_special_tokens=True, errors='ignore')
        return stream_generator()