import gc
import math
import os
import random
import shutil
from abc import ABCMeta, abstractmethod
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase
from transformers.generation.utils import GenerateOutput

from .types import ConversationMessage, APIConversationMessage, InferenceEngine, TokenizedConversationOutput

import time
from typing import Tuple
from openai import OpenAI

try:
    import torch_npu
except ImportError:
    torch_npu = None


class BaseChatTemplate(metaclass=ABCMeta):
    @abstractmethod
    def tokenize_conversation_one(
        self,
        message: ConversationMessage,
        tokenizer: PreTrainedTokenizerBase,
        idx: int,
        add_generation_prompt: bool = False,
    ) -> TokenizedConversationOutput:
        raise NotImplementedError

    def tokenize_conversation(
        self,
        conversation: list[ConversationMessage],
        tokenizer: PreTrainedTokenizerBase,
        add_generation_prompt: bool = False,
    ) -> TokenizedConversationOutput:
        text = ""
        input_ids = []
        action_mask = []
        for idx, message in enumerate(conversation):
            res = self.tokenize_conversation_one(
                message, tokenizer, idx, add_generation_prompt and idx == len(conversation) - 1
            )
            text += res["text"]
            input_ids += res["input_ids"]
            action_mask += res["action_mask"]
        return TokenizedConversationOutput(
            {
                "text": text,
                "input_ids": input_ids,
                "action_mask": action_mask,
            }
        )


class Agent:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        chat_template: BaseChatTemplate | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.chat_template = chat_template or Llama2Template()
        self._vllm = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def generate(
        self,
        input,
        generation_config: GenerationConfig,
        refresh_engine: bool = False,
    ):
    
        state = input[-1]["content"]
        print("------------------------state------------------------")
        print(state)
        print(self.tokenizer(state, return_tensors="pt")["input_ids"].size(1))
        print("------------------------state------------------------")
        tokens = self.tokenizer.apply_chat_template(
            input,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=True,  # Switches between thinking and non-thinking modes. Default is True.
        ).to(self.model.device)

        # Move *all* tensor values in inputs to GPU
        # tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.no_grad():
            output = self.model.generate(
                **tokens,
                generation_config=generation_config,
            )
            generated_text = self.tokenizer.decode(output[0][tokens["input_ids"].size(1):], skip_special_tokens=True)

            print("--------------------")
            print(tokens["input_ids"].size(1), output[0][tokens["input_ids"].size(1):].shape)
            print(generated_text)
            return generated_text


class APIAgent:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 1,
        top_p: float = 1,
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        # self.role = {"system": "system", "human": "user", "gpt": "assistant"}

    def generate(
        self,
        conversation: list[APIConversationMessage],
    ) -> Tuple[str, str | None]:
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    # messages=[{"role": self.role[c["role"]], "content": c["value"]} for c in conversation],
                    # messages=conversation,
                    messages=[{"role": c["role"], "content": c["content"]} for c in conversation],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p
                )
                return response.choices[0].message.content, response.choices[0].message.reasoning_content if hasattr(response.choices[0].message, "reasoning_content") else None
            except Exception as e:
                print(e)
                time.sleep(1)


class Llama2Template(BaseChatTemplate):
    def tokenize_conversation_one(
        self,
        message: ConversationMessage,
        tokenizer: PreTrainedTokenizerBase,
        idx: int = -1,
        add_generation_prompt: bool = False,
    ) -> TokenizedConversationOutput:
        """
        This function applied Llama Chat template on the given vicuna-styled conversation message.
        You can provide your own _tokenize_conversation_one to adapt to your own task.
        """
        if message["role"] == "user":
            text = f"<s>[INST] {message['content']} [/INST]"
            input_ids = tokenizer.encode(text, add_special_tokens=False)
        else:
            text = f"{message['content']}</s>"
            input_ids = tokenizer.encode(text, add_special_tokens=False)
            text = f" {text}"
        if message["loss"]:
            action_mask = [1] * len(input_ids)
        else:
            action_mask = [0] * len(input_ids)

        return TokenizedConversationOutput(
            {
                "text": text,
                "input_ids": input_ids,
                "action_mask": action_mask,
            }
        )


class ChatMLTemplate(BaseChatTemplate):
    def tokenize_conversation_one(
        self,
        message: ConversationMessage,
        tokenizer: PreTrainedTokenizerBase,
        idx: int = -1,
        add_generation_prompt: bool = False,
    ) -> TokenizedConversationOutput:
        """
        This function applied Llama Chat template on the given vicuna-styled conversation message.
        You can provide your own _tokenize_conversation_one to adapt to your own task.
        """
        if idx == 0 and message["role"] != "system":
            text = "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n"
        else:
            text = ""
        if add_generation_prompt:
            if message["role"] == "user":
                text += f"<|im_start|>user\n{message['content']}<|im_end|>\n<|im_start|>assistant\n"
                input_ids = tokenizer.encode(text, add_special_tokens=False)
            else:
                text += f"{message['content']}<|im_end|>"
                input_ids = tokenizer.encode(text, add_special_tokens=False)
                # text = f" {text}"
        else:
            if message["role"] == "user":
                text += f"<|im_start|>user\n{message['content']}<|im_end|>\n"
                input_ids = tokenizer.encode(text, add_special_tokens=False)
            else:
                text += f"<|im_start|>assistant\n{message['content']}<|im_end|>"
                input_ids = tokenizer.encode(text, add_special_tokens=False)
                # text = f" {text}"

        if message["loss"]:
            action_mask = [1] * len(input_ids)
        else:
            action_mask = [0] * len(input_ids)

        return TokenizedConversationOutput(
            {
                "text": text,
                "input_ids": input_ids,
                "action_mask": action_mask,
            }
        )


class Llama3Template(BaseChatTemplate):
    def tokenize_conversation_one(
        self,
        message: ConversationMessage,
        tokenizer: PreTrainedTokenizerBase,
        idx: int = -1,
        add_generation_prompt: bool = False,
    ) -> TokenizedConversationOutput:
        val = message["content"]
        while len(val) and val[-1] in [" ", "\n", "\t"]:
            val = val[:-1]
        mfrom = message["role"]
        if add_generation_prompt:
            mfrom = message["role"]
            if idx == 0:
                text = f"<|begin_of_text|>"
            else:
                text = ""
            if mfrom == "user":
                text += f"<|start_header_id|>user<|end_header_id|>\n\n{val}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            elif mfrom == "assistant":
                text += f"{val}<|eot_id|>"
        else:
            if mfrom == "user":
                mfrom = "user"
            elif mfrom == "assistant":
                mfrom = "assistant"
            if idx == 0:
                text = f"<|begin_of_text|><|start_header_id|>{mfrom}<|end_header_id|>\n\n{val}<|eot_id|>"
            else:
                text = f"<|start_header_id|>{mfrom}<|end_header_id|>\n\n{val}<|eot_id|>"

        input_ids = tokenizer.encode(text, add_special_tokens=False)
        if message["loss"]:
            action_mask = [1] * len(input_ids)
        else:
            action_mask = [0] * len(input_ids)
        return TokenizedConversationOutput(
            {
                "text": text,
                "input_ids": input_ids,
                "action_mask": action_mask,
            }
        )


class ChatGLM4Template(BaseChatTemplate):
    def tokenize_conversation_one(
        self,
        message: ConversationMessage,
        tokenizer: PreTrainedTokenizerBase,
        idx: int = -1,
        add_generation_prompt: bool = False,
    ) -> TokenizedConversationOutput:
        val = message["content"]
        if add_generation_prompt:
            mfrom = message["role"]
            if idx == 0:
                text = "[gMASK]<sop>"
            else:
                text = ""
            if mfrom == "user":
                text += f"<|user|>\n{val}<|assistant|>"
            else:
                text += f"\n{val}"
        else:
            mfrom = message["role"]
            if mfrom == "user":
                mfrom = "user"
            elif mfrom == "assistant":
                mfrom = "assistant"
            if idx == 0:
                text = f"[gMASK]<sop><|{mfrom}|>\n{val}"
            else:
                text = f"<|{mfrom}|>\n{val}"
        input_ids = tokenizer.encode(text, add_special_tokens=False)
        if message["loss"]:
            action_mask = [1] * len(input_ids)
        else:
            action_mask = [0] * len(input_ids)
        return TokenizedConversationOutput(
            {
                "text": text,
                "input_ids": input_ids,
                "action_mask": action_mask,
            }
        )
