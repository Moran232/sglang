
import argparse
import asyncio
import json
import os
import pickle
import random
import resource
import sys
import time
import traceback
import warnings
from argparse import ArgumentParser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np
import requests
from tqdm.asyncio import tqdm
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

def get_tokenizer(
    pretrained_model_name_or_path: str,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    if pretrained_model_name_or_path.endswith(
        ".json"
    ) or pretrained_model_name_or_path.endswith(".model"):
        from sglang.srt.hf_transformers_utils import get_tokenizer

        return get_tokenizer(pretrained_model_name_or_path)

    if pretrained_model_name_or_path is not None and not os.path.exists(
        pretrained_model_name_or_path
    ):
        pretrained_model_name_or_path = get_model(pretrained_model_name_or_path)
    return AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=True
    )

def sample_HLE_requeset(
    data_name: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
    context_len: Optional[int] = None,
    prompt_suffix: Optional[str] = "",
    apply_chat_template=False,
):
    from datasets import load_dataset
    dataset = load_dataset(data_name, split="test")
    filtered_dataset = []
    for i in tqdm(range(len(dataset)), desc=f"sampling data {data_name}"):
        data = dataset[i]
        if len(filtered_dataset) == num_requests:
            break
        prompt = ''
        answer = ''
        if data_name == 'cais/hle':
            prompt = data.get('question')
            answer =  data.get('answer')
        elif data_name == 'SciCode1/SciCode':
            prompt = data.get('problem_background_main') + "\n" + data.get('problem_description_main') + \
                "\n" + data.get('problem_io') 
            answer = data.get('general_solution')
        else:
            break
        # if answer is None:
        #     continue
        prompt_token_ids = tokenizer.encode(prompt)
        if answer:
            completion_token_ids = tokenizer.encode(answer)
        else:
            completion_token_ids = [0]
        prompt_len = len(prompt_token_ids)
        if prompt_len < 1000 or prompt_len > 1048:
            # print(prompt_len, len(completion_token_ids))
            continue

        output_len = (
            len(completion_token_ids) if fixed_output_len is None else fixed_output_len
        )

        print(prompt_len, output_len, context_len)
        filtered_dataset.append((prompt, prompt_len, output_len))

    print(f"#Input tokens: {np.sum([x[1] for x in filtered_dataset])}")
    print(f"#Output tokens: {np.sum([x[2] for x in filtered_dataset])}")
    return filtered_dataset

import os
import sys
tokenizer_path = sys.argv[1]

tokenizer = get_tokenizer(tokenizer_path)
all_dataset = {}
for data_name in (
    'cais/hle', 
    'SciCode1/SciCode',
    ):
    dataset = sample_HLE_requeset(data_name, 1000, tokenizer)
    all_dataset[data_name] = dataset

for k, v in all_dataset.items():
    print(k, len(v))