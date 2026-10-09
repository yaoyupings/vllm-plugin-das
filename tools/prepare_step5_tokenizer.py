# SPDX-License-Identifier: Apache-2.0
"""Prepare a writable tokenizer sidecar without changing checkpoint files."""

import argparse
import json
from pathlib import Path
import shutil

from transformers import AutoTokenizer


def prepare_tokenizer(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
        "config.json",
    ):
        if (source / name).exists():
            shutil.copy2(source / name, target / name)

    config_path = target / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    # LlamaTokenizerFast replaces this checkpoint's ByteLevel BPE backend.
    config["tokenizer_class"] = "PreTrainedTokenizerFast"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2))

    tokenizer = AutoTokenizer.from_pretrained(str(target), local_files_only=True)
    for text in ("你好，2+2等于多少？", "Hello, 2+2 equals 4."):
        assert (
            tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)) == text
        )
    print("Step5 ByteLevel tokenizer round-trip passed.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare_tokenizer(args.model, args.output)
