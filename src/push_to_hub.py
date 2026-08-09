"""
push_to_hub.py
Uploads a local checkpoint directory (config.json, tokenizer files, model.safetensors, README.md)
to the Hugging Face Hub via huggingface_hub's HfApi/create_repo/upload_folder.

Usage:
    python3 src/push_to_hub.py --checkpoint-dir <local dir> --repo-id <user/repo-name> \
        [--private] [--commit-message "..."]
"""

from __future__ import annotations

import argparse
import os
import sys

DEFAULT_COMMIT_MESSAGE = "Upload checkpoint (draft, not final)"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a local checkpoint directory to the Hugging Face Hub.")
    parser.add_argument("--checkpoint-dir", required=True, help="Local directory containing the checkpoint files.")
    parser.add_argument("--repo-id", required=True, help="Target repo on the Hub, e.g. user/repo-name.")
    parser.add_argument("--private", action="store_true", help="Create the repo as private.")
    parser.add_argument("--commit-message", default=DEFAULT_COMMIT_MESSAGE, help="Commit message for the upload.")
    parser.add_argument("--token", default=None, help="HF token; overrides the HF_TOKEN env var if given.")
    return parser.parse_args(argv)


def upload_checkpoint(args: argparse.Namespace) -> None:
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print(
            "HF_TOKEN environment variable is not set; export it or pass --token",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.isdir(args.checkpoint_dir):
        print(f"Checkpoint directory does not exist or is not a directory: {args.checkpoint_dir}", file=sys.stderr)
        sys.exit(1)

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True, repo_type="model")
        api.upload_folder(
            folder_path=args.checkpoint_dir,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=args.commit_message,
        )
    except Exception as exc:
        print(f"[!] Upload failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"https://huggingface.co/{args.repo_id}")


def main(argv: list[str] | None = None) -> None:
    upload_checkpoint(parse_args(argv))


if __name__ == "__main__":
    main()
