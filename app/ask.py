"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"

_client = None
_chat_session = None


def reset_chat() -> None:
    """会話履歴をクリアして、新しい会話を開始する。"""
    global _client, _chat_session
    _chat_session = None
    # 既存のクライアントはそのまま残し、次回のユーザー質問時に再利用する。
    # ただし、閉じられている場合は、次回の再生成時に _client を置き換える。


def _get_chat_session():
    global _client, _chat_session
    if _client is None:
        _client = genai.Client(api_key=get_api_key())
    if _chat_session is None:
        _chat_session = _client.chats.create(model=MODEL)
    return _chat_session


def ask(question: str) -> str:
    try:
        chat = _get_chat_session()
        response = chat.send_message(question)
    except RuntimeError as exc:
        if "closed" not in str(exc):
            raise
        global _client, _chat_session
        _client = genai.Client(api_key=get_api_key())
        _chat_session = _client.chats.create(model=MODEL)
        response = _chat_session.send_message(question)

    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        input_tokens = getattr(usage, "prompt_token_count", None)
        output_tokens = getattr(usage, "candidates_token_count", None)
        print(f"[token usage] input_tokens={input_tokens}, output_tokens={output_tokens}")

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=None)
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="対話型 CLI モードで、質問を順番に入力して続けて会話する",
    )
    args = parser.parse_args()

    print(f"[model] {MODEL}")

    if args.interactive:
        print("[対話モード] 質問を入力してください。終了は 'quit' または 'exit'。")
        while True:
            try:
                question = input("Q> ")
            except EOFError:
                print()
                break
            if question.strip().lower() in {"", "quit", "exit"}:
                break
            print(f"[質問] {question}\n")
            try:
                print(ask(question))
            except errors.APIError as e:
                print(api_error_message(e.code, e.message))
                raise SystemExit(1)
        return

    question = args.question or DEFAULT_QUESTION
    print(f"[質問] {question}\n")
    try:
        print(ask(question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
