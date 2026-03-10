"""
NBSI v1.0 — Synthesiser (Phase 3)

Reads reasoning paths returned by beam search and produces
natural language narration using a local quantised LLM.

No API calls. No GPU required. Runs entirely on CPU.
Model: Qwen2.5-1.5B-Instruct Q4 (~900MB, ~10-15 tok/s on modern CPU)

The LLM does not reason here — the graph already did that.
The LLM narrates: it reads structured paths and describes
what they mean in plain language.
"""
from __future__ import annotations

import os
from typing import Optional


class Synthesiser:
    """
    Local LLM narration of NBSI reasoning paths.

    Usage:
        synth = Synthesiser("~/nbsi-models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
        answer = synth.narrate(query, paths, graph)
    """

    def __init__(
        self,
        model_path: str,
        n_threads: int = 4,
        n_ctx: int = 2048,
        verbose: bool = False,
    ):
        model_path = os.path.expanduser(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Download with:\n"
                f"  mkdir -p ~/nbsi-models\n"
                f"  wget -P ~/nbsi-models https://huggingface.co/Qwen/"
                f"Qwen2.5-1.5B-Instruct-GGUF/resolve/main/"
                f"qwen2.5-1.5b-instruct-q4_k_m.gguf"
            )
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed.\n"
                "Install with: uv pip install llama-cpp-python"
            )

        print(f"[Synthesiser] Loading model from {model_path}...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=0,      # CPU only — explicit
            verbose=verbose,
        )
        print("[Synthesiser] Model ready.")

    def narrate(
        self,
        query: str,
        paths: list[dict],
        max_tokens: int = 400,
        temperature: float = 0.2,
    ) -> str:
        """
        Narrate what the reasoning paths mean in relation to the query.

        paths: list of path dicts from session.query() — each has
               'path' (list of labels), 'conductivity', 'length'

        Returns natural language answer grounded in the paths.
        Does not hallucinate beyond what the paths contain.
        """
        if not paths:
            return "No relevant paths found in the current session graph."

        path_text = self._format_paths(paths)
        prompt    = self._build_prompt(query, path_text)

        result = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|im_end|>", "\n\n\n"],
            echo=False,
        )
        return result["choices"][0]["text"].strip()

    def narrate_streaming(
        self,
        query: str,
        paths: list[dict],
        max_tokens: int = 400,
        temperature: float = 0.2,
    ):
        """
        Streaming version — yields tokens as they are generated.
        Use when you want output to appear progressively.

        Usage:
            for token in synth.narrate_streaming(query, paths):
                print(token, end="", flush=True)
            print()
        """
        if not paths:
            yield "No relevant paths found in the current session graph."
            return

        path_text = self._format_paths(paths)
        prompt    = self._build_prompt(query, path_text)

        stream = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|im_end|>", "\n\n\n"],
            echo=False,
            stream=True,
        )
        for chunk in stream:
            token = chunk["choices"][0]["text"]
            if token:
                yield token

    def _format_paths(self, paths: list[dict]) -> str:
        lines = []
        for i, p in enumerate(paths[:5], 1):
            chain = " → ".join(p["path"])
            conf  = p.get("conductivity", 0.0)
            hops  = p.get("length", len(p["path"])) - 1
            lines.append(
                f"Path {i} (confidence {conf:.2f}, {hops} hop{'s' if hops != 1 else ''}): {chain}"
            )
        return "\n".join(lines)

    def _build_prompt(self, query: str, path_text: str) -> str:
        return (
            "<|im_start|>system\n"
            "You are reading reasoning paths from a knowledge graph built from a document.\n"
            "Each path shows how concepts in the document connect to each other.\n"
            "Higher confidence means the connection is more strongly grounded in the source.\n"
            "Your job is to narrate what these paths mean — not to add outside knowledge.\n"
            "Be concise. Be grounded. Flag uncertainty if the paths are weak or ambiguous.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"Question: {query}\n\n"
            f"Reasoning paths found in the document:\n{path_text}\n\n"
            "Based only on these paths, explain what they show about the question.\n"
            "State what the strongest path reveals. Note the confidence level.\n"
            "If the paths do not clearly answer the question, say so.\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
