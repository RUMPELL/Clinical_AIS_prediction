from __future__ import annotations

def build_chatml_prompt(report_text: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are an AIS-coding assistant. Given a trauma CT report, output ALL applicable AIS codes separated by commas.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{report_text}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
