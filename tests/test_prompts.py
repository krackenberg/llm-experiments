from pathlib import Path


def test_prompt_loads():
    p = Path("src/prompts/prompt_v001.txt")
    text = p.read_text()
    assert "{input_data}" in text
