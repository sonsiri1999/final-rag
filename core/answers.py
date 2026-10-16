"""🔒 Turn LLM replies into what the API contract requires."""
import json
import logging
import re

log = logging.getLogger("starter")
_CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def _match_key(text: str, keys: list[str]) -> str | None:
    text = text.strip()
    if text in keys:
        return text
    for key in keys:
        # "ค." or "ค) ..." counts, but "คำตอบ..." must not match the key "ค"
        if text.startswith(key) and (len(text) == len(key) or text[len(key)] in ".): "):
            return key
    return None


def parse_answer(raw: str, keys: list[str]) -> str:
    """Always returns one of the option keys (the first one if the reply cannot be read)."""
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            found = _match_key(str(data.get("answer", "")), keys)
            if found:
                return found
    except json.JSONDecodeError:
        pass
    match = re.search(r'"answer"\s*:\s*"([^"]+)"', raw)
    if match and (found := _match_key(match.group(1), keys)):
        return found
    log.warning("could not read an option key from the LLM reply: %r", raw[:200])
    return keys[0]


def extract_code(raw: str) -> str:
    """Take the ```python block if there is one, otherwise the whole reply."""
    match = _CODE_BLOCK.search(raw)
    return (match.group(1) if match else raw).strip() + "\n"
