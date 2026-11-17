"""Conservative exact-number grading. Never evaluates model-generated expressions."""

import re
from fractions import Fraction


def braced(text, start):
    if start >= len(text) or text[start] != "{":
        raise ValueError("Expected opening brace")
    depth = 0
    for i in range(start, len(text)):
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            return text[start + 1 : i], i + 1
    raise ValueError("Unbalanced braces")


def unwrap(text):
    text = text.strip()
    while text.startswith("{"):
        inside, end = braced(text, 0)
        if end != len(text):
            break
        text = inside.strip()
    return text


def normalize_number(text):
    try:
        text = text.strip().replace("−", "-")
        for token in (r"\$", "$", r"\!", r"\,", r"\;", "\\ ", r"\left", r"\right"):
            text = text.replace(token, "")
        text = re.sub(r"\\(?:text|mathrm)\{[A-Za-z\s]+\}", "", text).strip()
        text = text.removesuffix(".").strip()
        text = re.sub(r"(?:\\)?%$", "", text).strip()
        text = re.sub(r"^[A-Za-z]\w*\s*=\s*", "", text, count=1)
        text = unwrap(text)
        match = re.match(r"^([+-]?)\\(?:dfrac|tfrac|frac)\s*", text)
        if match:
            numerator, end = braced(text, match.end())
            while end < len(text) and text[end].isspace():
                end += 1
            denominator, end = braced(text, end)
            if text[end:].strip():
                return None
            a, b = plain_number(unwrap(numerator)), plain_number(unwrap(denominator))
            return (-1 if match[1] == "-" else 1) * a / b
        if "/" in text:
            parts = text.split("/")
            if len(parts) != 2:
                return None
            return plain_number(unwrap(parts[0])) / plain_number(unwrap(parts[1]))
        return plain_number(text)
    except (ValueError, ZeroDivisionError, TypeError):
        return None


def plain_number(text):
    text = text.strip()
    if not re.fullmatch(r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)", text):
        raise ValueError("Unsupported numeric form")
    return Fraction(text.replace(",", ""))


def extract_gold(answer):
    if "####" not in answer:
        raise ValueError("Missing GSM8K gold marker")
    number = normalize_number(answer.rsplit("####", 1)[1])
    if number is None:
        raise ValueError("Invalid GSM8K reference number")
    return number


def extract_final_answer(text):
    expressions = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        try:
            content, _ = braced(text, match.end() - 1)
            expressions.append(content)
        except ValueError:
            continue
    if expressions:
        return expressions[-1]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and lines[-1].lower().startswith("final answer:"):
        return lines[-1].split(":", 1)[1].strip()
    return None


def continuation_tokens(token_ids, stop_ids, input_length=0):
    output = list(token_ids[input_length:])
    for i, token in enumerate(output):
        if token in stop_ids:
            return output[: i + 1]
    return output


def count_generated_tokens(token_ids, stop_ids, input_length=0):
    return len(continuation_tokens(token_ids, stop_ids, input_length))


def split_thinking(token_ids, opening_id, closing_id, prompt_in_thinking=False):
    closes = [i for i, token in enumerate(token_ids) if token == closing_id]
    if closes:
        cut = closes[-1]
        # A later unmatched opening delimiter means reasoning resumed unfinished.
        if opening_id in token_ids[cut + 1 :]:
            return list(token_ids), [], "no_final_segment"
        return list(token_ids[:cut]), list(token_ids[cut + 1 :]), "available"
    if opening_id in token_ids or prompt_in_thinking:
        return list(token_ids), [], "no_final_segment"
    return [], list(token_ids), "available"


def grade_response(token_ids, tokenizer, gold, stop_ids, max_new_tokens, prompt_in_thinking=False):
    ids = continuation_tokens(token_ids, stop_ids)
    opening = tokenizer.convert_tokens_to_ids("<think>")
    closing = tokenizer.convert_tokens_to_ids("</think>")
    thinking, final, status = split_thinking(ids, opening, closing, prompt_in_thinking)
    text = tokenizer.decode(final, skip_special_tokens=True)
    expression = extract_final_answer(text) if status == "available" else None
    number = normalize_number(expression) if expression is not None else None
    stopped = bool(ids and ids[-1] in stop_ids)
    return {
        "generated_token_ids": ids,
        "raw_output": tokenizer.decode(ids, skip_special_tokens=False),
        "thinking_segment": tokenizer.decode(thinking, skip_special_tokens=True),
        "final_answer_segment": text,
        "parsed_number": str(number) if number is not None else None,
        "correct": number is not None and number == Fraction(gold),
        "parse_status": status if status != "available" else ("parsed" if number is not None else "unparsed"),
        "token_count": count_generated_tokens(ids, stop_ids),
        "stop_reason": "eos" if stopped else ("length" if len(ids) >= max_new_tokens else "other"),
        "truncated": not stopped and len(ids) >= max_new_tokens,
    }
