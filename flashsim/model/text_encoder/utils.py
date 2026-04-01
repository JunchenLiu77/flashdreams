import ftfy
import html
import re


def basic_clean(text: str) -> str:
    """
    Clean text by fixing encoding issues and unescaping HTML entities.
    """
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text: str) -> str:
    """
    Clean text by replacing multiple spaces with a single space and removing leading/trailing whitespace.
    """
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text: str) -> str:
    """
    Clean text by applying basic and whitespace cleaning.
    """
    text = whitespace_clean(basic_clean(text))
    return text


def str2bool(v: str | bool) -> bool:
    """
    Convert a string or boolean to a boolean.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise ValueError("Boolean value expected.")
