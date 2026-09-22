"""Publication policy for known raw-input text fields in structured artifacts.

This check covers a specific excluded field, not every possible identity or text
leak. Fixed templates and third-party scientific attribution in code are valid.
"""


def has_dataset_text(value):
    """Return whether an artifact retains a nonempty rendered prompt text."""
    if isinstance(value, dict):
        return any(
            (key == "rendered_prompt_text" and bool(item)) or has_dataset_text(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(has_dataset_text(item) for item in value)
    return False


def strip_rendered_prompt_text(value):
    """Remove that field in place; return the number of removed fields.

    Scores, labels, sample IDs, input hashes, and all other fields are preserved.
    Intended for publication copies, never for original run artifacts.
    """
    count = 0
    if isinstance(value, dict):
        if "rendered_prompt_text" in value:
            del value["rendered_prompt_text"]
            count += 1
        for item in value.values():
            count += strip_rendered_prompt_text(item)
    elif isinstance(value, list):
        for item in value:
            count += strip_rendered_prompt_text(item)
    return count
