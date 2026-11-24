import re


WORD = r"\w+(?:['’]\w+)*"

REPEATED = re.compile(
    rf"(?=(?P<run>\b(?P<seq>{WORD}(?:\s+{WORD}){{0,59}}?)"
    rf"(?:\s+(?P=seq)){{2,}}\b))"
)


def tokenize(text):
    return re.findall(WORD, text.casefold())


class RepetitionDetector:
    def __init__(self, history_size=60):
        self.history = []
        self.history_size = history_size

    def reset(self):
        self.history.clear()

    def update(self, output):
        new_words = tokenize(output)
        if not new_words:
            return None

        old_history = self.history
        words = old_history + new_words

        prefix = " ".join(old_history)
        text = " ".join(words)
        output_start = len(prefix) + (1 if old_history else 0)

        result = None

        for match in REPEATED.finditer(text):
            start, end = match.span("run")

            # Ignore repetitions entirely contained in old history.
            if end <= output_start:
                continue

            phrase = match.group("seq")
            repetitions = (
                len(tokenize(match.group("run"))) // len(tokenize(phrase))
            )

            result = {
                "phrase": phrase,
                "repetitions": repetitions,
            }
            break

        # Retain only the last 60 words, including newly received output.
        self.history = words[-self.history_size:]

        return result
