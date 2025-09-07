import re
from typing import List

VOCAB = {
    "Hello", "My", "Name", "Is", "I", "You", "Work", "Job", "Experience", "Skill",
    "Yes", "No", "ThankYou", "Please", "Sorry", "Good", "Bad", "Like", "Want", "Can",
    "Why", "What", "Where", "How", "Strong", "Weak", "Learn", "Team", "Company", "Love"
}

class GrammarAI:
    def __init__(self):
        # Normalize single tokens to readable text
        self.word_map = {
            "ThankYou": "Thank you",
            "Please": "Please",
            "Sorry": "Sorry"
        }

        # Exact phrase quick responses (lowercased)
        self.phrase_map = {
            "hello": "Hello!",
            "thank you": "Thank you.",
            "please": "Please.",
            "sorry": "Sorry.",
            "yes": "Yes.",
            "no": "No."
        }

        # Regex templates for common interview structures (built only from the 30 signs)
        self.patterns = [
            # Introductions
            (re.compile(r"\bHello My Name (?:Is )?([A-Za-z]+)\b", re.I), r"Hello, my name is \1"),
            (re.compile(r"\bMy Name (?:Is )?([A-Za-z]+)\b", re.I),       r"My name is \1"),

            # Motivation / ability
            (re.compile(r"\bI Want Job\b", re.I),                         "I want this job"),
            (re.compile(r"\bI Can Work Team\b", re.I),                    "I can work in a team"),
            (re.compile(r"\bI Can\b", re.I),                              "I can"),
            (re.compile(r"\bI Want\b", re.I),                             "I want"),

            # Experience / skills
            (re.compile(r"\bI Work Experience\b", re.I),                  "I have work experience"),
            (re.compile(r"\bI No Experience\b", re.I),                    "I do not have experience"),
            (re.compile(r"\bWhat Experience\b", re.I),                    "What experience do you have?"),
            (re.compile(r"\bWhat Skill\b", re.I),                         "What are your skills?"),
            (re.compile(r"\bI Skill ([A-Za-z]+)\b", re.I),                r"I have skills in \1"),
            (re.compile(r"\bI Like ([A-Za-z]+)\b", re.I),                 r"I like \1"),
            (re.compile(r"\bI Love ([A-Za-z]+)\b", re.I),                 r"I love \1"),

            # Team / company
            (re.compile(r"\bWhy Company\b", re.I),                        "Why this company?"),
            (re.compile(r"\bI Like Company\b", re.I),                     "I like this company"),
            (re.compile(r"\bI Love Company\b", re.I),                     "I love this company"),
            (re.compile(r"\bI Work Team\b", re.I),                        "I work well in a team"),

            # Strength / weakness
            (re.compile(r"\bMy Strong\b", re.I),                          "My strength is"),
            (re.compile(r"\bMy Weak\b", re.I),                            "My weakness is"),

            # Simple states/questions
            (re.compile(r"\bI Good\b", re.I),                             "I am good"),
            (re.compile(r"\bI Bad\b", re.I),                              "I am not well"),
            (re.compile(r"\bWhere You Work\b", re.I),                     "Where do you work?"),
            (re.compile(r"\bHow You Work Team\b", re.I),                  "How do you work in a team?"),
            (re.compile(r"\bWhy You Want Job\b", re.I),                   "Why do you want this job?"),
        ]

    # ---------- helpers ----------
    def _normalize_tokens(self, tokens: List[str]) -> List[str]:
        """Map special tokens and keep only allowed vocabulary (pass-through otherwise)."""
        norm = []
        for t in tokens:
            if t in self.word_map:
                norm.append(self.word_map[t])
            else:
                norm.append(t)
        return norm

    def _collapse_dupes(self, tokens: List[str]) -> List[str]:
        """Remove immediate duplicates (Hello Hello → Hello)."""
        if not tokens:
            return tokens
        out = [tokens[0]]
        for t in tokens[1:]:
            if t != out[-1]:
                out.append(t)
        return out

    def _polish(self, text: str) -> str:
        """Trim, capitalize, and ensure ending punctuation."""
        text = text.strip()
        if not text:
            return text
        # Capitalize first character
        text = text[0].upper() + text[1:]
        if text[-1] not in ".?!":
            text += "."
        return text

    # ---------- public API ----------
    def fix(self, tokens: List[str]) -> str:
        """
        Main pipeline:
        tokens → normalize → collapse dupes → quick phrase map → regex templates → polish
        """
        if not tokens:
            return ""

        tokens = self._collapse_dupes(self._normalize_tokens(tokens))
        sentence = " ".join(tokens).strip()

        # exact phrase quick wins
        low = sentence.lower()
        if low in self.phrase_map:
            return self.phrase_map[low]

        # apply regex templates
        fixed = sentence
        for pat, repl in self.patterns:
            fixed = pat.sub(repl, fixed)

        return self._polish(fixed)


# ---------- quick demo ----------
if __name__ == "__main__":
    ai = GrammarAI()
    tests = [
        ["Hello", "My", "Name", "Is", "Saad"],
        ["I", "Work", "Experience"],
        ["I", "No", "Experience"],
        ["I", "Can", "Work", "Team"],
        ["My", "Strong"],
        ["My", "Weak"],
        ["I", "Skill", "Python"],
        ["ThankYou"],
        ["I", "Like", "Company"],
        ["Why", "Company"],
        ["Where", "You", "Work"],
        ["How", "You", "Work", "Team"],
        ["Why", "You", "Want", "Job"],
        ["I", "Good"],
        ["I", "Bad"],
        ["I", "Love", "Machine", "Learning"],
    ]
    for raw in tests:
        print("Raw:", raw, "→", ai.fix(raw))
