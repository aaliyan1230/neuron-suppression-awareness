from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConceptSplit:
    train: tuple[str, ...]
    test: tuple[str, ...]
    categories: dict[str, tuple[str, ...]]


PROMPT_TEMPLATES: tuple[str, ...] = (
    "Tell me about {concept}.",
    "What is {concept}?",
    "Explain {concept} to me.",
    "Describe the concept of {concept}.",
    "What are the key characteristics of {concept}?",
)

BASELINE_CONCEPT = "something"

_CATEGORIES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "animals": (
        (
            "dog", "cat", "elephant", "whale", "eagle", "snake", "rabbit",
            "dolphin", "tiger", "bear", "horse", "wolf", "deer", "penguin",
            "shark", "owl", "fox", "parrot", "gorilla", "turtle", "bee",
            "octopus", "cheetah", "frog", "crocodile",
        ),
        (
            "flamingo", "hedgehog", "jellyfish", "panther", "seal",
            "koala", "bat", "otter",
        ),
    ),
    "cities_countries": (
        (
            "London", "Tokyo", "Paris", "Brazil", "Egypt", "Canada",
            "Mumbai", "Berlin", "Sydney", "Moscow", "Istanbul", "Bangkok",
            "Mexico", "Argentina", "Kenya", "Vietnam", "Norway",
            "Portugal", "Jamaica", "Switzerland",
        ),
        (
            "Dublin", "Seoul", "Havana", "Nairobi", "Helsinki", "Lima",
        ),
    ),
    "emotions": (
        (
            "joy", "anger", "fear", "sadness", "love", "disgust", "surprise",
            "pride", "guilt", "shame", "envy", "gratitude", "hope",
            "anxiety", "nostalgia", "jealousy", "compassion",
            "excitement", "boredom", "relief",
        ),
        (
            "awe", "contempt", "loneliness", "serenity", "frustration",
            "euphoria",
        ),
    ),
    "foods": (
        (
            "bread", "rice", "cheese", "apple", "chocolate", "coffee",
            "pasta", "sushi", "pizza", "curry", "honey", "yogurt",
            "cinnamon", "garlic", "lemon", "avocado", "butter",
            "mushroom", "olive", "vanilla",
        ),
        (
            "mango", "tofu", "ginger", "pomegranate", "truffle", "kimchi",
        ),
    ),
    "colors_materials": (
        (
            "red", "blue", "gold", "silver", "wooden", "glass", "steel",
            "plastic", "copper", "marble", "ivory", "bronze",
            "crystal", "leather", "silk",
        ),
        (
            "turquoise", "velvet", "ceramic", "titanium",
        ),
    ),
    "occupations": (
        (
            "doctor", "teacher", "pilot", "engineer", "farmer", "artist",
            "judge", "soldier", "architect", "journalist", "chef", "librarian",
            "detective", "surgeon", "philosopher",
        ),
        (
            "astronaut", "blacksmith", "diplomat", "botanist",
        ),
    ),
    "nature_science": (
        (
            "gravity", "ocean", "volcano", "lightning", "DNA", "oxygen",
            "earthquake", "photosynthesis", "magnetism", "glacier",
            "nebula", "ecosystem", "diamond", "coral", "fossil",
        ),
        (
            "aurora", "tundra", "monsoon", "quasar",
        ),
    ),
    "abstract": (
        (
            "justice", "freedom", "chaos", "democracy", "loyalty", "wisdom",
            "courage", "truth", "beauty", "entropy", "destiny", "harmony",
            "patience", "curiosity", "imagination", "empathy", "integrity",
            "forgiveness", "rebellion", "tradition",
        ),
        (
            "paradox", "solitude", "ambiguity", "resilience",
        ),
    ),
    # OOD categories (test-only)
    "musical_instruments": (
        (),
        ("violin", "trumpet", "harp", "accordion"),
    ),
    "sports": (
        (),
        ("tennis", "archery"),
    ),
    "diseases": (
        (),
        ("malaria", "diabetes"),
    ),
}


def load_concepts() -> ConceptSplit:
    train: list[str] = []
    test: list[str] = []
    categories: dict[str, tuple[str, ...]] = {}
    for category, (train_words, test_words) in _CATEGORIES.items():
        train.extend(train_words)
        test.extend(test_words)
        categories[category] = train_words + test_words
    return ConceptSplit(
        train=tuple(train),
        test=tuple(test),
        categories=categories,
    )
