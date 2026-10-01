"""
Simple, transparent side-effect detection with a keyword lexicon.

Why a lexicon and not a model? It is fully explainable, needs no labels,
and is good enough to answer "what do unhappy patients complain about?".
Each side effect maps to the phrases that count as a mention.
"""
import re

SIDE_EFFECTS = {
    "weight gain": ["weight gain", "gained weight", "gain weight", "gained \\d+ ?(?:lbs|pounds|kg)"],
    "nausea": ["nausea", "nauseous", "nauseated"],
    "headache": ["headache", "headaches", "migraine"],
    "anxiety": ["anxiety", "anxious", "panic attack"],
    "insomnia": ["insomnia", "can't sleep", "cannot sleep", "couldn't sleep", "trouble sleeping"],
    "fatigue": ["fatigue", "tired", "exhausted", "no energy"],
    "drowsiness": ["drowsy", "drowsiness", "sleepy"],
    "dizziness": ["dizzy", "dizziness", "lightheaded", "light headed"],
    "depression": ["depression", "depressed"],
    "mood swings": ["mood swings", "moody", "irritable"],
    "acne": ["acne", "breakout", "breakouts", "pimples"],
    "hair loss": ["hair loss", "hair falling", "losing hair", "hair thinning"],
    "bleeding": ["bleeding", "spotting"],
    "cramps": ["cramps", "cramping"],
    "diarrhea": ["diarrhea", "diarrhoea"],
    "constipation": ["constipation", "constipated"],
    "vomiting": ["vomiting", "vomit", "threw up", "throwing up"],
    "rash / itching": ["rash", "itching", "itchy", "hives"],
    "dry mouth": ["dry mouth"],
    "low libido": ["libido", "sex drive"],
    "heart palpitations": ["palpitations", "racing heart", "heart racing"],
    "joint / muscle pain": ["joint pain", "muscle pain", "muscle aches", "body aches"],
}

# If the patient is being TREATED for this, mentioning it is not a side effect
# (someone taking an antidepressant writes about depression). Effect -> condition keywords.
# Mental-health conditions overlap heavily (a patient treated for depression often writes
# about anxiety too), so mood-related effects are skipped for any of them.
MENTAL_HEALTH = ["depress", "anxiety", "panic", "bipolar", "obsessive", "ptsd",
                 "post traumatic", "schizo", "mood"]

CONDITION_OVERLAP = {
    "depression": MENTAL_HEALTH,
    "anxiety": MENTAL_HEALTH,
    "mood swings": MENTAL_HEALTH,
    "insomnia": ["insomnia"],
    "acne": ["acne"],
    "headache": ["headache", "migraine"],
    "nausea": ["nausea", "vomiting"],
    "vomiting": ["nausea", "vomiting"],
    "weight gain": ["weight", "obesity"],
    "diarrhea": ["diarrhea", "bowel"],
    "constipation": ["constipation", "bowel"],
    "hair loss": ["alopecia", "hair"],
    "rash / itching": ["rash", "urticaria", "eczema", "psoriasis", "dermatitis"],
    "joint / muscle pain": ["pain", "arthritis", "fibromyalgia"],
    "fatigue": ["fatigue", "narcolepsy"],
    "drowsiness": ["narcolepsy"],
    "low libido": ["libido", "erectile", "sexual"],
    "bleeding": ["bleeding", "menorrhagia"],
    "cramps": ["cramp", "dysmenorrhea"],
}

# One compiled regex per side effect; \b = whole words only ("tired" not "retired")
_PATTERNS = {
    effect: re.compile(r"\b(?:" + "|".join(phrases) + r")\b", flags=re.IGNORECASE)
    for effect, phrases in SIDE_EFFECTS.items()
}


def detect_side_effects(text: str, condition: str = "") -> list[str]:
    """
    Return the side effects mentioned in a review, e.g. ['nausea', 'headache'].
    Pass the condition being treated to skip effects that are really the condition.
    """
    if not isinstance(text, str):
        return []
    condition = (condition or "").lower()
    found = []
    for effect, pattern in _PATTERNS.items():
        if any(word in condition for word in CONDITION_OVERLAP.get(effect, [])):
            continue
        if pattern.search(text):
            found.append(effect)
    return found
