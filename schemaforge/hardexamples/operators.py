"""The ten corruption operators for hard-example generation.

Each operator is a pure function with the uniform signature::

    Operator = Callable[[str, dict, random.Random, float], tuple[str, dict, list[str]]]
    # (text, gold, rng, severity) -> (new_text, new_gold, tags)

with ``severity`` in ``[0.0, 1.0]`` and ``tags`` naming the operator(s)
applied (e.g. ``["ocr_noise"]``).

Label provenance rule (SCHEMAFORGE_V2_RESEARCH_DIRECTION.md §3): labels are
NEVER inferred back from corrupted text.  Every operator starts from a clean
document whose gold JSON is known by construction and never re-parses the text
to build a label.  With the single exception of ``nest``, no operator changes
the gold; ``nest`` only changes it into a shape that still validates against
the schema model, and returns the input unchanged when the nesting cannot be
expressed.

Every operator satisfies the invariants asserted in ``tests/test_operators.py``:

- ``severity=0.0`` is the identity transform (text unchanged, empty tags).
- The caller's ``gold`` object is never mutated (deep-copied before any change).
- The operator never raises on any registered schema's clean seed.
"""

from __future__ import annotations

import copy
import datetime as _dt
import random
import re
from typing import Callable

Operator = Callable[[str, dict, random.Random, float], tuple[str, dict, list[str]]]

# ---------------------------------------------------------------------------
# ocr_noise
# ---------------------------------------------------------------------------

_OCR_MAP = {
    "0": "O",
    "O": "0",
    "1": "l",
    "l": "1",
    "5": "S",
    "S": "5",
}


def ocr_noise(text: str, gold: dict, rng: random.Random, severity: float) -> tuple[str, dict, list[str]]:
    """Corrupt the text with optical-character-recognition artifacts.

    Per-character corruption rate is ``0.06 * severity``.  Kinds applied with
    equal probability: ``rn`` -> ``m``, a dropped character, a duplicated
    character, and the visual-confusion substitutions ``0<->O``, ``1<->l``,
    ``5<->S``.  A space additionally becomes a line break with probability
    ``0.08 * severity`` (broken line join).  The gold is unchanged.  When
    ``severity > 0`` at least one corruption is guaranteed.
    """
    if severity <= 0.0 or not text or not isinstance(gold, dict):
        return text, gold, []
    rate = 0.06 * severity
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if rng.random() < rate:
            kind = rng.randrange(4)
            if kind == 0 and ch == "r" and i + 1 < n and text[i + 1] == "n":
                out.append("m")
                i += 2
            elif kind == 1:
                i += 1  # dropped character
            elif kind == 2:
                out.append(ch)
                out.append(ch)  # duplicated character
                i += 1
            else:
                out.append(_OCR_MAP.get(ch, ch))
                i += 1
        else:
            if ch == " " and rng.random() < 0.08 * severity:
                out.append("\n")
            else:
                out.append(ch)
            i += 1
    new_text = "".join(out)
    if new_text == text and text:
        new_text = text[0] + text if text else text  # guarantee a visible corruption
    return new_text, gold, (["ocr_noise"] if new_text != text else [])


# ---------------------------------------------------------------------------
# delabel
# ---------------------------------------------------------------------------

_LABELS = (
    "Amount requested:",
    "Current employer:",
    "Highest degree:",
    "Governing law:",
    "Liability cap:",
    "Request type:",
    "Reply needed:",
    "Other party:",
    "Last contact:",
    "Medications:",
    "Deal stage:",
    "Attachment:",
    "Triple extracted:",
    "Subtotal:",
    "Priority:",
    "Sentiment:",
    "Attending:",
    "Category:",
    "Customer:",
    "Cashier:",
    "Location:",
    "Confidence:",
    "Paid by:",
    "Claimant:",
    "Support:",
    "Urgency:",
    "Details:",
    "Subject:",
    "Vendor:",
    "Store:",
    "Email:",
    "Phone:",
    "Status:",
    "Total:",
    "Policy:",
    "Items:",
    "From:",
    "Name:",
    "Sent:",
    "Tax:",
    "To:",
)

_BARE_LABELS = ("Invoice", "Receipt", "Ticket", "Claim", "Form")


def delabel(text: str, gold: dict, rng: random.Random, severity: float) -> tuple[str, dict, list[str]]:
    """Remove field labels (``"Total:"``, ``"Vendor:"``, ...) from the text.

    Colon labels are processed longest-first.  Bare document-kind words
    (``"Invoice"``, ``"Receipt"``, ...) are matched at word boundaries so they
    never corrupt longer labels such as ``"Claimant:"``.  Each occurrence of a
    label is removed with probability ``severity``; the value that followed it
    remains and must be recovered from context alone.  The gold is unchanged.
    """
    if severity <= 0.0 or not text or not isinstance(gold, dict):
        return text, gold, []
    changed = False
    for label in sorted(_LABELS, key=len, reverse=True):
        while label in text and rng.random() < severity:
            text = text.replace(label, "", 1)
            changed = True
    for label in _BARE_LABELS:
        pattern = re.compile(rf"\b{re.escape(label)}\b")
        while pattern.search(text) and rng.random() < severity:
            text = pattern.sub("", text, count=1)
            changed = True
    if changed:
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = text.replace(" .", ".").replace(" ,", ",").strip()
    return text, gold, (["delabel"] if changed else [])


# ---------------------------------------------------------------------------
# reorder
# ---------------------------------------------------------------------------

_FILLERS = (
    "Please refer to the attached documentation.",
    "For further details, see the original document.",
    "This information is repeated in the appendix below.",
)


def _split_sentences(text: str) -> list[str]:
    """Split ``text`` into non-empty chunks on sentence ends and line breaks."""
    parts = re.split(r"(?<=[.!?])\s+|\n", text)
    chunks = [part.strip() for part in parts if part.strip()]
    return chunks or [text.strip()]


def _split_longest(sentences: list[str]) -> list[str]:
    """Split a single-chunk list into two chunks at a space near the middle."""
    if len(sentences) >= 2:
        return sentences
    sentence = sentences[0]
    mid = len(sentence) // 2
    left = sentence.rfind(" ", 0, mid)
    right = sentence.find(" ", mid)
    split_at = left if left != -1 else right
    if split_at == -1:
        return [sentence]
    return [sentence[:split_at], sentence[split_at + 1 :]]


def _split_one_across(sentences: list[str], rng: random.Random) -> list[str]:
    """Split one sentence at a comma and insert a filler sentence between the halves."""
    if len(sentences) < 2:
        return sentences
    idx = rng.randrange(len(sentences))
    pieces = [p.strip() for p in re.split(r",\s*", sentences[idx]) if p.strip()]
    if len(pieces) < 2:
        return sentences
    split_at = rng.randrange(1, len(pieces))
    left = ", ".join(pieces[:split_at])
    right = ", ".join(pieces[split_at:])
    filler = _FILLERS[rng.randrange(len(_FILLERS))]
    return sentences[:idx] + [left, filler, right] + sentences[idx + 1 :]


def reorder(text: str, gold: dict, rng: random.Random, severity: float) -> tuple[str, dict, list[str]]:
    """Shuffle sentence/field order; at ``severity >= 0.7`` split one field's
    information across a filler sentence.  The gold is unchanged and all
    original content stays present in the text.
    """
    if severity <= 0.0 or not text or not isinstance(gold, dict):
        return text, gold, []
    sentences = _split_sentences(text)
    if len(sentences) < 2:
        sentences = _split_longest(sentences)
    original = list(sentences)
    rng.shuffle(sentences)
    if sentences == original and len(sentences) >= 2:
        sentences[0], sentences[1] = sentences[1], sentences[0]
    if severity >= 0.7:
        sentences = _split_one_across(sentences, rng)
    return " ".join(sentences), gold, ["reorder"]


# ---------------------------------------------------------------------------
# abbreviate
# ---------------------------------------------------------------------------

_ABBREVIATIONS = (
    ("Chronic Obstructive Pulmonary Disease", "COPD"),
    ("Software License Agreement", "SLA"),
    ("Type 2 Diabetes Mellitus", "T2DM"),
    ("Master Services Agreement", "MSA"),
    ("Non-Disclosure Agreement", "NDA"),
    ("Congestive Heart Failure", "CHF"),
    ("Myocardial Infarction", "MI"),
    ("Chronic Kidney Disease", "CKD"),
    ("International", "Intl."),
    ("Hypertension", "HTN"),
    ("Corporation", "Corp."),
    ("Incorporated", "Inc."),
    ("Technologies", "Tech."),
)


def abbreviate(text: str, gold: dict, rng: random.Random, severity: float) -> tuple[str, dict, list[str]]:
    """Apply the domain abbreviation glossary (``Hypertension`` -> ``HTN``,
    ``Incorporated`` -> ``Inc.``, ...).

    Each occurrence of a glossary long form is replaced with probability
    ``severity``.  The gold is NEVER changed: when the abbreviated surface form
    is a key in the schema's ontology (e.g. ``HTN`` for ``medical_note``), the
    gold keeps the canonical value -- exactly the normalization capability the
    example is testing.
    """
    if severity <= 0.0 or not text or not isinstance(gold, dict):
        return text, gold, []
    changed = False
    for long_form, short_form in _ABBREVIATIONS:
        pos = 0
        while True:
            idx = text.find(long_form, pos)
            if idx == -1:
                break
            if rng.random() < severity:
                text = text[:idx] + short_form + text[idx + len(long_form) :]
                changed = True
                pos = idx + len(short_form)
            else:
                pos = idx + len(long_form)
    return text, gold, (["abbreviate"] if changed else [])


# ---------------------------------------------------------------------------
# synonym
# ---------------------------------------------------------------------------

_LABEL_SYNONYMS = {
    "Amount requested:": "Amount sought:",
    "Current employer:": "Employer:",
    "Highest degree:": "Education level:",
    "Governing law:": "Jurisdiction:",
    "Liability cap:": "Cap on liability:",
    "Request type:": "Request category:",
    "Reply needed:": "Reply requested:",
    "Other party:": "Second party:",
    "Last contact:": "Last touched:",
    "Medications:": "Prescribed drugs:",
    "Deal stage:": "Stage:",
    "Attachment:": "Attachments:",
    "Triple extracted:": "Relation extracted:",
    "Subtotal:": "Items subtotal:",
    "Priority:": "Priority level:",
    "Sentiment:": "Customer sentiment:",
    "Attending:": "Treating physician:",
    "Category:": "Issue category:",
    "Customer:": "Client:",
    "Cashier:": "Clerk:",
    "Location:": "Residence:",
    "Confidence:": "Confidence score:",
    "Paid by:": "Payment method:",
    "Claimant:": "Insured party:",
    "Support:": "Agent:",
    "Urgency:": "Urgency level:",
    "Details:": "Description:",
    "Subject:": "Subject line:",
    "Vendor:": "Supplier:",
    "Store:": "Merchant:",
    "Email:": "Email address:",
    "Phone:": "Contact number:",
    "Status:": "Current status:",
    "Total:": "Grand total:",
    "Policy:": "Policy number:",
    "Items:": "Line items:",
    "From:": "Sender:",
    "Name:": "Full name:",
    "Sent:": "Sent on:",
    "Tax:": "Sales tax:",
    "To:": "Recipient:",
}

_VALUE_SYNONYMS = (
    ("seen", "examined"),
    ("partners with", "collaborates with"),
)


def synonym(text: str, gold: dict, rng: random.Random, severity: float) -> tuple[str, dict, list[str]]:
    """Paraphrase field labels and non-identifying value wording.

    Each label is replaced with its synonym with probability ``severity``, then
    a small set of non-identifying value phrases.  The gold is unchanged.
    """
    if severity <= 0.0 or not text or not isinstance(gold, dict):
        return text, gold, []
    changed = False
    for label, replacement in _LABEL_SYNONYMS.items():
        if label in text and rng.random() < severity:
            text = text.replace(label, replacement)
            changed = True
    for old, new in _VALUE_SYNONYMS:
        if old in text and rng.random() < severity:
            text = text.replace(old, new)
            changed = True
    return text, gold, (["synonym"] if changed else [])


# ---------------------------------------------------------------------------
# typo
# ---------------------------------------------------------------------------

_KEYBOARD = {
    "q": "was",
    "w": "qase",
    "e": "wsdr",
    "r": "ewdf",
    "t": "rfgy",
    "y": "tghu",
    "u": "yhji",
    "i": "ujko",
    "o": "iklp",
    "p": "ol",
    "a": "qswz",
    "s": "awedx",
    "d": "serfc",
    "f": "drtgv",
    "g": "ftyhb",
    "h": "gyujn",
    "j": "huikm",
    "k": "jiol",
    "l": "kop",
    "z": "asx",
    "x": "zsdc",
    "c": "xdfv",
    "v": "cfgb",
    "b": "vghn",
    "n": "bhjm",
    "m": "njk",
    "1": "2q",
    "2": "3wq",
    "3": "4we",
    "4": "5er",
    "5": "6rt",
    "6": "7ty",
    "7": "8yu",
    "8": "9ui",
    "9": "0io",
    "0": "9po",
}

_PHONETIC = (("ph", "f"), ("ck", "k"))


def typo(text: str, gold: dict, rng: random.Random, severity: float) -> tuple[str, dict, list[str]]:
    """Introduce keyboard-adjacency and phonetic errors.

    Phonetic digraphs (``ph`` -> ``f``, ``ck`` -> ``k``) are replaced with
    probability ``severity`` per occurrence; then every letter/digit is
    replaced by a QWERTY neighbor with probability ``0.05 * severity``.  The
    gold is unchanged.  When ``severity > 0`` at least one error is guaranteed.
    """
    if severity <= 0.0 or not text or not isinstance(gold, dict):
        return text, gold, []
    changed = False
    for old, new in _PHONETIC:
        pos = 0
        while True:
            idx = text.find(old, pos)
            if idx == -1:
                break
            if rng.random() < severity:
                text = text[:idx] + new + text[idx + len(old) :]
                changed = True
                pos = idx + len(new)
            else:
                pos = idx + len(old)
    rate = 0.05 * severity
    out: list[str] = []
    for ch in text:
        if (ch.isalpha() or ch.isdigit()) and rng.random() < rate:
            neighbors = _KEYBOARD.get(ch.lower(), "")
            if neighbors:
                sub = neighbors[rng.randrange(len(neighbors))]
                out.append(sub if ch.islower() else sub.upper())
                changed = True
            else:
                out.append(ch)
        else:
            out.append(ch)
    new_text = "".join(out)
    if new_text == text and text:
        new_text = text[0] + text if text else text  # guarantee a visible error
    return new_text, gold, (["typo"] if new_text != text else [])


# ---------------------------------------------------------------------------
# code_switch
# ---------------------------------------------------------------------------

_SPANISH_LABELS = {
    "Amount requested:": "Cantidad solicitada:",
    "Current employer:": "Empleador actual:",
    "Highest degree:": "Título académico:",
    "Governing law:": "Ley aplicable:",
    "Liability cap:": "Límite de responsabilidad:",
    "Request type:": "Tipo de solicitud:",
    "Reply needed:": "Respuesta necesaria:",
    "Other party:": "Otra parte:",
    "Last contact:": "Último contacto:",
    "Medications:": "Medicamentos:",
    "Deal stage:": "Etapa:",
    "Attachment:": "Adjunto:",
    "Triple extracted:": "Triple extraído:",
    "Subtotal:": "Subtotal:",
    "Priority:": "Prioridad:",
    "Sentiment:": "Sentimiento:",
    "Attending:": "Médico a cargo:",
    "Category:": "Categoría:",
    "Customer:": "Cliente:",
    "Cashier:": "Cajero:",
    "Location:": "Ubicación:",
    "Confidence:": "Confianza:",
    "Paid by:": "Pagado con:",
    "Claimant:": "Reclamante:",
    "Support:": "Soporte:",
    "Urgency:": "Urgencia:",
    "Details:": "Detalles:",
    "Subject:": "Asunto:",
    "Vendor:": "Proveedor:",
    "Store:": "Tienda:",
    "Email:": "Correo:",
    "Phone:": "Teléfono:",
    "Status:": "Estado:",
    "Total:": "Total:",
    "Policy:": "Póliza:",
    "Items:": "Artículos:",
    "From:": "De:",
    "Name:": "Nombre:",
    "Sent:": "Enviado:",
    "Tax:": "Impuesto:",
    "To:": "Para:",
}

_GERMAN_LABELS = {
    "Amount requested:": "Geforderter Betrag:",
    "Current employer:": "Arbeitgeber:",
    "Highest degree:": "Abschluss:",
    "Governing law:": "Rechtsgebiet:",
    "Liability cap:": "Haftungsgrenze:",
    "Request type:": "Anfragetyp:",
    "Reply needed:": "Antwort erforderlich:",
    "Other party:": "Andere Partei:",
    "Last contact:": "Letzter Kontakt:",
    "Medications:": "Medikamente:",
    "Deal stage:": "Phase:",
    "Attachment:": "Anhang:",
    "Triple extracted:": "Triple extrahiert:",
    "Subtotal:": "Zwischensumme:",
    "Priority:": "Priorität:",
    "Sentiment:": "Stimmung:",
    "Attending:": "Behandelnder Arzt:",
    "Category:": "Kategorie:",
    "Customer:": "Kunde:",
    "Cashier:": "Kassierer:",
    "Location:": "Ort:",
    "Confidence:": "Konfidenz:",
    "Paid by:": "Bezahlt mit:",
    "Claimant:": "Antragsteller:",
    "Support:": "Support:",
    "Urgency:": "Dringlichkeit:",
    "Details:": "Einzelheiten:",
    "Subject:": "Betreff:",
    "Vendor:": "Lieferant:",
    "Store:": "Geschäft:",
    "Email:": "E-Mail:",
    "Phone:": "Telefon:",
    "Status:": "Status:",
    "Total:": "Gesamt:",
    "Policy:": "Police:",
    "Items:": "Artikel:",
    "From:": "Von:",
    "Name:": "Name:",
    "Sent:": "Gesendet:",
    "Tax:": "Steuer:",
    "To:": "An:",
}

_VALUE_SWITCH = {
    "Spanish": {"Follow-up": "Seguimiento", "seen": "visto"},
    "German": {"Follow-up": "Nachsorge", "seen": "gesehen"},
}


def code_switch(text: str, gold: dict, rng: random.Random, severity: float) -> tuple[str, dict, list[str]]:
    """Translate labels (and at ``severity >= 0.7`` some value words) into
    Spanish or German, chosen per call.

    Each label is translated with probability ``severity``; value words are
    translated only when ``severity >= 0.7``.  The gold is unchanged and stays
    in the original language.
    """
    if severity <= 0.0 or not text or not isinstance(gold, dict):
        return text, gold, []
    language = "Spanish" if rng.random() < 0.5 else "German"
    glossary = _GERMAN_LABELS if language == "German" else _SPANISH_LABELS
    changed = False
    for label, replacement in glossary.items():
        if label in text and rng.random() < severity:
            text = text.replace(label, replacement)
            changed = True
    if severity >= 0.7:
        for old, new in _VALUE_SWITCH[language].items():
            if old in text and rng.random() < severity:
                text = text.replace(old, new)
                changed = True
    return text, gold, (["code_switch"] if changed else [])


# ---------------------------------------------------------------------------
# nest
# ---------------------------------------------------------------------------

_NEST_RE_CLAIMANT = re.compile(r"Claimant:\s*([^.]*?)\s*\.\s*")
_NEST_RE_OTHER = re.compile(r"\s*Other party:\s*[^.]*?\.\s*")


def nest(text: str, gold: dict, rng: random.Random, severity: float) -> tuple[str, dict, list[str]]:
    """Promote flat fields into nested entities / repeated line items.

    With the current schema set only insurance-claim-shaped golds (a flat
    ``claimant_name`` plus an ``involved_parties`` list) can express this: the
    claimant is promoted into ``involved_parties`` as a ``Claimant`` entry and
    the text is re-rendered as a parties listing.  The returned gold still
    validates against the schema model.  For any other shape the input is
    returned unchanged with empty tags -- a gold the schema cannot express is
    never produced.
    """
    if (
        severity <= 0.0
        or not text
        or not isinstance(gold, dict)
        or "claimant_name" not in gold
        or "involved_parties" not in gold
    ):
        return text, gold, []
    new_text = _NEST_RE_CLAIMANT.sub(r"Parties involved: (1) \1 (Claimant). ", text)
    new_text = _NEST_RE_OTHER.sub("", new_text)
    new_text = re.sub(r"[ \t]{2,}", " ", new_text).replace(" .", ".").strip()
    if new_text == text:
        return text, gold, []
    new_gold = copy.deepcopy(gold)
    new_gold["involved_parties"] = [
        {"name": gold["claimant_name"], "role": "Claimant"}
    ] + list(gold["involved_parties"])
    return new_text, new_gold, ["nest"]


# ---------------------------------------------------------------------------
# implicit
# ---------------------------------------------------------------------------

_AGE_RE = re.compile(r"\bage (\d{1,2})\b", re.I)
_AGE_RE2 = re.compile(r"\b(\d{1,2}) years? old\b", re.I)
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_TOTAL_PHRASES = {
    "total_amount": "the total amount due",
    "amount_requested": "the amount requested",
    "deal_value": "the deal value",
    "liability_cap": "the liability cap",
}

_UNITS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def _cardinal(n: int) -> str:
    """Return ``n`` (0-99) in English words."""
    if 0 <= n < 20:
        return _UNITS[n]
    tens, units = divmod(n, 10)
    word = _TENS[tens]
    if units:
        word += "-" + _UNITS[units]
    return word


def _ordinal(n: int) -> str:
    """Return ``n`` (0-99) as an English ordinal (e.g. 42 -> ``forty-second``)."""
    if 10 < n < 20:
        return {
            11: "eleventh",
            12: "twelfth",
            13: "thirteenth",
            14: "fourteenth",
            15: "fifteenth",
            16: "sixteenth",
            17: "seventeenth",
            18: "eighteenth",
            19: "nineteenth",
        }[n]
    tens, units = divmod(n, 10)
    if units == 0:
        if n >= 20:
            base = _TENS[tens]
            return base[:-1] + "ieth" if base.endswith("ty") else base + "th"
        return ("zeroth", "tenth", "twentieth", "thirtieth", "fortieth", "fiftieth", "sixtieth", "seventieth", "eightieth", "ninetieth")[tens]
    return _cardinal(n - units) + "-" + {1: "first", 2: "second", 3: "third"}.get(units, _UNITS[units] + "th")


def _collect_gold_strings(gold: dict) -> list[str]:
    """All non-empty string values in ``gold``, in document order."""
    out: list[str] = []
    if isinstance(gold, dict):
        for value in gold.values():
            out.extend(_collect_gold_strings(value))
    elif isinstance(gold, list):
        for item in gold:
            out.extend(_collect_gold_strings(item))
    elif isinstance(gold, str) and gold.strip():
        out.append(gold)
    return out


def _implicit_age(text: str) -> tuple[str, bool]:
    """Replace an ``age <n>``/``<n> years old`` literal with a birthday phrase."""
    match = _AGE_RE.search(text) or _AGE_RE2.search(text)
    if match is None:
        return text, False
    age = int(match.group(1))
    phrase = f"who celebrated his or her {_ordinal(age)} birthday"
    return text[: match.start()] + phrase + text[match.end() :], True


def _implicit_date(text: str, gold: dict) -> tuple[str, bool]:
    """Rewrite one date as a relative expression anchored to another in-text date."""
    dates = sorted({v for v in _collect_gold_strings(gold) if _ISO_DATE_RE.match(v) and v in text})
    if not dates:
        return text, False
    replaced = dates[0]
    idx = text.find(replaced)
    if idx == -1:
        return text, False
    if len(dates) >= 2:
        anchor = dates[1]
        try:
            delta = (_dt.date.fromisoformat(anchor) - _dt.date.fromisoformat(replaced)).days
        except ValueError:
            return text, False
        if delta >= 0:
            phrase = f"{delta} days before {anchor}"
        else:
            phrase = f"{-delta} days after {anchor}"
    else:
        phrase = "recently"
    return text[:idx] + phrase + text[idx + len(replaced) :], True


def _implicit_total(text: str, gold: dict) -> tuple[str, bool]:
    """Rewrite a total amount literal as a phrase that implies it."""
    for key, phrase in _TOTAL_PHRASES.items():
        value = gold.get(key)
        if isinstance(value, str) and value and value in text:
            return text.replace(value, phrase, 1), True
    return text, False


def implicit(text: str, gold: dict, rng: random.Random, severity: float) -> tuple[str, dict, list[str]]:
    """Replace literals with inferences the reader must make.

    At minimum, with probability ``severity`` each: a numeric age becomes
    ``"who celebrated his or her forty-second birthday"``, a date becomes a
    relative expression anchored to another date still in the document, and a
    total becomes a phrase implying it.  The gold is unchanged -- that is the
    point.
    """
    if severity <= 0.0 or not text or not isinstance(gold, dict):
        return text, gold, []
    changed = False

    if rng.random() < severity:
        text, did = _implicit_age(text)
        changed = changed or did
    if rng.random() < severity:
        text, did = _implicit_date(text, gold)
        changed = changed or did
    if rng.random() < severity:
        text, did = _implicit_total(text, gold)
        changed = changed or did

    return text, gold, (["implicit"] if changed else [])


# ---------------------------------------------------------------------------
# ambiguate
# ---------------------------------------------------------------------------

_MONEY_RE = re.compile(r"^\d[\d,]*\.\d{2}$")

_CONFUSABLE = {
    "Hypertension": "Hypotension",
    "Refund": "Rebate",
    "Cancellation": "Cancelation",
    "Battery": "Battery pack",
    "Display": "Display screen",
    "Customer": "Costumer",
    "Support": "Supporter",
    "Witness": "Watcher",
    "Driver": "Rider",
}


def _ambiguous_alternative(word: str) -> str | None:
    """Return a second reading for ``word``, or ``None`` when none exists.

    Money amounts get their cents digits reversed (a genuinely ambiguous
    total); words in the confusable map get their paired alternative.  Anything
    else admits no clean second reading.
    """
    if _MONEY_RE.fullmatch(word):
        intpart, cents = word.split(".")
        return f"{intpart}.{cents[::-1]}"
    return _CONFUSABLE.get(word)


def ambiguate(text: str, gold: dict, rng: random.Random, severity: float) -> tuple[str, dict, list[str]]:
    """Introduce a genuine second reading for one gold string in the text.

    The first gold string (in sorted order) that admits an alternative is
    rewritten as ``"<value> (or <alternative>)"``.  The gold is unchanged.
    The returned ``["ambiguate"]`` tag is what downstream scoring uses to route
    the item to the confidence evaluation rather than the accuracy numerator.
    """
    if severity <= 0.0 or not text or not isinstance(gold, dict):
        return text, gold, []
    candidates = sorted(v for v in _collect_gold_strings(gold) if v in text)
    for target in candidates:
        alternative = _ambiguous_alternative(target)
        if alternative and alternative != target:
            new_text = text.replace(target, f"{target} (or {alternative})", 1)
            return new_text, gold, ["ambiguate"]
    return text, gold, []


OPERATORS: dict[str, Operator] = {
    "ocr_noise": ocr_noise,
    "delabel": delabel,
    "reorder": reorder,
    "abbreviate": abbreviate,
    "synonym": synonym,
    "typo": typo,
    "code_switch": code_switch,
    "nest": nest,
    "implicit": implicit,
    "ambiguate": ambiguate,
}
