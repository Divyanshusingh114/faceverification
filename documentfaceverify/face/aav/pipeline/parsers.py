"""
parsers.py
----------
Best-effort regex parsers for Indian government identity documents.

Honest about what these are:
  - PATTERNS, not understanding. A clean print of an Aadhaar PDF will parse
    well; a smudged scan of a Voter ID may extract nothing.
  - The same field can appear in many ways across regional variants and
    photo OCR errors. Each parser counts how many of the expected fields
    it found and surfaces a `parse_confidence` score so callers can route
    low-confidence parses to manual review.
  - Sensitive numbers (Aadhaar, PAN) are MASKED here at the parser level
    so they cannot accidentally leak into logs or session state.

All parsers return:
    {
      "fields": {...},
      "parse_confidence": 0.0–1.0,
    }
"""

from __future__ import annotations

import re
from typing import Callable

DOC_AADHAAR = "aadhaar"
DOC_PAN = "pan"
DOC_VOTER = "voter_id"
DOC_DL = "driving_licence"
DOC_PASSPORT = "passport"

SUPPORTED_DOC_TYPES = (DOC_AADHAAR, DOC_PAN, DOC_VOTER, DOC_DL, DOC_PASSPORT)


# --------------------------------------------------------------- masking
def _mask_aadhaar(digits: str) -> str:
    digits = re.sub(r"\D", "", digits)
    if len(digits) < 12:
        return "INVALID"
    return f"XXXX-XXXX-{digits[-4:]}"


def _mask_pan(pan: str) -> str:
    pan = pan.upper()
    if not re.fullmatch(r"[A-Z]{5}\d{4}[A-Z]", pan):
        return "INVALID"
    return f"{pan[:3]}XX{pan[5:9]}{pan[-1]}"


def _mask_passport(num: str) -> str:
    num = num.upper()
    if not re.fullmatch(r"[A-Z]\d{7}", num):
        return "INVALID"
    return f"{num[0]}XXX{num[-4:]}"


# --------------------------------------------------------------- helpers
_DATE_RE = r"(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})"
_YEAR_RE = r"(19\d{2}|20\d{2})"


def _first_match(patterns: list[str], text: str, flags: int = re.I) -> str | None:
    for p in patterns:
        m = re.search(p, text, flags)
        if m:
            return m.group(1).strip()
    return None


# --------------------------------------------------------------- aadhaar
def parse_aadhaar(text: str) -> dict:
    fields: dict[str, str] = {}
    expected = 4

    # Tight separator (space or dash only) so a label like "Year of Birth:
    # 1985" followed by a newline + an 8-digit run can't be glued into a
    # phantom 12-digit match.
    num = _first_match([r"\b(\d{4}[ -]?\d{4}[ -]?\d{4})\b"], text)
    if num:
        fields["aadhaar_number_masked"] = _mask_aadhaar(num)

    dob = _first_match(
        [
            rf"(?:DOB|Date of Birth)[:\s]*{_DATE_RE}",
            rf"(?:Year of Birth|YoB)[:\s]*{_YEAR_RE}",
            rf"जन्म\s*तिथि[:\s]*{_DATE_RE}",
        ],
        text,
    )
    if dob:
        fields["dob"] = dob

    gender = _first_match(
        [r"\b(MALE|FEMALE|TRANSGENDER)\b", r"\b(Male|Female|Transgender)\b"],
        text,
        flags=0,
    )
    if gender:
        fields["gender"] = gender.capitalize()

    name = _first_match(
        [
            r"(?:Name|नाम)[:\s]+([A-Z][A-Za-z][A-Za-z .]{1,60})",
            # fallback: a line of CAPS letters not containing typical keywords
            r"^([A-Z][A-Z .]{4,40})\s*$",
        ],
        text,
    )
    if name and not any(
        x in name.upper() for x in ("GOVERNMENT", "INDIA", "AADHAAR", "UNIQUE")
    ):
        fields["name"] = name

    return {"fields": fields, "parse_confidence": round(len(fields) / expected, 2)}


# --------------------------------------------------------------- pan
def parse_pan(text: str) -> dict:
    fields: dict[str, str] = {}
    expected = 4

    num = _first_match([r"\b([A-Z]{5}\d{4}[A-Z])\b"], text, flags=0)
    if num:
        fields["pan_number_masked"] = _mask_pan(num)

    dob = _first_match(
        [rf"(?:DOB|Date of Birth|जन्म\s*तिथि)[:\s]*{_DATE_RE}", rf"\b{_DATE_RE}\b"],
        text,
    )
    if dob:
        fields["dob"] = dob

    name = _first_match(
        [
            r"(?:Name|नाम)[:\s]+([A-Z][A-Za-z .]{1,60})",
            r"Name\s*\n\s*([A-Z][A-Z .]{2,60})",
        ],
        text,
    )
    if name:
        fields["name"] = name.strip()

    father = _first_match(
        [
            r"(?:Father['’]?s?\s*Name|पिता\s*का\s*नाम)[:\s]+([A-Z][A-Za-z .]{1,60})",
            r"Father.*?\n\s*([A-Z][A-Z .]{2,60})",
        ],
        text,
    )
    if father:
        fields["fathers_name"] = father.strip()

    return {"fields": fields, "parse_confidence": round(len(fields) / expected, 2)}


# --------------------------------------------------------------- voter id
def parse_voter_id(text: str) -> dict:
    fields: dict[str, str] = {}
    expected = 6   # epic, name, father_name, gender, dob_or_age, address

    epic = _first_match([r"\b([A-Z]{3}\d{7})\b"], text, flags=0)
    if epic:
        fields["identity_card_no"] = epic

    name = _first_match(
        [
            r"(?:Elector['’]?s?\s*Name|Name)[:\s]+([A-Z][A-Za-z .]{1,60})",
            r"नाम[:\s]+([A-Za-z .]{2,60})",
        ],
        text,
    )
    if name:
        fields["name"] = name.strip()

    relation = _first_match(
        [
            r"(?:Father['’]?s?\s*Name|Husband['’]?s?\s*Name)[:\s]+([A-Z][A-Za-z .]{1,60})",
            r"(?:पति|पिता)\s*का\s*नाम[:\s]+([A-Za-z .]{2,60})",
        ],
        text,
    )
    if relation:
        fields["father_name"] = relation.strip()

    gender = _first_match(
        [r"(?:Sex|Gender)[:\s]+(MALE|FEMALE|TRANSGENDER|Male|Female|Transgender)"],
        text,
        flags=re.I,
    )
    if not gender:
        gender = _first_match(
            [r"\b(MALE|FEMALE|TRANSGENDER|Male|Female|Transgender)\b"], text, flags=0
        )
    if gender:
        fields["gender"] = gender.capitalize()

    dob = _first_match(
        [rf"(?:DOB|Date of Birth)[:\s]*{_DATE_RE}", r"Age[:\s]*(\d{1,3})"],
        text,
    )
    if dob:
        fields["dob_or_age"] = dob

    address = _first_match(
        [r"(?:Address|पता)[:\s]+([A-Za-z0-9 ,./\-]{8,200})"], text
    )
    if address:
        fields["address"] = address.strip()

    return {"fields": fields, "parse_confidence": round(len(fields) / expected, 2)}


# --------------------------------------------------------------- driving licence
def parse_driving_licence(text: str) -> dict:
    fields: dict[str, str] = {}
    expected = 8   # licence_no, name, father_name, dob, gender, validity, date_of_issue, address

    # State-prefixed DL numbers vary; this is intentionally loose.
    dl = _first_match(
        [
            r"\b([A-Z]{2}[- ]?\d{2}\s*\d{4}\s*\d{7})\b",   # MH-04 2019 0012345
            r"\b([A-Z]{2}\d{13,14})\b",                     # MH04201900123450
            r"(?:DL\s*No\.?|DL\s*Number|Licence\s*No\.?)[:\s]+([A-Z0-9\-/ ]{8,25})",
        ],
        text,
        flags=0,
    )
    if dl:
        fields["licence_no"] = dl.strip()

    name = _first_match([r"(?:Name|NAME)[:\s]+([A-Z][A-Za-z .]{1,60})"], text)
    if name:
        fields["name"] = name.strip()

    father = _first_match(
        [
            r"(?:Father['’]?s?\s*Name|S/O|D/O|W/O)[:\s]+([A-Z][A-Za-z .]{1,60})",
        ],
        text,
        flags=re.I,
    )
    if father:
        fields["father_name"] = father.strip()

    dob = _first_match([rf"(?:DOB|Date of Birth)[:\s]*{_DATE_RE}"], text)
    if dob:
        fields["dob"] = dob

    gender = _first_match(
        [r"(?:Sex|Gender)[:\s]+(MALE|FEMALE|TRANSGENDER|Male|Female|Transgender)"],
        text,
        flags=re.I,
    )
    if gender:
        fields["gender"] = gender.capitalize()

    validity = _first_match(
        [rf"(?:Valid\s*Till|Validity|Expiry|Valid\s*Upto)[:\s]*{_DATE_RE}"], text
    )
    if validity:
        fields["validity"] = validity

    date_of_issue = _first_match(
        [rf"(?:Date\s*of\s*Issue|Issue\s*Date|DOI)[:\s]*{_DATE_RE}"], text
    )
    if date_of_issue:
        fields["date_of_issue"] = date_of_issue

    address = _first_match(
        [r"(?:Address|Add)[:\s]+([A-Za-z0-9 ,./\-]{8,200})"], text
    )
    if address:
        fields["address"] = address.strip()

    return {"fields": fields, "parse_confidence": round(len(fields) / expected, 2)}


# --------------------------------------------------------------- passport
def parse_passport(text: str) -> dict:
    fields: dict[str, str] = {}
    expected = 11  # name, surname, address, dob, gender, expiry,
                   # date_of_issue, place_of_issue, mother_name,
                   # legal_guardian, file_no, spouse, passport_number

    num = _first_match([r"\b([A-Z]\d{7})\b"], text, flags=0)
    if num:
        fields["passport_number_masked"] = _mask_passport(num)

    surname = _first_match(
        [r"(?:Surname|उपनाम)[:\s]+([A-Z][A-Z .]{1,40})"], text, flags=re.I
    )
    if surname:
        fields["surname"] = surname.strip()

    given = _first_match(
        [r"(?:Given\s*Name(?:\(s\))?|Given Names)[:\s]+([A-Z][A-Z .]{1,60})"],
        text,
        flags=re.I,
    )
    if given:
        fields["given_name"] = given.strip()

    address = _first_match(
        [r"(?:Address|Residential\s*Address)[:\s]+([A-Za-z0-9 ,./\-]{8,200})"], text
    )
    if address:
        fields["address"] = address.strip()

    dob = _first_match([rf"(?:Date of Birth|DOB)[:\s]*{_DATE_RE}"], text)
    if dob:
        fields["dob"] = dob

    gender = _first_match(
        [r"(?:Sex|Gender)[:\s]+([MF]|MALE|FEMALE|Male|Female)"], text, flags=re.I
    )
    if gender:
        g = gender.upper()
        fields["gender"] = "Male" if g in ("M", "MALE") else (
            "Female" if g in ("F", "FEMALE") else gender.capitalize()
        )

    date_of_issue = _first_match(
        [rf"(?:Date of Issue|Issue\s*Date|DOI)[:\s]*{_DATE_RE}"], text
    )
    if date_of_issue:
        fields["date_of_issue"] = date_of_issue

    expiry = _first_match(
        [rf"(?:Date of Expiry|Expiry|Valid\s*Upto)[:\s]*{_DATE_RE}"], text
    )
    if expiry:
        fields["expiry"] = expiry

    place_of_issue = _first_match(
        [r"(?:Place of Issue|Issuing\s*Authority)[:\s]+([A-Z][A-Za-z .]{1,60})"], text
    )
    if place_of_issue:
        fields["place_of_issue"] = place_of_issue.strip()

    mother = _first_match(
        [r"(?:Mother['’]?s?\s*Name|Name\s*of\s*Mother)[:\s]+([A-Z][A-Za-z .]{1,60})"],
        text,
        flags=re.I,
    )
    if mother:
        fields["mother_name"] = mother.strip()

    guardian = _first_match(
        [r"(?:Legal\s*Guardian|Guardian)[:\s]+([A-Z][A-Za-z .]{1,60})"], text, flags=re.I
    )
    if guardian:
        fields["legal_guardian"] = guardian.strip()

    file_no = _first_match(
        [r"(?:File\s*No\.?|File\s*Number)[:\s]+([A-Z0-9\-/]{4,20})"], text, flags=re.I
    )
    if file_no:
        fields["file_no"] = file_no.strip()

    spouse = _first_match(
        [r"(?:Name\s*of\s*Spouse|Spouse['’]?s?\s*Name)[:\s]+([A-Z][A-Za-z .]{1,60})"],
        text,
        flags=re.I,
    )
    if spouse:
        fields["spouse"] = spouse.strip()

    return {"fields": fields, "parse_confidence": round(len(fields) / expected, 2)}


# --------------------------------------------------------------- dispatch
_PARSERS: dict[str, Callable[[str], dict]] = {
    DOC_AADHAAR: parse_aadhaar,
    DOC_PAN: parse_pan,
    DOC_VOTER: parse_voter_id,
    DOC_DL: parse_driving_licence,
    DOC_PASSPORT: parse_passport,
}


def parse(doc_type: str, text: str) -> dict:
    """Dispatch to the right parser; raises ValueError for unknown types."""
    fn = _PARSERS.get(doc_type)
    if fn is None:
        raise ValueError(f"unsupported doc_type: {doc_type}")
    return fn(text)
