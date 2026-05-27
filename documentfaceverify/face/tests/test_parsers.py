"""
Parser tests use synthetic text — no PDFs / images, no OCR.

For each doc type we feed a representative text blob and assert the right
fields come out AND that sensitive numbers are masked at the parser level
(so they cannot leak even if the response is logged).
"""

import pytest

from aav.pipeline import parsers


# -------------------------------------------------------- aadhaar
def test_aadhaar_masks_number_and_extracts_dob():
    text = """
    Government of India
    UNIQUE IDENTIFICATION AUTHORITY
    Name: Asha Rani Kumari
    DOB: 14/08/1992
    Female
    1234 5678 9012
    """
    result = parsers.parse_aadhaar(text)
    assert result["fields"]["aadhaar_number_masked"] == "XXXX-XXXX-9012"
    assert result["fields"]["dob"] == "14/08/1992"
    assert result["fields"]["gender"] == "Female"
    assert "Asha" in result["fields"]["name"]
    assert result["parse_confidence"] == 1.0


def test_aadhaar_year_of_birth_variant():
    text = "Year of Birth: 1985\n5555 4444 3333\nMale"
    result = parsers.parse_aadhaar(text)
    assert result["fields"]["dob"] == "1985"
    assert result["fields"]["aadhaar_number_masked"].endswith("3333")
    assert result["parse_confidence"] >= 0.5


# -------------------------------------------------------- pan
def test_pan_masks_number_and_extracts_fields():
    text = """
    INCOME TAX DEPARTMENT
    Name
    RAHUL VERMA
    Father's Name
    SUNIL VERMA
    Date of Birth: 02/11/1990
    Permanent Account Number: ABCDE1234F
    """
    result = parsers.parse_pan(text)
    masked = result["fields"]["pan_number_masked"]
    assert masked.startswith("ABC") and masked.endswith("F")
    assert "1234" in masked
    assert result["fields"]["dob"] == "02/11/1990"
    assert "RAHUL" in result["fields"]["name"]


def test_pan_rejects_malformed_number():
    # 5 letters + 4 digits + 1 letter is required; this is wrong shape.
    text = "ABCD12345F"
    result = parsers.parse_pan(text)
    assert "pan_number_masked" not in result["fields"]


# -------------------------------------------------------- voter id
def test_voter_extracts_epic_and_relation():
    text = """
    ELECTION COMMISSION OF INDIA
    XYZ1234567
    Elector's Name: Priya Sharma
    Husband's Name: Rohit Sharma
    Sex: Female
    Age: 34
    Address: 12 MG Road, Pune
    """
    result = parsers.parse_voter_id(text)
    assert result["fields"]["identity_card_no"] == "XYZ1234567"
    assert "Priya" in result["fields"]["name"]
    assert "Rohit" in result["fields"]["father_name"]
    assert result["fields"]["gender"] == "Female"
    assert "MG Road" in result["fields"]["address"]


# -------------------------------------------------------- driving licence
def test_dl_extracts_number_and_validity():
    text = """
    DRIVING LICENCE
    DL No.: MH-04 2019 0012345
    Name: AMIT KUMAR
    Father's Name: KIRAN KUMAR
    DOB: 12/12/1990
    Sex: Male
    Date of Issue: 12/12/2010
    Valid Till: 11/12/2030
    Address: 7 Lane, Pune
    """
    result = parsers.parse_driving_licence(text)
    assert "MH" in result["fields"]["licence_no"]
    assert "AMIT" in result["fields"]["name"]
    assert "KIRAN" in result["fields"]["father_name"]
    assert result["fields"]["dob"] == "12/12/1990"
    assert result["fields"]["validity"] == "11/12/2030"
    assert result["fields"]["date_of_issue"] == "12/12/2010"
    assert result["fields"]["gender"] == "Male"
    assert "Lane" in result["fields"]["address"]


# -------------------------------------------------------- passport
def test_passport_masks_number_and_extracts_fields():
    text = """
    REPUBLIC OF INDIA PASSPORT
    Surname: GUPTA
    Given Name(s): NEHA
    Date of Birth: 05/06/1988
    Date of Expiry: 04/06/2033
    Passport No.: M1234567
    """
    result = parsers.parse_passport(text)
    masked = result["fields"]["passport_number_masked"]
    assert masked.startswith("M") and masked.endswith("4567")
    assert "X" in masked
    assert result["fields"]["surname"] == "GUPTA"
    assert "NEHA" in result["fields"]["given_name"]
    assert result["fields"]["dob"] == "05/06/1988"
    assert result["fields"]["expiry"] == "04/06/2033"


# -------------------------------------------------------- dispatch
def test_dispatch_known_type():
    out = parsers.parse("aadhaar", "DOB: 01/01/2000\n1111 2222 3333")
    assert "aadhaar_number_masked" in out["fields"]


def test_dispatch_unknown_type_raises():
    with pytest.raises(ValueError):
        parsers.parse("ration_card", "anything")
