import pytest
from pydantic import ValidationError

from app.schemas import AskRequest


def test_valid_question():
    request = AskRequest(question="What is network security?")

    assert request.question == "What is network security?"
    assert request.top_k is None


def test_question_whitespace_is_trimmed():
    request = AskRequest(question="  What is network security?  ")

    assert request.question == "What is network security?"


def test_reject_short_question():
    with pytest.raises(ValidationError):
        AskRequest(question="Hi")


def test_reject_whitespace_question():
    with pytest.raises(ValidationError):
        AskRequest(question="   ")


def test_reject_excessive_top_k():
    with pytest.raises(ValidationError):
        AskRequest(question="Explain network security", top_k=11)
