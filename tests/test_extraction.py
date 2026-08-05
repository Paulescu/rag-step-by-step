from __future__ import annotations

from customer_support_chatbot.ingestion.extraction import quarantine_reason
from customer_support_chatbot.ingestion.models import Page


def test_a_file_with_no_pages_is_quarantined() -> None:
    assert quarantine_reason([]) is not None


def test_a_scan_with_an_empty_text_layer_is_quarantined() -> None:
    pages = [Page(number=number, text="") for number in range(1, 6)]

    reason = quarantine_reason(pages)

    assert reason is not None
    assert "scan" in reason


def test_a_document_with_real_text_is_not_quarantined() -> None:
    pages = [Page(number=number, text="a" * 500) for number in range(1, 4)]

    assert quarantine_reason(pages) is None


def test_one_good_page_does_not_rescue_a_mostly_empty_document() -> None:
    pages = [Page(number=1, text="a" * 400)] + [
        Page(number=number, text="") for number in range(2, 10)
    ]

    assert quarantine_reason(pages) is not None
