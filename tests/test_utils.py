import pytest

from src.utils import parse_xml_content


def test_parse_xml_content_with_match():
    """Test parse_xml_content when a match is found"""
    text = "<example>Hello World</example>"
    result = parse_xml_content(text, "example")
    assert result == "Hello World"


def test_parse_xml_content_with_match_and_whitespace():
    """Test parse_xml_content when a match is found with whitespace"""
    text = "<example>  Hello World  </example>"
    result = parse_xml_content(text, "example")
    assert result == "Hello World"


def test_parse_xml_content_with_multiline():
    """Test parse_xml_content with multiline content"""
    text = "<example>\nHello\nWorld\n</example>"
    result = parse_xml_content(text, "example")
    assert result == "Hello\nWorld"


def test_parse_xml_content_no_match():
    """Test parse_xml_content when no match is found"""
    text = "<other>Hello World</other>"
    result = parse_xml_content(text, "example")
    assert result == ""


def test_parse_xml_content_empty_text():
    """Test parse_xml_content with empty text"""
    text = ""
    result = parse_xml_content(text, "example")
    assert result == ""


def test_parse_xml_content_empty_content():
    """Test parse_xml_content with empty content between tags"""
    text = "<example></example>"
    result = parse_xml_content(text, "example")
    assert result == ""


def test_parse_xml_content_with_nested_tags():
    """Test parse_xml_content with nested content"""
    text = "<example><inner>Hello</inner> World</example>"
    result = parse_xml_content(text, "example")
    assert result == "<inner>Hello</inner> World"