"""
Utility modules for the Honcho app.
""" 
import re

def parse_xml_content(text: str, tag: str) -> str:
    """
    Extract content from XML-like tags in a string.
    
    Args:
        text: The text containing XML-like tags
        tag: The tag name to extract content from
        
    Returns:
        The content between the opening and closing tags, or an empty string if not found
    """
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""