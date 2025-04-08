from pykospacing import Spacing

def correct_spacing(text):
    """
    Corrects spacing in Korean text using the PyKoSpacing library.
    
    Parameters:
    text (str): The input Korean text with incorrect spacing.
    
    Returns:
    str: The text with corrected spacing.
    """
    spacing = Spacing()
    corrected_text = spacing(text)
    return corrected_text