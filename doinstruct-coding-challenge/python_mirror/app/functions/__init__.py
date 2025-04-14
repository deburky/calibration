"""
Functions package for the application.
Contains Lambda handlers.
"""

from app.functions.generate_cards import handler as generate_cards_handler

__all__ = ["generate_cards_handler"]