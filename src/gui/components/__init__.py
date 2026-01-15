"""
GUI Components Module.

This module provides reusable UI component factories for the Homeguard GUI.
All factories return standard Flet components with consistent styling.

Usage:
    from src.gui.components import (
        create_section_container,
        create_number_field,
        create_labeled_dropdown
    )

    container = create_section_container(
        title="My Section",
        children=[...],
        border_color=ft.Colors.BLUE_700
    )
"""

from .factories import (
    create_section_container,
    create_number_field,
    create_text_field,
    create_labeled_dropdown,
    create_checkbox,
    create_action_button,
    create_icon_button,
    create_slider_with_label,
    create_date_picker_row,
)

__all__ = [
    'create_section_container',
    'create_number_field',
    'create_text_field',
    'create_labeled_dropdown',
    'create_checkbox',
    'create_action_button',
    'create_icon_button',
    'create_slider_with_label',
    'create_date_picker_row',
]
