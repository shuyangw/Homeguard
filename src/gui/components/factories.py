"""
GUI Component Factory Functions.

This module provides factory functions for creating Flet UI components
with consistent styling patterns used throughout the Homeguard GUI.

All factories return standard Flet components - no logic changes,
just code organization for reusability.
"""

import flet as ft
from typing import List, Optional, Callable, Any


def create_section_container(
    title: str,
    children: List[ft.Control],
    border_color: str = ft.Colors.BLUE_700,
    spacing: int = 10
) -> ft.Container:
    """
    Create a styled section container with title and children.

    Args:
        title: Section title text
        children: List of child controls
        border_color: Border color (default: BLUE_700)
        spacing: Spacing between children (default: 10)

    Returns:
        Container with styled border, background, and title
    """
    return ft.Container(
        content=ft.Column(
            [ft.Text(title, size=18, weight=ft.FontWeight.W_500)] + children,
            spacing=spacing
        ),
        border=ft.border.all(2, border_color),
        border_radius=8,
        padding=15,
        bgcolor=ft.Colors.GREY_900
    )


def create_number_field(
    label: str,
    value: str = "0",
    width: int = 190,
    hint_text: Optional[str] = None,
    tooltip: Optional[str] = None
) -> ft.TextField:
    """
    Create a number input text field.

    Args:
        label: Field label
        value: Default value (default: "0")
        width: Field width (default: 190)
        hint_text: Optional hint text
        tooltip: Optional tooltip

    Returns:
        TextField configured for number input
    """
    return ft.TextField(
        label=label,
        value=value,
        keyboard_type=ft.KeyboardType.NUMBER,
        width=width,
        hint_text=hint_text,
        tooltip=tooltip
    )


def create_text_field(
    label: str,
    value: str = "",
    width: int = 400,
    hint_text: Optional[str] = None,
    tooltip: Optional[str] = None,
    multiline: bool = False,
    read_only: bool = False
) -> ft.TextField:
    """
    Create a standard text input field.

    Args:
        label: Field label
        value: Default value
        width: Field width (default: 400)
        hint_text: Optional hint text
        tooltip: Optional tooltip
        multiline: Enable multiline input
        read_only: Make field read-only

    Returns:
        TextField with standard styling
    """
    return ft.TextField(
        label=label,
        value=value,
        width=width,
        hint_text=hint_text,
        tooltip=tooltip,
        multiline=multiline,
        read_only=read_only
    )


def create_labeled_dropdown(
    label: str,
    options: List[ft.dropdown.Option],
    value: Optional[str] = None,
    width: int = 400,
    on_change: Optional[Callable] = None,
    tooltip: Optional[str] = None,
    hint_text: Optional[str] = None
) -> ft.Dropdown:
    """
    Create a dropdown with label and options.

    Args:
        label: Dropdown label
        options: List of dropdown options
        value: Default selected value
        width: Dropdown width (default: 400)
        on_change: Optional change handler
        tooltip: Optional tooltip
        hint_text: Optional hint text

    Returns:
        Dropdown with standard styling
    """
    return ft.Dropdown(
        label=label,
        options=options,
        value=value,
        width=width,
        on_change=on_change,
        tooltip=tooltip,
        hint_text=hint_text
    )


def create_checkbox(
    label: str,
    value: bool = False,
    on_change: Optional[Callable] = None,
    tooltip: Optional[str] = None
) -> ft.Checkbox:
    """
    Create a checkbox with label.

    Args:
        label: Checkbox label
        value: Default checked state
        on_change: Optional change handler
        tooltip: Optional tooltip

    Returns:
        Checkbox with standard styling
    """
    return ft.Checkbox(
        label=label,
        value=value,
        on_change=on_change,
        tooltip=tooltip
    )


def create_action_button(
    text: str,
    icon: str,
    on_click: Optional[Callable] = None,
    bgcolor: str = ft.Colors.GREEN_700,
    color: str = ft.Colors.WHITE,
    height: int = 50,
    width: int = 200,
    tooltip: Optional[str] = None,
    disabled: bool = False
) -> ft.ElevatedButton:
    """
    Create an action button with icon.

    Args:
        text: Button text
        icon: Icon name (e.g., ft.Icons.PLAY_ARROW)
        on_click: Optional click handler
        bgcolor: Background color (default: GREEN_700)
        color: Text color (default: WHITE)
        height: Button height (default: 50)
        width: Button width (default: 200)
        tooltip: Optional tooltip
        disabled: Disabled state

    Returns:
        ElevatedButton with standard styling
    """
    return ft.ElevatedButton(
        text,
        icon=icon,
        on_click=on_click,
        style=ft.ButtonStyle(
            color=color,
            bgcolor=bgcolor
        ),
        height=height,
        width=width,
        tooltip=tooltip,
        disabled=disabled
    )


def create_icon_button(
    icon: str,
    on_click: Optional[Callable] = None,
    tooltip: Optional[str] = None,
    icon_color: Optional[str] = None
) -> ft.IconButton:
    """
    Create an icon button.

    Args:
        icon: Icon name (e.g., ft.Icons.DOWNLOAD)
        on_click: Optional click handler
        tooltip: Optional tooltip
        icon_color: Optional icon color

    Returns:
        IconButton with standard styling
    """
    return ft.IconButton(
        icon=icon,
        on_click=on_click,
        tooltip=tooltip,
        icon_color=icon_color
    )


def create_slider_with_label(
    label_prefix: str,
    min_val: int = 1,
    max_val: int = 16,
    value: int = 4,
    divisions: int = 15,
    on_change: Optional[Callable] = None,
    width: int = 350,
    tooltip: Optional[str] = None,
    recommended_info: Optional[str] = None
) -> tuple:
    """
    Create a slider with accompanying label text.

    Args:
        label_prefix: Prefix for the label (e.g., "Parallel Workers")
        min_val: Minimum slider value
        max_val: Maximum slider value
        value: Default value
        divisions: Number of slider divisions
        on_change: Optional change handler
        width: Slider width
        tooltip: Optional tooltip
        recommended_info: Optional text about recommended value

    Returns:
        Tuple of (slider, label_text)
    """
    label_text = f"{label_prefix}: {value}"
    if recommended_info:
        label_text = f"{label_prefix}: {value} ({recommended_info})"

    slider = ft.Slider(
        min=min_val,
        max=max_val,
        divisions=divisions,
        value=value,
        label="{value} workers",
        on_change=on_change,
        width=width,
        tooltip=tooltip
    )

    text = ft.Text(label_text, size=14)

    return slider, text


def create_date_picker_row(
    label: str,
    value: str,
    on_calendar_click: Optional[Callable] = None,
    field_width: int = 140,
    tooltip: Optional[str] = None
) -> tuple:
    """
    Create a date picker row with text field and calendar button.

    Args:
        label: Field label (e.g., "Start Date")
        value: Default date value (YYYY-MM-DD format)
        on_calendar_click: Calendar button click handler
        field_width: Text field width
        tooltip: Optional tooltip

    Returns:
        Tuple of (text_field, calendar_button)
    """
    text_field = ft.TextField(
        label=label,
        hint_text="YYYY-MM-DD",
        value=value,
        width=field_width,
        read_only=True,
        tooltip=tooltip
    )

    calendar_button = ft.IconButton(
        icon=ft.Icons.CALENDAR_MONTH,
        tooltip=f"Select {label.lower()} from calendar",
        on_click=on_calendar_click
    )

    return text_field, calendar_button
