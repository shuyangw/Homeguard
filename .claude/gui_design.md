# GUI Design Standards

**CRITICAL**: All GUI elements MUST follow the dark theme design with bright text for readability.

## Color Scheme Requirements

**Background Colors**:
- Primary background: Dark gray/black (`#1e1e1e`, `#2d2d2d`)
- Secondary background: Slightly lighter dark (`#3d3d3d`)
- Container backgrounds: Dark gray (`#2d2d2d`, `GREY_900`, `GREY_800`)

**Text Colors**:
- Primary text: **Bright white** (`#e0e0e0`, `WHITE`)
- Secondary text: Light gray (`#a0a0a0`, `GREY_400`)
- Disabled text: Medium gray (`GREY_600`)
- **NEVER use dark text on dark backgrounds**

**Accent Colors** (Consistent Theme):
- **Primary Accent**: Blue (`#6366f1`, `BLUE_700`, `INDIGO_600`)
  - Used for: Primary buttons, important actions, links
- **Success/Positive**: Green (`#10b981`, `GREEN_700`, `GREEN_400`)
  - Used for: Success messages, positive metrics, profit indicators
- **Warning**: Orange/Amber (`#f59e0b`, `ORANGE_700`, `AMBER_600`)
  - Used for: Warnings, moderate risk, cache management
- **Error/Negative**: Red (`#ef4444`, `RED_700`, `RED_400`)
  - Used for: Errors, losses, delete actions
- **Info**: Cyan (`#3b82f6`, `CYAN_700`)
  - Used for: Informational messages, neutral actions
- **Secondary Action**: Purple (`#9333ea`, `PURPLE_700`)
  - Used for: Secondary features, quick actions

## UI Component Standards

**Buttons**:
```python
# Primary action (blue)
ft.ElevatedButton(
    "Run Backtests",
    style=ft.ButtonStyle(
        color=ft.Colors.WHITE,  # Bright text
        bgcolor=ft.Colors.BLUE_700  # Primary accent
    )
)

# Success action (green)
ft.ElevatedButton(
    "Load Cached",
    style=ft.ButtonStyle(
        color=ft.Colors.WHITE,
        bgcolor=ft.Colors.GREEN_700
    )
)

# Warning/caution (orange)
ft.ElevatedButton(
    "Clear Cache",
    style=ft.ButtonStyle(
        color=ft.Colors.WHITE,
        bgcolor=ft.Colors.ORANGE_700
    )
)

# Destructive action (red)
ft.ElevatedButton(
    "Delete",
    style=ft.ButtonStyle(
        color=ft.Colors.WHITE,
        bgcolor=ft.Colors.RED_700
    )
)
```

**Containers**:
```python
# Section container
ft.Container(
    content=...,
    border=ft.border.all(2, ft.Colors.BLUE_700),  # Colored border
    border_radius=8,
    padding=15,
    bgcolor=ft.Colors.GREY_900  # Dark background
)

# Card-style container
ft.Container(
    content=...,
    bgcolor=ft.Colors.GREY_800,
    border_radius=12,
    padding=20
)
```

**Text Elements**:
```python
# Headers - bright and prominent
ft.Text(
    "Section Title",
    size=18,
    weight=ft.FontWeight.W_500,
    color=ft.Colors.WHITE  # Bright white
)

# Body text - readable
ft.Text(
    "Description text",
    size=13,
    color=ft.Colors.WHITE  # Bright white, not gray
)

# Secondary/helper text - slightly dimmer
ft.Text(
    "Helper text",
    size=11,
    color=ft.Colors.GREY_400  # Light gray, still readable
)
```

## Consistent Border Colors

Use colored borders to visually group related sections:
- **Blue**: Configuration, general inputs
- **Green**: Results, success indicators
- **Orange**: Dates, time-related
- **Purple**: Presets, saved configurations
- **Cyan**: Symbol lists, data inputs
- **Red**: Errors, warnings

## Visual Hierarchy

**Primary Elements** (most prominent):
- Bright white text (#FFFFFF)
- Bold or larger font
- Colored backgrounds (blue, green, etc.)

**Secondary Elements** (readable):
- White text (#e0e0e0)
- Normal weight
- Subtle borders

**Tertiary Elements** (de-emphasized):
- Light gray text (#a0a0a0 - GREY_400)
- Smaller font
- Dim backgrounds

## Color Consistency Rules

1. **Never mix color schemes** - stick to one accent color per section
2. **Use semantic colors** - green = success, red = error, blue = primary
3. **Maintain contrast** - always test text readability on backgrounds
4. **Be consistent** - same action = same color across all views

## Examples of INCORRECT styling:

[-] Dark text on dark background:
```python
ft.Text("Hard to read", color=ft.Colors.GREY_800)  # Too dark!
```

[-] Inconsistent button colors:
```python
# Run button is blue in one view, green in another
```

[-] Low contrast:
```python
ft.Container(
    bgcolor=ft.Colors.GREY_700,
    content=ft.Text("Text", color=ft.Colors.GREY_600)  # Hard to read
)
```

## Examples of CORRECT styling:

[+] Bright text on dark background:
```python
ft.Container(
    bgcolor=ft.Colors.GREY_900,
    content=ft.Text("Readable", color=ft.Colors.WHITE, size=14)
)
```

[+] Consistent accent usage:
```python
# Strategy section - blue theme
ft.Container(border=ft.border.all(2, ft.Colors.BLUE_700))

# Date section - orange theme
ft.Container(border=ft.border.all(2, ft.Colors.ORANGE_700))

# Symbol section - cyan theme
ft.Container(border=ft.border.all(2, ft.Colors.CYAN_700))
```

[+] Proper visual hierarchy:
```python
ft.Column([
    ft.Text("Header", size=18, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
    ft.Text("Body", size=13, color=ft.Colors.WHITE),
    ft.Text("Caption", size=11, color=ft.Colors.GREY_400)
])
```

## Testing Readability

Before committing GUI changes:
1. **Run the GUI** - visually verify all text is readable
2. **Check contrast** - no dark text on dark backgrounds
3. **Verify consistency** - same colors for same actions across views
4. **Test on actual dark theme** - ensure colors work in dark mode

## Accessibility

- Minimum contrast ratio: 4.5:1 for body text
- Minimum contrast ratio: 3:1 for large text (≥18pt)
- Use icons with text labels, not icons alone
- Ensure color is not the only indicator (use icons, text, etc.)
