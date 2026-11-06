"""
Dark and light themes for the S2L application.

Design language:
  - Charcoal (night) or soft-white (day) surfaces with subtle layering
  - Indigo/violet accent for interactive elements
  - Card-based layout with soft rounded corners and 1px borders

``COLORS`` is a live dict: :func:`set_theme` updates it in place, so code that
imported it (custom-painted widgets, matplotlib plots) always sees the current
palette. Widgets are styled through the global stylesheet using object names
and the ``variant`` property instead of per-widget stylesheets, so switching
theme only needs ``app.setStyleSheet(get_complete_stylesheet())``.
"""
import tempfile
from pathlib import Path

# ── colour tokens ──────────────────────────────────────────────────────────

PALETTES = {
    "dark": {
        # accent
        "accent":          "#7c6aef",
        "accent_hover":    "#9585f5",
        "accent_pressed":  "#6455d0",
        "accent_muted":    "rgba(124, 106, 239, 30)",
        "accent_glow":     "rgba(124, 106, 239, 64)",
        # surfaces
        "bg_window":       "#101014",
        "bg_base":         "#16161a",
        "bg_card":         "#1e1e24",
        "bg_input":        "#25252d",
        "bg_hover":        "#2c2c36",
        "bg_elevated":     "#323240",
        "canvas":          "#0d0d0f",   # behind microscopy images
        # text
        "text":            "#eaeaef",
        "text_secondary":  "#a0a0b0",
        "text_dim":        "#65657a",
        "text_on_accent":  "#ffffff",
        # semantic
        "success":         "#34d399",
        "success_bg":      "rgba(52, 211, 153, 28)",
        "warning":         "#fbbf24",
        "error":           "#f87171",
        "error_bg":        "rgba(248, 113, 113, 25)",
        # borders
        "border":          "#2a2a35",
        "border_subtle":   "#222230",
        "border_focus":    "#7c6aef",
        # plots
        "plot_grid":       "#2e2e38",
    },
    "light": {
        "accent":          "#6552e0",
        "accent_hover":    "#7a69ec",
        "accent_pressed":  "#5241c4",
        "accent_muted":    "rgba(101, 82, 224, 28)",
        "accent_glow":     "rgba(101, 82, 224, 60)",
        "bg_window":       "#f4f5f9",
        "bg_base":         "#ffffff",
        "bg_card":         "#ffffff",
        "bg_input":        "#f0f1f6",
        "bg_hover":        "#e8e9f0",
        "bg_elevated":     "#d9dae4",
        "canvas":          "#3a3a42",
        "text":            "#1c1c26",
        "text_secondary":  "#545468",
        "text_dim":        "#8a8a9c",
        "text_on_accent":  "#ffffff",
        "success":         "#059669",
        "success_bg":      "rgba(5, 150, 105, 22)",
        "warning":         "#b45309",
        "error":           "#dc2626",
        "error_bg":        "rgba(220, 38, 38, 18)",
        "border":          "#dcdde6",
        "border_subtle":   "#e9eaf0",
        "border_focus":    "#6552e0",
        "plot_grid":       "#d9dae4",
    },
}

COLORS = dict(PALETTES["dark"])
C = COLORS  # shorthand used inside f-strings
_current = "dark"


def set_theme(name: str):
    """Switch the live palette ("dark" or "light")."""
    global _current
    if name not in PALETTES:
        name = "dark"
    _current = name
    COLORS.clear()
    COLORS.update(PALETTES[name])


def current_theme() -> str:
    return _current


# ── reusable style fragments ──────────────────────────────────────────────

FONT_STACK = "'Segoe UI', 'Inter', 'SF Pro Display', 'Helvetica Neue', Arial, sans-serif"


def _icon_dir() -> Path:
    d = Path(tempfile.gettempdir()) / "s2l_theme_icons"
    d.mkdir(exist_ok=True)
    return d


def _svg_icon(name: str, svg: str) -> str:
    """Write a small SVG for the stylesheet and return its url() path."""
    path = _icon_dir() / f"{_current}_{name}.svg"
    if not path.exists() or path.read_text() != svg:
        path.write_text(svg)
    return path.as_posix()


def _icons() -> dict:
    chevron = ('<svg xmlns="http://www.w3.org/2000/svg" width="10" height="6" viewBox="0 0 10 6">'
               '<path d="M1 1 L5 5 L9 1" fill="none" stroke="{c}" stroke-width="1.6" '
               'stroke-linecap="round" stroke-linejoin="round" transform="{t}"/></svg>')
    tick = ('<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 14 14">'
            '<path d="M3 7.5 L6 10.5 L11.5 4" fill="none" stroke="#ffffff" stroke-width="2" '
            'stroke-linecap="round" stroke-linejoin="round"/></svg>')
    return {
        "down": _svg_icon("down", chevron.format(c=C["text_secondary"], t="")),
        "up": _svg_icon("up", chevron.format(c=C["text_secondary"], t="rotate(180 5 3)")),
        "tick": _svg_icon("tick", tick),
    }


def get_primary_button_style():
    """Kept for compatibility; prefer ``button.setProperty("variant", "primary")``."""
    return ""


def get_danger_button_style():
    """Kept for compatibility; prefer ``button.setProperty("variant", "danger")``."""
    return ""


def get_complete_stylesheet():
    """Master stylesheet applied to the whole application."""
    icons = _icons()
    return f"""
    /* ── base ─────────────────────────────────────────────── */
    * {{
        font-family: {FONT_STACK};
        outline: none;
    }}
    QMainWindow, QDialog {{
        background-color: {C['bg_window']};
        color: {C['text']};
    }}
    QWidget {{
        color: {C['text']};
        font-size: 13px;
    }}
    QWidget#SideBar {{
        background-color: {C['bg_base']};
        border-right: 1px solid {C['border']};
    }}
    QWidget#PageStack, QWidget#PageContent {{
        background-color: {C['bg_window']};
    }}
    QLabel#BrandLabel {{
        font-size: 20px; font-weight: 800; color: {C['accent']};
        letter-spacing: 2px; padding: 0 4px;
    }}
    QLabel#BrandSub, QLabel#VersionLabel {{
        font-size: 10px; color: {C['text_dim']}; padding: 0 4px;
    }}

    /* ── sidebar nav buttons ─────────────────────────────── */
    QPushButton#NavBtn {{
        background: transparent;
        color: {C['text_secondary']};
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 13px;
        font-weight: 500;
        text-align: left;
        min-height: 22px;
    }}
    QPushButton#NavBtn:hover {{
        background-color: {C['bg_hover']};
        color: {C['text']};
    }}
    QPushButton#NavBtn[active="true"] {{
        background-color: {C['accent_muted']};
        color: {C['accent']};
        font-weight: 600;
    }}
    QPushButton#ThemeToggle {{
        background-color: {C['bg_input']};
        border: 1px solid {C['border']};
        border-radius: 8px;
        padding: 6px 10px;
        font-size: 12px;
        text-align: left;
        min-height: 20px;
    }}
    QPushButton#ThemeToggle:hover {{ border-color: {C['accent']}; }}

    /* ── cards ────────────────────────────────────────────── */
    QFrame#Card {{
        background-color: {C['bg_card']};
        border: 1px solid {C['border']};
        border-radius: 12px;
    }}
    QFrame#ActionBar {{
        background-color: {C['bg_base']};
        border-top: 1px solid {C['border']};
    }}
    QPushButton#HomeCard {{
        background-color: {C['bg_card']};
        border: 1px solid {C['border']};
        border-radius: 14px;
        padding: 18px;
        text-align: left;
        font-size: 13px;
        color: {C['text_secondary']};
        min-height: 96px;
    }}
    QPushButton#HomeCard:hover {{
        border-color: {C['accent']};
        background-color: {C['bg_hover']};
    }}
    QLabel#HomeCardTitle {{ font-size: 15px; font-weight: 700; color: {C['text']}; }}
    QLabel#HomeCardDesc {{ font-size: 12px; color: {C['text_secondary']}; }}
    QLabel#HomeCardIcon {{ font-size: 26px; }}
    QPushButton#RecentFolder {{
        background: transparent; border: none; text-align: left;
        color: {C['accent']}; padding: 2px 0; min-height: 18px;
    }}
    QPushButton#RecentFolder:hover {{ text-decoration: underline; }}

    /* ── inputs ───────────────────────────────────────────── */
    QLineEdit, QTextEdit, QPlainTextEdit {{
        background-color: {C['bg_input']};
        color: {C['text']};
        border: 1px solid {C['border']};
        border-radius: 6px;
        padding: 9px 12px;
        font-size: 13px;
        selection-background-color: {C['accent']};
    }}
    QLineEdit {{ min-height: 20px; }}
    QTextEdit#LogBox {{
        background-color: {C['canvas']};
        color: {C['text_secondary']};
        font-family: Consolas, 'Courier New', monospace;
        font-size: 11px;
        padding: 6px;
    }}
    QSpinBox, QDoubleSpinBox {{
        background-color: {C['bg_input']};
        color: {C['text']};
        border: 1px solid {C['border']};
        border-radius: 6px;
        padding: 6px 10px;
        font-size: 13px;
        min-height: 28px;
        selection-background-color: {C['accent']};
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {{
        border-color: {C['accent']};
    }}
    QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
        border-color: {C['bg_elevated']};
    }}
    QSpinBox::up-button, QDoubleSpinBox::up-button {{
        subcontrol-origin: border;
        subcontrol-position: top right;
        background: {C['bg_hover']};
        border: none;
        border-top-right-radius: 5px;
        width: 24px;
    }}
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        background: {C['bg_hover']};
        border: none;
        border-bottom-right-radius: 5px;
        width: 24px;
    }}
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
        background: {C['bg_elevated']};
    }}
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
        image: url({icons['up']}); width: 10px; height: 6px;
    }}
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
        image: url({icons['down']}); width: 10px; height: 6px;
    }}

    /* ── combo box ────────────────────────────────────────── */
    QComboBox {{
        background-color: {C['bg_input']};
        color: {C['text']};
        border: 1px solid {C['border']};
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 13px;
        min-height: 20px;
    }}
    QComboBox:hover {{ border-color: {C['bg_elevated']}; }}
    QComboBox:focus {{ border-color: {C['accent']}; }}
    QComboBox::drop-down {{ border: none; width: 30px; }}
    QComboBox::down-arrow {{
        image: url({icons['down']}); width: 10px; height: 6px; margin-right: 10px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {C['bg_card']};
        color: {C['text']};
        border: 1px solid {C['border']};
        border-radius: 6px;
        selection-background-color: {C['accent_muted']};
        selection-color: {C['accent']};
        outline: none;
        padding: 4px;
    }}
    QComboBox QAbstractItemView::item {{
        padding: 8px 12px;
        border-radius: 4px;
    }}
    QComboBox QAbstractItemView::item:hover {{
        background-color: {C['bg_hover']};
    }}

    /* ── check box ────────────────────────────────────────── */
    QCheckBox {{
        color: {C['text']};
        font-size: 13px;
        spacing: 10px;
        padding: 4px 0;
    }}
    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border-radius: 5px;
        border: 2px solid {C['bg_elevated']};
        background-color: {C['bg_input']};
    }}
    QCheckBox::indicator:hover {{
        border-color: {C['accent']};
    }}
    QCheckBox::indicator:checked {{
        background-color: {C['accent']};
        border-color: {C['accent']};
        image: url({icons['tick']});
    }}
    QCheckBox::indicator:checked:hover {{
        background-color: {C['accent_hover']};
        border-color: {C['accent_hover']};
    }}
    QCheckBox:disabled {{
        color: {C['text_dim']};
    }}
    QCheckBox::indicator:disabled {{
        border-color: {C['border_subtle']};
        background-color: {C['bg_base']};
    }}

    /* ── progress bar ─────────────────────────────────────── */
    QProgressBar {{
        background-color: {C['bg_input']};
        border: 1px solid {C['border']};
        border-radius: 6px;
        text-align: center;
        font-size: 11px;
        font-weight: 600;
        color: {C['text_secondary']};
        min-height: 20px;
        max-height: 20px;
    }}
    QProgressBar::chunk {{
        background-color: {C['accent']};
        border-radius: 5px;
        margin: 1px;
    }}

    /* ── labels ───────────────────────────────────────────── */
    QLabel {{
        color: {C['text']};
        font-size: 13px;
        background: transparent;
    }}
    QLabel#SectionTitle {{
        font-size: 15px;
        font-weight: 700;
        color: {C['text']};
        padding: 0;
    }}
    QLabel#SectionDesc {{
        font-size: 12px;
        color: {C['text_secondary']};
        padding: 0;
    }}
    QLabel#FieldLabel {{
        font-size: 12px;
        font-weight: 600;
        color: {C['text_secondary']};
        padding: 0;
    }}
    QLabel#StatusSuccess {{ color: {C['success']}; font-weight: 600; font-size: 12px; }}
    QLabel#StatusWarning {{ color: {C['warning']}; font-weight: 600; font-size: 12px; }}
    QLabel#StatusError   {{ color: {C['error']};   font-weight: 600; font-size: 12px; }}
    QLabel#StatusPending {{ color: {C['text_dim']}; font-weight: 600; font-size: 12px; }}
    QLabel#Hint {{
        color: {C['text_dim']};
        font-size: 11px;
        font-style: italic;
    }}
    QLabel#PageTitle {{
        font-size: 22px;
        font-weight: 700;
        color: {C['text']};
        background: transparent;
    }}
    QLabel#PageSubtitle {{
        font-size: 13px;
        color: {C['text_secondary']};
        background: transparent;
    }}
    QLabel#ImagePanel {{
        background: {C['bg_input']};
        border: 1px solid {C['border']};
        border-radius: 6px;
        color: {C['text_dim']};
    }}
    QLabel#RunSummary {{
        background: {C['success_bg']};
        border: 1px solid {C['success']};
        border-radius: 8px;
        padding: 10px 12px;
        color: {C['text']};
    }}

    /* ── lists ────────────────────────────────────────────── */
    QListWidget {{
        background-color: {C['bg_input']};
        color: {C['text']};
        border: 1px solid {C['border']};
        border-radius: 6px;
        padding: 6px;
        outline: none;
    }}
    QListWidget::item {{
        color: {C['text_secondary']};
        border-radius: 6px;
        padding: 4px;
    }}
    QListWidget::item:hover {{
        background-color: {C['bg_hover']};
    }}
    QListWidget::item:selected {{
        background-color: {C['accent_muted']};
        color: {C['accent']};
    }}

    /* ── tables ───────────────────────────────────────────── */
    QTableWidget, QTableView {{
        background-color: {C['bg_input']};
        alternate-background-color: {C['bg_card']};
        color: {C['text']};
        gridline-color: {C['border']};
        border: 1px solid {C['border']};
        border-radius: 6px;
        selection-background-color: {C['accent_muted']};
        selection-color: {C['accent']};
        font-size: 12px;
    }}
    QHeaderView::section {{
        background-color: {C['bg_card']};
        color: {C['text_secondary']};
        border: none;
        border-bottom: 1px solid {C['border']};
        padding: 6px 8px;
        font-weight: 600;
        font-size: 12px;
    }}
    QTableCornerButton::section {{ background-color: {C['bg_card']}; border: none; }}
    QHeaderView {{ background-color: {C['bg_card']}; border: none; }}

    /* ── tabs ─────────────────────────────────────────────── */
    QTabWidget::pane {{ border: none; }}
    QTabBar::tab {{
        background: transparent;
        color: {C['text_secondary']};
        padding: 9px 18px;
        margin-right: 4px;
        border: none;
        border-bottom: 2px solid transparent;
        font-size: 13px;
        font-weight: 600;
    }}
    QTabBar::tab:hover {{ color: {C['text']}; }}
    QTabBar::tab:selected {{
        color: {C['accent']};
        border-bottom: 2px solid {C['accent']};
    }}

    /* ── menus / toolbars ─────────────────────────────────── */
    QMenu {{
        background-color: {C['bg_card']};
        color: {C['text']};
        border: 1px solid {C['border']};
        border-radius: 8px;
        padding: 6px;
    }}
    QMenu::item {{ padding: 7px 18px; border-radius: 4px; }}
    QMenu::item:selected {{ background-color: {C['accent_muted']}; color: {C['accent']}; }}
    QMenu::item:disabled {{ color: {C['text_dim']}; }}
    QMenu::separator {{ height: 1px; background: {C['border']}; margin: 4px 8px; }}
    QToolBar {{
        background: {C['bg_input']};
        border: 1px solid {C['border']};
        border-radius: 6px;
        spacing: 2px;
    }}
    QToolButton {{ background: transparent; border: none; border-radius: 4px; padding: 3px; }}
    QToolButton:hover {{ background: {C['bg_elevated']}; }}

    /* ── scroll area ──────────────────────────────────────── */
    QScrollArea {{
        border: none;
        background: transparent;
    }}
    QScrollArea > QWidget > QWidget {{
        background: transparent;
    }}
    QScrollArea#ImageCanvas {{
        background: {C['canvas']};
    }}
    QScrollBar:vertical {{
        background: transparent;
        width: 8px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {C['bg_elevated']};
        border-radius: 4px;
        min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {C['accent']};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 8px;
    }}
    QScrollBar::handle:horizontal {{
        background: {C['bg_elevated']};
        border-radius: 4px;
        min-width: 30px;
    }}
    QScrollBar::handle:horizontal:hover {{ background: {C['accent']}; }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

    /* ── group box ────────────────────────────────────────── */
    QGroupBox {{
        font-size: 12px;
        font-weight: 600;
        color: {C['text_secondary']};
        background-color: transparent;
        border: 1px solid {C['border']};
        border-radius: 8px;
        margin-top: 10px;
        padding: 14px 10px 10px 10px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 6px;
        color: {C['text_secondary']};
    }}

    /* ── collapsible section header ───────────────────────── */
    QToolButton#CollapseHeader {{
        background: transparent;
        border: none;
        color: {C['accent']};
        font-size: 14px;
        font-weight: 700;
        padding: 4px 0;
    }}

    /* ── tooltips ─────────────────────────────────────────── */
    QToolTip {{
        background-color: {C['bg_elevated']};
        color: {C['text']};
        border: 1px solid {C['border']};
        border-radius: 6px;
        padding: 6px 10px;
        font-size: 12px;
    }}

    /* ── message box ──────────────────────────────────────── */
    QMessageBox {{
        background-color: {C['bg_card']};
        color: {C['text']};
    }}
    QMessageBox QPushButton {{
        min-width: 80px;
        padding: 8px 16px;
    }}

    /* ── buttons ──────────────────────────────────────────── */
    QPushButton {{
        background-color: {C['bg_hover']};
        color: {C['text']};
        border: 1px solid {C['border']};
        border-radius: 6px;
        padding: 6px 14px;
        font-size: 13px;
        font-weight: 500;
        min-height: 30px;
    }}
    QPushButton:hover {{
        background-color: {C['bg_elevated']};
        border-color: {C['accent']};
    }}
    QPushButton:pressed {{
        background-color: {C['bg_input']};
    }}
    QPushButton:checked {{
        background-color: {C['accent_muted']};
        border-color: {C['accent']};
        color: {C['accent']};
    }}
    QPushButton:disabled {{
        background-color: {C['bg_base']};
        color: {C['text_dim']};
        border-color: {C['border_subtle']};
    }}
    QPushButton[variant="primary"] {{
        background-color: {C['accent']};
        color: {C['text_on_accent']};
        border: none;
        padding: 6px 20px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }}
    QPushButton[variant="primary"]:hover {{ background-color: {C['accent_hover']}; }}
    QPushButton[variant="primary"]:pressed {{ background-color: {C['accent_pressed']}; }}
    QPushButton[variant="primary"]:disabled {{
        background-color: {C['bg_elevated']};
        color: {C['text_dim']};
    }}
    QPushButton[variant="danger"] {{
        background-color: transparent;
        color: {C['error']};
        border: 1px solid {C['error']};
        padding: 6px 20px;
        font-weight: 600;
    }}
    QPushButton[variant="danger"]:hover {{ background-color: {C['error_bg']}; }}
    QPushButton[variant="danger"]:disabled {{
        border-color: {C['border']};
        color: {C['text_dim']};
    }}
    QPushButton[variant="success"] {{
        background-color: {C['success_bg']};
        color: {C['success']};
        border: 1px solid {C['success']};
        padding: 8px 16px;
        font-weight: 600;
    }}
    QPushButton[variant="success"]:hover {{ border-width: 2px; }}
    QPushButton[variant="solid-success"] {{
        background-color: {C['success']};
        color: #ffffff;
        border: none;
        font-weight: 600;
    }}
    QPushButton[variant="small"] {{
        padding: 4px 8px;
        font-size: 12px;
        min-height: 24px;
    }}
    QPushButton[channel="r"] {{ color: #e5484d; font-weight: 700; }}
    QPushButton[channel="g"] {{ color: #2f9e5b; font-weight: 700; }}
    QPushButton[channel="b"] {{ color: #3b7ddd; font-weight: 700; }}
    QPushButton[channel="rgb"] {{ font-weight: 700; }}
    QPushButton[channel]:disabled {{ color: {C['text_dim']}; }}
    """
