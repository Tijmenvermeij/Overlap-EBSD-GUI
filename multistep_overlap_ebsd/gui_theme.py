"""Shared Tk appearance, independent of the operating system's native theme."""
from tkinter import font, ttk


BACKGROUND = "#f3f5f8"
TEXT = "#243247"
MUTED = "#526176"
BORDER = "#c5ceda"
ACCENT = "#245a91"


def configure_theme(root):
    # Native Windows/macOS tab elements can ignore background and border colors.
    # Clam provides the same configurable elements on both platforms while using
    # Tk's platform font and DPI scaling.
    style = ttk.Style(root)
    style.theme_use("clam")
    heading = font.nametofont("TkDefaultFont", root=root).copy()
    heading.configure(weight="bold")
    root._heading_font = heading  # Keep the named Tk font alive.
    root.configure(background=BACKGROUND)
    style.configure(".", background=BACKGROUND, foreground=TEXT,
                    bordercolor=BORDER, lightcolor=BACKGROUND, darkcolor=BORDER)
    style.configure("TButton", padding=(8, 5), background="#ffffff", relief="raised")
    style.map("TButton", background=[("disabled", "#edf0f4"),
                                     ("pressed", "#d5e4f3"), ("active", "#e8f0f9")],
              foreground=[("disabled", "#788393")])
    style.configure("TEntry", padding=4, fieldbackground="#ffffff")
    style.configure("TCombobox", padding=4, fieldbackground="#ffffff")
    style.configure("TSpinbox", padding=4, fieldbackground="#ffffff")
    for widget in ("TEntry", "TCombobox", "TSpinbox"):
        style.map(widget, fieldbackground=[("disabled", "#e9edf2"),
                                          ("readonly", "#edf2f7")],
                  foreground=[("disabled", "#788393")],
                  bordercolor=[("focus", ACCENT)])
    style.configure("TLabelframe", borderwidth=1, relief="solid")
    style.configure("TLabelframe.Label", font=heading, foreground=ACCENT)
    style.configure("Hint.TLabel", foreground=MUTED)
    style.configure("Status.TLabel", background="#e5ebf2", foreground=MUTED,
                    padding=(10, 6))
    style.configure("Disclosure.TButton", anchor="w", padding=(8, 5),
                    background="#e5ebf2")
    style.configure("Horizontal.TProgressbar", background=ACCENT,
                    troughcolor="#e1e7ef", borderwidth=0)
    style.configure("Workflow.TNotebook", background=BACKGROUND,
                    borderwidth=1, tabmargins=(0, 5, 0, 0))
    style.configure("Workflow.TNotebook.Tab", font=heading, padding=(12, 9),
                    background="#e0e6ee", foreground=TEXT, borderwidth=1)
    style.map("Workflow.TNotebook.Tab",
              background=[("disabled", "#edf0f4"), ("selected", ACCENT),
                          ("active", "#cbdced")],
              foreground=[("disabled", "#788393"), ("selected", "#ffffff")],
              lightcolor=[("selected", ACCENT)],
              darkcolor=[("selected", ACCENT)],
              bordercolor=[("selected", ACCENT)])
    return style
