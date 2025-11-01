class ResponsiveScreen:
    """Keeps pygame Surface references in sync with the resizable window."""

    def __init__(self, app):
        self.app = app
        self.screen = app.screen

    def on_window_resize(self, size: tuple[int, int]) -> None:
        """Update cached Surface reference and notify subclasses to relayout."""
        self.screen = self.app.screen
        relayout = getattr(self, "_relayout", None)
        if callable(relayout):
            relayout(size)


def get_font(size: int, *, bold: bool = False) -> "pygame.font.Font":
    """Try to obtain a clean modern system font with sensible fallbacks."""
    import pygame

    families = [
        "SF Pro Display",
        "Helvetica Neue",
        "Avenir Next",
        "Segoe UI",
        "Roboto",
        "Arial",
        "sans-serif",
    ]
    return pygame.font.SysFont(families, size, bold=bold)


def build_vertical_gradient(
    size: tuple[int, int], top_color: tuple[int, int, int], bottom_color: tuple[int, int, int]
) -> "pygame.Surface":
    """Create a vertical gradient surface for backgrounds."""
    import pygame

    width, height = size
    surface = pygame.Surface((max(width, 1), max(height, 1)))
    if height <= 1:
        surface.fill(top_color)
        return surface
    for y in range(height):
        ratio = y / (height - 1)
        color = tuple(int(top_color[i] + (bottom_color[i] - top_color[i]) * ratio) for i in range(3))
        pygame.draw.line(surface, color, (0, y), (width, y))
    return surface
