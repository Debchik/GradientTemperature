import pygame
import screeninfo

import config
import language
from authors_screen import AuthorsScreen
from demo_screen import DemoScreen
from menu_screen import MenuScreen
from theory_screen import TheoryScreen


class App:
    MIN_WINDOW = (1280, 720)
    WINDOW_SCALE = 0.86

    def __init__(self):
        pygame.init()
        monitor = screeninfo.get_monitors()[0]
        monitor.height -= 1
        self.monitor = monitor
        self.window_size = self._get_initial_window_size(monitor)
        self.screen = pygame.display.set_mode(self.window_size, pygame.RESIZABLE)

        self.clock = pygame.time.Clock()

        self.menu_screen = MenuScreen(self)
        self.authors_screen = AuthorsScreen(self)
        self.theory_screen = TheoryScreen(self)
        self.demo_screen = DemoScreen(self)

        self._screens = (
            self.menu_screen,
            self.authors_screen,
            self.theory_screen,
            self.demo_screen,
        )
        self._notify_resize(self.window_size)

        self.active_screen = self.menu_screen

        self._config = config.ConfigLoader()

    # ------------------------------------------------------------------ Locale
    def set_language(self, language_code: str) -> None:
        """Persist the selected language and notify screens to refresh text."""
        cfg = config.ConfigLoader()
        available = cfg['language_files']
        if language_code not in available:
            raise ValueError(f"Unknown language code: {language_code!r}")

        current = cfg['language']
        if current == language_code:
            return

        cfg.set('language', language_code)
        language.Language().reload()

        for screen in self._screens:
            handler = getattr(screen, "on_language_change", None)
            if callable(handler):
                handler()
        self._notify_resize(self.window_size)

    def toggle_language(self) -> None:
        cfg = config.ConfigLoader()
        current = cfg['language']
        available = list(cfg['language_files'])
        if len(available) < 2:
            return
        current_index = available.index(current)
        next_lang = available[(current_index + 1) % len(available)]
        self.set_language(next_lang)

    def _get_initial_window_size(self, monitor) -> tuple[int, int]:
        width = int(monitor.width * self.WINDOW_SCALE)
        height = int(monitor.height * self.WINDOW_SCALE)
        min_w, min_h = self.MIN_WINDOW
        return max(min_w, width), max(min_h, height)

    def _notify_resize(self, size: tuple[int, int]) -> None:
        for screen in self._screens:
            if hasattr(screen, "on_window_resize"):
                screen.on_window_resize(size)

    def handle_resize(self, size: tuple[int, int]) -> None:
        min_w, min_h = self.MIN_WINDOW
        width = max(min_w, size[0])
        height = max(min_h, size[1])
        resized = (width, height)
        if resized == self.window_size:
            return
        self.window_size = resized
        self.screen = pygame.display.set_mode(self.window_size, pygame.RESIZABLE)
        self._notify_resize(self.window_size)

    def run(self):
        """Запуск основного цикла игры."""
        while True:
            # Mouse and keyboard events handling
            self.active_screen._check_events()
            self.active_screen._update_screen()
            # Отображение последнего прорисованного экрана.
            pygame.display.flip()

            fps = self._config['FPS']
            self.clock.tick(fps)


if __name__ == '__main__':
    app = App()
    app.run()
