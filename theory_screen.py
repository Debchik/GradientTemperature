import pygame

import language
from button import Button


class TheoryScreen():
    def __init__(self, app):
        lang = language.Language()
        self.app = app
        self.screen = app.screen
        self.folder = '_internal/images/'
        self.bg_color = (255, 255, 255)
        self.font = 'sans'
        self.little_font = pygame.font.SysFont(self.font, 38)
        self.middle_font = pygame.font.SysFont(self.font, 40, bold=True)
        self.big_font = pygame.font.SysFont(self.font, 50)
        self.page = 0
        self.pictures = []
        self.pictures_positions = []
        for i in range(1, 4):
            self.pictures.append(pygame.transform.scale(pygame.image.load(self.folder + f"theory_page_{i}.png"), (1290, 1010)))
            self.pictures_positions.append(((self.app.monitor.width - 1290) * 0.5, 0))

        self.buttons = [Button(app, "<—", (30, self.app.monitor.height - 80 - 60 ), (100, 80), self.last_page),
                        Button(app, lang['btn_menu'], (self.app.monitor.width * 0.5 - 150, self.app.monitor.height - 80 - 60 ), (300, 80), self.to_menu),
                        Button(app, "—>", (self.app.monitor.width - 100 - 30, self.app.monitor.height - 80 - 60 ), (100, 80), self.next_page)]

    def to_menu(self):
        self.app.active_screen = self.app.menu_screen

    def last_page(self):
        self.page = max(0, self.page - 1)

    def next_page(self):
        self.page = min(len(self.pictures) - 1, self.page + 1)

    def _update_screen(self):
        self.screen.fill(self.bg_color)
#        for index, surface in enumerate(self.strings_surfaces):
#            self.screen.blit(surface, self.text_positions[index])

        self.screen.blit(self.pictures[self.page], self.pictures_positions[self.page])

        for button in self.buttons:
            button.draw_button()

    def _check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_position = pygame.mouse.get_pos()
                self._check_buttons(mouse_position)

    def _check_buttons(self, mouse_position):
        for button in self.buttons:
            if button.rect.collidepoint(mouse_position):
                button.command()

