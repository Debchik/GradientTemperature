import sys

import pygame

import config
import language
from button import Button


class MenuScreen:
    def __init__(self, app):
        lang = language.Language()
        self.app = app
        self.screen = app.screen
        self.folder = '_internal/images/'
        self.bg_color = (255, 255, 255)
        self.font = 'sans'
        self.little_font = pygame.font.SysFont(self.font, 35)
        self.middle_font = pygame.font.SysFont(self.font, 40, bold=True)
        self.big_font = pygame.font.SysFont(self.font, 50)
        self.msu_name = lang['university_name']
        self.faculty_name = lang['faculty_name']
        self.demonstration_label = lang['comp_demo']
        self.subject_name = lang['subject_name']
        self.demonstration_name = lang['job_title']
        self.demonstration_name_2 = lang['job_title2']
        self.strings = [self.msu_name, self.faculty_name, self.demonstration_label, self.subject_name,
                        self.demonstration_name, self.demonstration_name_2]
        self.strings_surfaces = []
        for index, string in enumerate(self.strings):
            if index < 2:
                self.strings_surfaces.append(self.middle_font.render(string, False, (0, 0, 0)))
            elif index < 4:
                self.strings_surfaces.append(self.little_font.render(string, False, (0, 0, 0)))
            else:
                self.strings_surfaces.append(self.big_font.render(string, True, (50, 50, 50)))

        if lang.lang == 'rus':
            self.positions = [(self.app.monitor.width * 0.5 - 410, 100),
                            (self.app.monitor.width * 0.5 - 540, 150),
                            (self.app.monitor.width * 0.5 - 270, 250),
                            (self.app.monitor.width * 0.5 - 170, 300),
                            (self.app.monitor.width * 0.5 - 375, 400),
                            (self.app.monitor.width * 0.5 - 240, 470)]
        elif lang.lang == 'eng':
            self.positions = [(self.app.monitor.width * 0.5 - 235, 100),
                            (self.app.monitor.width * 0.5 - 510, 150),
                            (self.app.monitor.width * 0.5 - 155, 250),
                            (self.app.monitor.width * 0.5 - 200, 300),
                            (self.app.monitor.width * 0.5 - 305, 400),
                            (self.app.monitor.width * 0.5 - 215, 470)]
        self.cmc_logo = pygame.transform.scale(pygame.image.load(self.folder + "cmc_logo.jpg"), (150, 150))
        self.physfac_logo = pygame.transform.scale(pygame.image.load(self.folder + "physfac_logo.jpg"), (150, 150))
        self.buttons = [Button(app, lang['btn_demo'], (app.monitor.width // 2 - 200, 600), (400, 80), self.to_demo),
                        Button(app, lang['btn_theory'], (app.monitor.width // 2 - 200, 700), (400, 80), self.to_theory),
                        Button(app, lang['btn_authors'], (app.monitor.width // 2 - 200, 800), (400, 80), self.to_authors),
                        Button(app, lang['btn_exit'], (app.monitor.width // 2 - 200, 900), (400, 80), self.quite_demo),
                        Button(app, lang['btn_lang'], (app.monitor.width - 40 - 80, app.monitor.height - 70 - 80), (80, 80), self.lang_change)]

    def to_demo(self):
        self.app.active_screen = self.app.demo_screen

    def to_theory(self):
        self.app.active_screen = self.app.theory_screen

    def to_authors(self):
        self.app.active_screen = self.app.authors_screen

    def quite_demo(self):
        pygame.quit()
        sys.exit()

    def lang_change(self):
        cfg = config.ConfigLoader()
        if cfg['language'] == "rus":
            cfg.set("language", "eng")
        elif cfg['language'] == "eng":
            cfg.set("language", "rus")
        lang = language.Language()
        lang.reload()
        self.app.__init__()

    def _update_screen(self):
        self.screen.fill(self.bg_color)
        for index, surface in enumerate(self.strings_surfaces):
            self.screen.blit(surface, self.positions[index])
        self.screen.blit(self.cmc_logo, (self.app.monitor.width * 0.9 - 150, 80))
        self.screen.blit(self.physfac_logo, (self.app.monitor.width * 0.1, 80))
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
