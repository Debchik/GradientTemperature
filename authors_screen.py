import pygame

import language
from button import Button

class AuthorsScreen():
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
        self.strings = [lang['university_name'],
                        lang['faculty_name'],
                        lang['lecturer'],
                        lang['supervisor'],
                        lang['author_valeev'],
                        lang['author_bogachev'],
                        lang['author_malakhov']]
        
        self.strings_surfaces = []
        for index, string in enumerate(self.strings):
            if index < 2:
                self.strings_surfaces.append(self.middle_font.render(string, False, (0, 0, 0)))
            else:
                self.strings_surfaces.append(self.little_font.render(string, False, (0, 0, 0)))
        
        #self.text_positions = [(400, 100), (500, 150), (670, 850), (600, 790), (395, 720), (1230, 720)]
        if lang.lang == 'rus':
            self.text_positions = [(self.app.monitor.width * 0.5 - 410, 100),
                                   (self.app.monitor.width * 0.5 - 540, 150)]
        elif lang.lang == 'eng':
            self.text_positions = [(self.app.monitor.width * 0.5 - 235, 100),
                            (self.app.monitor.width * 0.5 - 510, 150)]
        self.text_positions += [(self.app.monitor.width * 0.5 - 290, 850),
                                   (self.app.monitor.width * 0.5 - 340, 790),
                                   (self.app.monitor.width * 0.1 + 20 + 100, self.app.monitor.height * 0.2 + 100 + 400 + 10),
                                   (self.app.monitor.width * 0.9 - 20 - 340, self.app.monitor.height * 0.2 + 100 + 400 + 10),
                                   (self.app.monitor.width * 0.5 - 140, self.app.monitor.height * 0.2 + 100 + 400 + 10)]
        
        self.pictures = [pygame.transform.scale(pygame.image.load(self.folder + "cmc_logo.jpg"), (150, 150)),
                         pygame.transform.scale(pygame.image.load(self.folder + "physfac_logo.jpg"), (150, 150)),
                         pygame.transform.scale(pygame.image.load(self.folder + "Arslan_Valeev.jpg"), (400, 400)),
                         pygame.transform.scale(pygame.image.load(self.folder + "Vladimir_Bogachev.jpg"), (400, 400)),
                         pygame.transform.scale(pygame.image.load(self.folder + "Aleksander_Malakhov.jpg"), (400, 400))
                         ]
        
        self.pictures_positions = [(self.app.monitor.width * 0.9 - 150, 80),
                                   (self.app.monitor.width * 0.1, 80),
                                   (self.app.monitor.width * 0.1 + 30, self.app.monitor.height * 0.2 + 100),
                                   (self.app.monitor.width * 0.9 - 30 - 400, self.app.monitor.height * 0.2 + 100),
                                   (self.app.monitor.width * 0.5 - 200, self.app.monitor.height * 0.2 + 100)]
        self.buttons = [Button(app, lang['btn_menu'], (self.app.monitor.width * 0.9 - 30 - 300 , self.app.monitor.height - 80 - 60), (300, 80), self.to_menu)]

    def to_menu(self):
        self.app.active_screen = self.app.menu_screen

    def _update_screen(self):
        self.screen.fill(self.bg_color)
        for index, surface in enumerate(self.strings_surfaces):
            self.screen.blit(surface, self.text_positions[index])


        for index, picture in enumerate(self.pictures):
            self.screen.blit(picture, self.pictures_positions[index])

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
