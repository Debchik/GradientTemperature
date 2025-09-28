import pygame.font
import pygame

class Button():
    
    def __init__(self, app, msg, position, button_size, command = lambda: print('no action for button'), **kwargs):
        """Инициализирует атрибуты кнопки."""
        self.screen = app.screen
        self.screen_rect = self.screen.get_rect()
        self.font = kwargs.get('font', 'corbel')
        # Назначение размеров и свойств кнопок.
        self.width, self.height = button_size
        self.command = command
        self.button_color = kwargs.get('button_color', (240, 240, 240))
        self.text_color = kwargs.get('text_color', (0, 0, 0))
        self.font = pygame.font.SysFont(self.font, kwargs.get('fontSize', 36), kwargs.get('bold', True))

        # Построение объекта rect кнопки и выравнивание по центру экрана.
        self.rect = pygame.Rect(*position, self.width, self.height)
        
        # Сообщение кнопки создается только один раз.
        self._prep_msg(msg)
    
    def _prep_msg(self, msg):
        self.msg_image = self.font.render(msg, True, self.text_color,
                                            self.button_color)
        self.msg_image_rect = self.msg_image.get_rect()
        self.msg_image_rect.center = self.rect.center
    
    def draw_button(self):
        # Отображение пустой кнопки и вывод сообщения.
        pygame.draw.rect(self.screen, self.button_color, self.rect)
        self.screen.blit(self.msg_image, self.msg_image_rect)