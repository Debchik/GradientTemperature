import configloader
from py_singleton import singleton
from config import ConfigLoader


@singleton
class Language(object):
    def __init__(self):
        cfg = ConfigLoader()
        self.lang = cfg['language']
        self.lang_path = cfg['language_files'][self.lang]
        self._loader = configloader.ConfigLoader()
        self.update()

    def update(self):
        with open(self.lang_path, "r", encoding="utf-8") as f:
            self._loader.update_from_json_file(f)

    def reload(self):
        cfg = ConfigLoader()
        self.lang = cfg['language']
        self.lang_path = cfg['language_files'][self.lang]
        self.update()

    def __getitem__(self, item):
        return self._loader[item]
