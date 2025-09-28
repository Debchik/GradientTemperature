import configloader
import py_singleton
import json


@py_singleton.singleton
class ConfigLoader(object):
    def __init__(self):
        self._loader = configloader.ConfigLoader()
        self._path = "config.json"
        self.update()

    def update(self):
        with open(self._path, "r", encoding="utf-8") as f:
            self._loader.update_from_json_file(f)

    def __getitem__(self, item):
        return self._loader[item]

    def set(self, key: str | tuple, value):
        """
        Change record with key in config.
        It's not implemented by __setitem__ for config safety
        :param key: If key is str then changing cfg[key].
        If key is tuple (key_1, ..., key_n) then changing cfg[key_1][...][key_n]
        :param value: New value
        """

        if isinstance(key, str):
            if key not in self._loader:
                raise ValueError('key not in config keys')
            self._loader[key] = value
        else:
            to_update = self._loader
            for k in key[:-1]:
                if k not in to_update:
                    raise ValueError('key not in config keys')
                to_update = to_update[k]

            if key[-1] not in to_update:
                raise ValueError('key not in config keys')
            to_update[key[-1]] = value

        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(obj=dict(self._loader), fp=f, ensure_ascii=False, indent=2, separators=(',', ': '))

