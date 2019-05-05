# -*- coding: utf-8 -*-

import json


class Map(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def dumps(self, indent=2, sort_keys=False) -> str:
        return json.dumps(self, indent=indent, sort_keys=sort_keys)


def get_mapper() -> Map:
    return Map()
