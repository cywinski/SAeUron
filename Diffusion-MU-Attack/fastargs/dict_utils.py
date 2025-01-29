from collections import defaultdict

from types import SimpleNamespace


class NestedNamespace(SimpleNamespace):
    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.__setattr__(key, NestedNamespace(value))
            else:
                self.__setattr__(key, value)

def rec_dd():
    return defaultdict(rec_dd)

def recursive_get(dic, path):
    if len(path) == 0:
        return dic
    head = path[0]
    tail = path[1:]
    return recursive_get(dic[head], tail)

def recursive_set(dic, path, value):
    if len(path) == 1:
        dic[path[0]] = value
    else:
        head = path[0]
        tail = path[1:]

        recursive_set(dic[head], tail, value)


def expand_keys(dic, prefix=tuple(), result=None):
    if result is None:
        result = rec_dd()

    if not isinstance(dic, dict):
        recursive_set(result, prefix, dic)
    else:
        for k, v in dic.items():
            path = tuple(k.split('.'))
            full_path = prefix + path
            expand_keys(v, full_path, result)
    return result

def fix_dict(defdict):
    if isinstance(defdict, defaultdict):
        for k in list(defdict.keys()):
            defdict[k] = fix_dict(defdict[k])
        return dict(defdict)
    return defdict
