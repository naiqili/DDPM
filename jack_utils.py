# -*- coding: utf-8 -*-
# Author: Jack Lee
# GitHub: http://github.com/still2009


import yaml
import json
from collections import Iterable
import hashlib
import copy

# named dict class
class NamedDict:
    """
    字典dict的包装类, 提供dict.key的属性访问方式
    可以嵌套NamedDict
    """

    def __init__(self, data=None, **kwargs):
        if data is not None:
            assert isinstance(data, dict)
            self.__data__ = data
        else:
            self.__data__ = kwargs

    def get_raw(self):
        return self.__data__

    def copy(self):
        return NamedDict(data=copy.deepcopy(self.__data__))

    def copy2update(self, **kwargs):
        raw = copy.deepcopy(self.__data__)
        raw.update(kwargs)
        return NamedDict(raw)

    def update(self, **kwargs):  # chained update function
        self.__data__.update(kwargs)
        return self

    @staticmethod
    def from_yaml(fname):
        data = None
        with open(fname, 'r') as f:
            data = yaml.load(f)
        if data is None:
            raise ValueError('Cannot be None')
        return NamedDict(data=data)

    def to_yaml(self, fname):
        with open(fname, 'w') as f:
            yaml.dump(self.__data__, f)
            print('yaml file dumped')

    @staticmethod
    def from_json(fname):
        data = None
        with open(fname, 'r') as f:
            data = json.load(f)
        if data is None:
            raise ValueError('Cannot be None')
        return NamedDict(data=data)

    def to_json(self, fname):
        with open(fname, 'w') as f:
            json.dump(self.__data__, f)
            print('json file dumped')

    def __getattr__(self, item):
        if item is '__data__':
            super(NamedDict, self).__getattribute__(item)
        elif item in self.__data__:
            result = self.__data__.get(item)
            if isinstance(result, dict) and not isinstance(result, NamedDict):
                return NamedDict(data=result)
            return result
        else:
            try:
                x = self.__data__.__getattribute__(item)
                return x
            except AttributeError as e:
                # print(e)
                raise AttributeError('NamedDict has no attribute %s' % item)

    def __setattr__(self, key, value):
        if key is '__data__':
            super(NamedDict, self).__setattr__(key, value)
        elif key in self.__data__:
            self.__data__[key] = value
        else:
            try:
                self.__data__.__setattr__(key, value)
            except AttributeError as e:
                # print(e)
                raise AttributeError('NamedDict has no attribute %s' % key)

    __getitem__ = __getattr__

    def __repr__(self):
        return str(self.__data__)

    __str__ = __repr__

    def __iter__(self):
        return self.__data__.__iter__()

    @staticmethod
    def sorted_flatten(value, key=None, it2str=False):
        result = {}
        # 递归基: value不再是一个dict
        if (not hasattr(value, 'keys')) and (not hasattr(value, 'values')):
            if isinstance(value, Iterable) and it2str and (not isinstance(value, str)):
                result[key] = '-'.join(str(x) for x in value)
            else:
                result[key] = value
            return result
        else:
            flatten_values = {}
            for k in sorted(value.keys()):
                key_prefix = '{}_{}'.format(key, k) if key is not None else k
                flatten_values.update(NamedDict.sorted_flatten(value[k], key_prefix, it2str))
            return flatten_values

    @staticmethod
    def flat_uid(obj):
        s = '\n'.join(['{}: {}'.format(k, obj[k]) for k in sorted(obj.keys())])
        return hashlib.md5(s.encode('utf8')).hexdigest()

    def sorted_str(self):
        flated = self.sorted_flatten(self.get_raw())
        return '\n'.join(['{}: {}'.format(k, flated[k]) for k in sorted(flated.keys())])

    def digest(self):
        return NamedDict.flat_uid(self.sorted_flatten(self.get_raw()))

    def equal(self, another):
        assert isinstance(another, NamedDict)
        return self.digest() == another.digest()

    def in_set(self, target_set):
        for obj in target_set:
            if self.equal(obj):
                return True
        return False

    @staticmethod
    def diff(one, other):
        a, b = NamedDict.sorted_flatten(one), NamedDict.sorted_flatten(other)
        info_a, info_b = {}, {}
        inter_keys = a.keys() & b.keys()
        for k in inter_keys:
            if a[k] != b[k]:
                info_a[k], info_b[k] = a[k], b[k]
        res_a = {k: a[k] for k in (a.keys() - inter_keys)}
        info_a.update(res_a)

        res_b = {k: b[k] for k in (b.keys() - inter_keys)}
        info_b.update(res_b)
        return info_a, info_b

    @staticmethod
    def inter(one, other):
        a, b = NamedDict.sorted_flatten(one), NamedDict.sorted_flatten(other)
        inter_keys = a.keys() & b.keys()
        return {k: a[k] for k in inter_keys}

    @staticmethod
    def set_inter(a, b):
        # 求两个NamedDict collection的交集，并按hash顺序返回list
        dict_a = {o.digest(): o for o in a}
        dict_b = {o.digest(): o for o in b}
        inter = dict_a.keys() & dict_b.keys()
        return [dict_a[k] for k in sorted(inter)]

    @staticmethod
    def set_sub(a, b):
        # 求两个NamedDict collection的差集，并按hash顺序返回list
        dict_a = {o.digest(): o for o in a}
        dict_b = {o.digest(): o for o in b}
        subtract = dict_a.keys() - dict_b.keys()
        return [dict_a[k] for k in sorted(subtract)]

    @staticmethod
    def set_unique(target):
        # 对NamedDict collection去重，并按hash顺序返回list
        dict_data = {o.digest(): o for o in target}
        return [dict_data[k] for k in sorted(dict_data.keys())]