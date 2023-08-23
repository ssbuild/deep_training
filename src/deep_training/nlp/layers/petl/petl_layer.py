# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/22 9:13

class PetlLayerAbstract:
    active_adapter = None

    def merge(self):
        raise NotImplementedError

    def unmerge(self):
        raise NotImplementedError