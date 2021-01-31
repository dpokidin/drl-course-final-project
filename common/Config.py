# -*- coding: utf-8 -*-

class Config:
    "Stores configuration for the trainning"
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)