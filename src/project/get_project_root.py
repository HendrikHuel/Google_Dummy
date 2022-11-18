#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 09:45:28 2022

@author: huelsbusch
"""
from pathlib import Path


def get_project_root():
    return Path(__file__).parents[2]


if __name__ == "__main__":
    print(get_project_root())
