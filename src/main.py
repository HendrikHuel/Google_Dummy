#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 09:37:52 2022

@author: huelsbusch
"""

import argparse

from project import test_skript as ts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ek", help="The EK group.", type=int, default=1)
    parser.add_argument(
        "--do_not_preprocess", help="Do not run preprocessing.", action="store_false",
    )
    parser.add_argument(
        "--do_not_engineer",
        help="Do not run the feature engineering.",
        action="store_false",
    )

    parser.add_argument(
        "--do_not_train", help="Do not train models.", action="store_false"
    )

    args = parser.parse_args()

    ts.preprocess_and_train(
        ek=args.ek,
        preprocess=args.do_not_preprocess,
        engineer=args.do_not_engineer,
        train=args.do_not_train,
    )
