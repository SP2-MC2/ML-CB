"""
__version__ = '1.0'
__author__ = 'Nathan Reitinger'

simple wrapper around ml-cb.py which allows piecemeal testing in runs
"""


import os

if __name__ == "__main__":

    # number of runs for model assessment
    # set to 5 to replicate runs in the paper (using SKF cross validation)
    # CNN used 10 (using holdout cross validation)
    LOOP_COUNT = 1

    for i in range(LOOP_COUNT):
        os.system("python3 ml-cb.py --model embedding --corpus jsnice")
