"""
Global configuration variables.
"""
import os

ROOT_DIR = os.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.sep)[:-1])

TEST_DIR = os.path.join(ROOT_DIR, 'tests')

TEST_DATA_DIR = os.path.join(TEST_DIR, 'validation_data')
