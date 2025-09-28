"""
Helper function for create directories
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os


def create_dirs(dir_):
    """
    Utility function for creating directories

    Args:
        dirs (list of string): A list of directories to create if these
             directories are not found.
    """
    logger = logging.getLogger('Create Directory')
    try:
        os.makedirs(dir_)
    except FileExistsError:
        logger.warning('Directory already exist: %s', dir_)
