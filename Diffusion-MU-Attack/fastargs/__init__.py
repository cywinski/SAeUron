"""
fastargs
--------

library for configuration/parameters management

"""
from .section import Section
from .param import Param
from .config import Config
from .state import get_current_config, set_current_config

from . import validation
from . import decorators

__version__ = "1.1.1"
