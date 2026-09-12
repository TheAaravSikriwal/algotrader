"""Importing this package registers every built-in strategy."""
from core.strategy import REGISTRY, available, get_strategy  # noqa: F401

from . import advanced  # noqa: F401,E402
from . import builtin  # noqa: F401,E402
from . import classic  # noqa: F401,E402
from . import news  # noqa: F401,E402
from . import published  # noqa: F401,E402
