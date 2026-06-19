import logging
import sys

_loggers = {}


def get_logger(name: str = "agent_q_mix", level: int = logging.INFO) -> logging.Logger:
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        fmt = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    _loggers[name] = logger
    return logger
