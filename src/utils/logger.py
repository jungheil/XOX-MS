import logging
import os

from utils.tools import SingletonMeta


class LoggerMaster(metaclass=SingletonMeta):
    def __init__(self) -> None:
        self._loggers = {}

    def __call__(self, name, *args, **kwds):
        if name not in self._loggers:
            self._loggers[name] = Logger(name, *args, **kwds)
        return self._loggers[name]


class Logger:
    def __init__(self, name, log_level=logging.INFO, log_file_path=None) -> None:
        self.name = name
        self.logger = logging.getLogger(name)
        self.log_file_path = log_file_path

        format_str = "%(asctime)s %(levelname)s: %(message)s"
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(format_str))
        self.logger.addHandler(stream_handler)
        self.log_file_path = log_file_path
        if log_file_path is not None:
            file_handler = logging.FileHandler(
                os.path.join(log_file_path, name + ".log"), "w"
            )
            file_handler.setFormatter(logging.Formatter(format_str))
            self.logger.addHandler(file_handler)
        self.logger.setLevel(log_level)
        self.logger.propagate = False

    def debug(self, *args, **kwds):
        return self.logger.debug(*args, **kwds)

    def info(self, *args, **kwds):
        return self.logger.info(*args, **kwds)

    def warning(self, *args, **kwds):
        return self.logger.warning(*args, **kwds)

    def error(self, *args, **kwds):
        return self.logger.error(*args, **kwds)

    def critical(self, *args, **kwds):
        return self.logger.critical(*args, **kwds)

    def training_log(self):
        pass


LM = LoggerMaster()
