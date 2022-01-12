import datetime
import sys
from logging import INFO, StreamHandler, basicConfig, getLogger

from pythonjsonlogger import jsonlogger


# https://github.com/madzak/python-json-logger#customizing-fields
class JsonLogFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):  # type: ignore
        # Cloud Logging uses `time` and `severity` fields
        log_record["time"] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f%z")
        log_record["severity"] = record.levelname

        super().add_fields(log_record, record, message_dict)


def get_logger(name: str, formatter=None):  # type: ignore
    basicConfig(level=INFO)
    logger = getLogger(name)
    logger.setLevel(INFO)

    # to remove duplicated logs for GKE
    # c.f., https://www.ai-shift.co.jp/techblog/1217
    # c.f., https://shunyaueta.com/posts/2021-03-03/
    handler = StreamHandler(stream=sys.stdout)
    logger.propagate = False
    handler.setFormatter(formatter or JsonLogFormatter())
    logger.addHandler(handler)
    return logger
