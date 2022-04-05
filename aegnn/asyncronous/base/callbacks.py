import logging
from typing import Any


class CallbackFactory:

    def __init__(self, listeners, log_name: str):
        self.listeners = listeners
        self.log_name = log_name
        logging.debug(f"Setting callback for module {log_name} with {len(listeners)} listeners")

    def __call__(self, key: str, value: Any):
        for listener in self.listeners:
            logging.debug(f"Setting attribute {key} of module {listener} from module {self.log_name}")
            setattr(listener, key, value)
