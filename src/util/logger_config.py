import logging
import sys

def set_log_file(log_file, level=logging.INFO):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def flush_all_loggers():
    for handler in logging.root.handlers:
        handler.flush()
