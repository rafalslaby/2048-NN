import logging
from pathlib import Path


def move_file_handlers(logger, new_path):
    for old_handler in logger.handlers[:]:
        if isinstance(old_handler, logging.FileHandler):
            file_path = Path(new_path) / Path(old_handler.baseFilename).name
            new_fh = logging.FileHandler(str(file_path))
            new_fh.setLevel(old_handler.level)
            new_fh.setFormatter(old_handler.formatter)
            logger.removeHandler(old_handler)
            logger.addHandler(new_fh)


if __name__ == '__main__':
    logger = logging.getLogger("abc")
    fh = logging.FileHandler('abc.log')
    formatter = logging.Formatter('%(message)s')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    logger.info("stary")
    move_file_handlers(logger,'folder')
    logger.info("nowy")
