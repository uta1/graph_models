from config import *
from lib_imports import *
from utils import *


class Logger:
    _log_file = None

    def __init__(self):
        create_path(config.LOGS_FOLDER_PATH)
        log_file_path = config.log_file_path_snapshot()
        os.system('touch ' + log_file_path)
        self._log_file = open(log_file_path, 'w')

    def log(self, text, print_timestamp=False, flush=True):
        self._log_file.write(
            (time.strftime('%d_%H:%M:%S ', time.gmtime()) if print_timestamp else '')
            + text + '\n'
        )
        if flush:
            self._log_file.flush()


logger = Logger()
