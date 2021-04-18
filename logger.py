from lib_imports import *

from config import config, unet_config, classifier_config
from utils.filesystem_helper import create_path, log_file_path_snapshot, logs_folder_path


class Logger:
    _log_file = None

    def __init__(self):
        create_path(logs_folder_path())
        log_file_path = log_file_path_snapshot()
        os.system('touch ' + log_file_path)
        self._log_file = open(log_file_path, 'w')
        self.log(str(dataclasses.asdict(config)))
        self.log(str(dataclasses.asdict(unet_config)))
        self.log(str(dataclasses.asdict(classifier_config)))

    def log(self, text, print_timestamp=False, flush=True):
        self._log_file.write(
            (time.strftime('%d_%H:%M:%S ', time.gmtime()) if print_timestamp else '')
            + text + '\n'
        )
        if flush:
            self._log_file.flush()


logger = Logger()
