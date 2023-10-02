import os
import logging


class Logger:
    _log_directory = os.getcwd() + '/log'

    def __init__(self):
        # ensure the correct log directory
        if not os.path.isdir(self._log_directory):
            os.mkdir(self._log_directory)

        self.logger = logging
        # self.logger = logging.getLogger(__name__)
        # f_handler = logging.FileHandler(self._log_directory + '/sql2cypher.log')
        # f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # f_handler.setFormatter(f_format)
        #
        # self.logger.addHandler(f_handler)
        self.logger.basicConfig(filename=self._log_directory + '/sql2cypher.log',
                                format='%(asctime)s - %(name)s: %(levelname)s %(message)s')

    def error(self, msg):
        self.logger.error(msg)

    def warning(self, msg):
        self.logger.warning(msg)