import re, os, jsonlines
from difflib import SequenceMatcher
import os
import logging, json
import pandas as pd

table_pattern = re.compile('[A-Za-z_]\w+|t')
alias_pattern = re.compile('([A-Z_]+alias\d)|'
                           '(T\d)|'
                           '(t\d)')
alias_id_pattern = re.compile('\d+')
alias_id_revtok_pattern = re.compile('\d+ ')
field_pattern = re.compile('([A-Z_]{1,100}alias\d+\.[A-Za-z_]\w+)|'
                           '([A-Za-z_]{1,100}\d+\.[A-Za-z_]\w+)|'
                           '([A-Za-z_]\w+\.[A-Za-z_]\w+)|'
                           '(T\d\.[A-Za-z_]\w+)|'
                           '(t\d\.[A-Za-z_]\w+)')
number_pattern = re.compile('\d+((\.\d+)|(,\d+))?')
time_pattern = re.compile('(\d{2}:\d{2}:\d{2})|(\d{2}:\d{2})')
datetime_pattern = re.compile('(\d{4})-(\d{2})-(\d{2})( (\d{2}):(\d{2}):(\d{2}))?')


DERIVED_TABLE_PREFIX = 'DERIVED_TABLE'
DERIVED_FIELD_PREFIX = 'DERIVED_FIELD'

class Logger:
    _log_directory = os.getcwd() + '/log'

    def __init__(self, file):
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
        self.logger.basicConfig(filename=self._log_directory + file,
                                format='%(asctime)s - %(name)s: %(levelname)s %(message)s')

    def error(self, msg):
        self.logger.error(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def info(self, msg):
        self.logger.info(msg)

def is_derived_table(s):
    return s.startswith(DERIVED_TABLE_PREFIX)


def is_derived_field(s):
    return s.startswith(DERIVED_FIELD_PREFIX)


def is_derived(s):
    return is_derived_table(s) or is_derived_field(s)


def is_subquery(json):
    is_subquery = False
    if isinstance(json, list):
        for part in json:
            if isinstance(part, dict):
                if 'from' or  'query'  or  'union' or  'intersect' or  'except' in json :
                    is_subquery = True
            else:
                is_subquery = False
        return is_subquery
    return 'from' in json or \
           'query' in json or \
           'union' in json or \
           'intersect' in json or \
           'except' in json 


def save2json(data, output_filename):
    if not os.path.exists(os.path.dirname(output_filename)):
        try:
            os.makedirs(os.path.dirname(output_filename))
            print("make zzy?")
        except OSError as exc:
            if exc.errno != errno.EEXITST:
                raise
    with open(output_filename, "w") as writer:
        json.dump(data, writer, indent=2)

    # with jsonlines.open(output_filename, 'w') as writer:
    #     for row in data:
    #         writer.write(row)

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
    return data


def read(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(line.strip('\n'))
    return data

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def save2graph(out_path, table_headers, table_records):   
   outfile = open(out_path, 'w')
   print("output:", outfile)
   while True:
      df = pd.DataFrame(table_records.fetchall())
      # Drop duplicate rows in place.
      df.drop_duplicates(inplace=True)
      if len(df) == 0:
         break
      else:
         print("record_header:", table_headers)
         print("check_records:", df)
         df.to_csv(outfile, header = table_headers, index = False)
         outfile.close() 

def check_compound_pk(primary_keys):
   compound_pk_check=False
   if len(primary_keys)!=1:  
      compound_pk_check=True
   return compound_pk_check




