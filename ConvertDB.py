"""
This section code working on convert the whole database to cypher
"""
import os
from re import error
import sqlite3
import sys
import time
import pickle
from mo_parsing.helpers import Log
import psycopg2
import pandas as pd
import mysql.connector
from mysql.connector import errorcode
from neo4j import GraphDatabase, graph
from psycopg2 import Error, OperationalError, errorcodes, errors
import argparse, glob
import configparser
config = configparser.ConfigParser()


import sqlite3

class ConvertDB:
    _neo4j_export_path = '/home/22842219/Desktop/phd/neo4j-community-4.4.13/import'
    _cache_path = os.getcwd() + '/cache/'
    _output_path = os.getcwd() + '/data'

    def __init__(self, mysql_config, sqlite3_config, neo4j_config, psql_config, db_name, logger, output):
        # self.__neo4j_export_path = None
        self.logger = logger
        # if only output then just output the result
        self.output = output
        self.db_name = db_name
        self.delete_files = []
        self.mysql_config = mysql_config
        self.sqlite3_config = sqlite3_config
        self.neo4j_config = neo4j_config
        self.psql_config = psql_config
  
        # to make sure the output directory is correct
        self._ensure_directory(self._output_path)
        self._ensure_directory(self._cache_path)

    def _ensure_directory(self, path):
        """
        to make sure all the directories are valid
        :return: nothing
        """
        if not os.path.isdir(path):
            self.logger.warning("Create directory: {}".format(path))
            os.mkdir(path)

    def execute_cypher(self, query):
        """
        set up the cypher server
        db connect:
            db.set_connection('bolt://{}:{}@{}:7474'.format(self.cypher_user, self.cypher_password, self.cypher_ip))
        :return:
        """
        try:
            # ZZY: rewrite driver for 4.4.10 version.
            uri = "neo4j://localhost:7687"
            driver = GraphDatabase.driver(uri, auth=(self.neo4j_config['username'], self.neo4j_config['password']))

            # Driver for neo4j 3.x
            # driver = GraphDatabase.driver("bolt://{}:{}".format(self.neo4j_config['host'], self.neo4j_config['port']),
                                        #   auth=(self.neo4j_config['username'], self.neo4j_config['password']))
            session = driver.session()
            res = session.run(query)

            # Note: res, which refers to neo4j.Result, provides a handle to the result of the query, giving access to the records within it as well as the result metadata.
            # Ref: https://neo4j.com/docs/api/python-driver/current/api.html

            # Helper function that return the remainder of the result as a list of dictionaries.
            data = res.data()
            print("cypher data:", data)
            driver.close()
            return data
        except Exception as error:
            print("Can not connect the neo4j, please check the services and config")
            self.logger.error("Can not connect the neo4j, please check the services and config")
            raise IOError("Something error")

    def _extract_sql_result(self, cursor, query, args=()):
        """
        due to mysql and  psql have same steps when executing the query then put one function
        :param query:
        :param args:
        :return:
        """
        
        cursor.execute(query, args)
        res = [dict((cursor.description[idx][0], value)
                    for idx, value in enumerate(row)) for row in cursor.fetchall()]

        return res

    def execute_psql(self, query, args=()):
        """
        execute psql
        :param query: psql query
        :param args: args
        :return: tuples
        """
        try:
            conn = psycopg2.connect(**self.psql_config)
        except OperationalError as err:
            print("psql error: ", err)
            self.logger.error("psql error")
            self.logger.error("psql extensions.Diagnostics: " + err.diag)
            raise ValueError("Check the config")

        res = self._extract_sql_result(conn.cursor(), query, args)

        conn.commit()
        conn.close()
        return res

    def execute_mysql(self, query, args=()):
        """
        execute the sql query language
        :param query: sql query language
        :param args: args in sql
        :return: all the values get from db
        """
        try:
            mydb = mysql.connector.connect(
                **self.mysql_config
            )
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                self.logger.error("mysql Something is wrong with your user name or password!")
                raise ValueError("Something is wrong with your user name or password!")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                self.logger.error("mysql Database dose not exist!")
                raise IOError("Database dose not exist!")
            else:
                raise ValueError("err")

        res = self._extract_sql_result(mydb.cursor(), query, args)

        mydb.commit()
        mydb.close()
        return res

    def connect_sqlite3(self):
        """
        """
        
        db_path = os.path.join(self.sqlite3_config['spider_path'], self.sqlite3_config['database'])
        db_paths =glob.glob(db_path + '/*.sqlite', recursive = True) 
        print(db_paths)
        if len(db_paths)==1:
            db_path = db_paths[0]
            conn = sqlite3.connect(db_path)
            print(db_path)

        else:
            self.logger.error("There are more than one sqlite files in {}.db!".format(self.sqlite3_config['database']))
            raise ValueError("Something is wrong with your user name or password!")
       
        
        return conn

    def execute_sqlite3(self, query):
        """
        """
        try:
            # cursor = conn.cursor()  
            # res = cursor.execute(query)
            conn = self.connect_sqlite3()
            res = self._extract_sql_result(conn.cursor(), query, args=())
            print("heyyy res:", res)
        except sqlite3.ProgrammingError as e:
            print(e)
        
        return res

    def _load_pickle(self):
        """
        load pickle files
        :return: pickle info
        """
        filepath = self._cache_path + "/relation.pickle"
        try:
            files = open(filepath, "rb")
            data = pickle.load(files)
            if type(data) is list:
                return data
        except FileNotFoundError:
            self.logger.warning("relationship cache does not exist")
            return None
        return None

    def _export_file(self, filename, data):
        """
        export the cypher data to files
        :param filename: output file name
        :param data: output data
        :return: nothing
        """
        with open(self._output_path + "/{}.cypher".format(filename), 'a+') as f:
            for query in data:
                f.write(query + "\n")

    def get_tables(self):
        """
        return all the tables in the db
        :return: all the tables name
        """
        # get all the tables as a node
        tables = self.execute_mysql("SELECT table_name as id FROM "
                                   "information_schema.tables "
                                   "where table_schema='{}';".format(self.mysql_config['database']))
        return tables
    

    def get_mysql_relations(self, only_table=False):
        """
        get all the relationship between tables
        :return: array of relations
        """
        if not only_table:
            query = "SELECT `TABLE_NAME`, `COLUMN_NAME`, `REFERENCED_TABLE_NAME`, `REFERENCED_COLUMN_NAME`"
        else:
            query = "SELECT `TABLE_NAME`,`REFERENCED_TABLE_NAME`"
        query += "FROM `INFORMATION_SCHEMA`.`KEY_COLUMN_USAGE` WHERE `TABLE_SCHEMA` = SCHEMA() " \
                 "AND `REFERENCED_TABLE_NAME` IS NOT NULL;"
        relation_tables = self.execute_mysql(query)
        # print(relation_tables)

        return relation_tables

    def get_psql_relations(self, only_table=False):
        """
        get the table relationship
        :param only_table:
        :return:
        """
        query = """
        SELECT
            tc.table_name as "TABLE_NAME", 
            kcu.column_name as "COLUMN_NAME", 
            ccu.table_name AS "REFERENCED_TABLE_NAME",
            ccu.column_name AS "REFERENCED_COLUMN_NAME" 
        FROM 
            information_schema.table_constraints AS tc 
            JOIN information_schema.key_column_usage AS kcu
              ON tc.constraint_name = kcu.constraint_name
              AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
              ON ccu.constraint_name = tc.constraint_name
              AND ccu.table_schema = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY';
        """
        return self.execute_psql(query)

    def read_relations(self):
        """
        get the tables relation by user typing,
        the cache relation like: {
            'db name': {
                'src': {
                    'on': 'xx',
                    'dst': 'xx',
                    'dst_on': 'xxx',
                    'label': 'xxx'
                }
            }
        }
        :return: nothing
        """
        # get all the tables
        filepath = self._cache_path + "/relation.pickle"
        database = self.psql_config['database'] if self.db_name == 'psql' else self.mysql_config['database']

        # execute different sql query for select tables
        all_table = self.execute_mysql("SHOW TABLES;") if self.db_name != 'psql' else \
            self.execute_psql("SELECT table_name as \"Tables_in_{}\" FROM information_schema.tables "
                                           "WHERE table_schema = 'public';".format(self.psql_config['database']))
        tables = []
        for t in all_table:
            tables.append(t['Tables_in_{}'.format(database)])

        # all the relations which stored in pickle
        data = self._load_pickle()
        # set the relation dict is empty if no data
        relation = {} if data is None else data

        # for this time which need to be export
        relationship = {}
        visited_tables = set()

        # read the relationship between tables
        relation_tables = self.get_mysql_relations() if self.db_name != 'psql' else self.get_psql_relations()
        print(relation_tables)

        for rt in relation_tables:
            label = input(
                "Please enter the relation between {}->{}: ".format(rt['REFERENCED_TABLE_NAME'], rt['TABLE_NAME']))
            label = "{}_{}".format(rt['REFERENCED_TABLE_NAME'], rt['TABLE_NAME']) if label == "" else label

            if rt['REFERENCED_TABLE_NAME'] not in relationship:
                relationship[rt['REFERENCED_TABLE_NAME']] = []
            # it cloud have multiple relationship then use an array to store that
            relationship[rt['REFERENCED_TABLE_NAME']].append({
                'src_key': rt['COLUMN_NAME'],
                'dst': rt['TABLE_NAME'],
                'dst_key': rt['REFERENCED_COLUMN_NAME'],
                'label': label
            })

            visited_tables.add(rt['REFERENCED_TABLE_NAME'])
            visited_tables.add(rt['TABLE_NAME'])

        # add the single table
        for table in tables:
            if table not in visited_tables:
                relationship[table] = None

        # now try to solve the relation to pickle
        # print(relation)
        relation[database] = relationship
        files = open(filepath, "wb")
        pickle.dump(relation, files)

        return relationship

    def _load_with_csv(self, table_name, data):
        """
        load the data with csv model if the count less than 100000
        otherwise load with cypher model
        :return:
        """
        # export the csv file into neo4j export path
        cols = data[0].keys()
        df = pd.DataFrame(data, columns=cols)
        filepath = self._neo4j_export_path + 'spider/musical/{}.csv'.format(table_name)
        df.to_csv(filepath, index=False)

        query = "LOAD CSV WITH HEADERS FROM 'file:///spider/musical/{}.csv' AS row ".format(table_name)
        table_schema = self.execute_mysql("show columns from %s;" % table_name) if self.db_name != 'psql' else \
            self.execute_psql("select column_name as \"Field\" "
                               "from information_schema.columns where table_name = {}".format(table_name))

        query += " MERGE ({}:{} ".format(str(table_name).lower(), table_name)
        query += "{"
        query += ", ".join("{}: coalesce(row.{}, \"Unknown\")".format(name['Field'], name['Field'])
                           for index, name in enumerate(table_schema))
        query += "}); "
        self.delete_files.append(filepath)

        return query

    def _load_with_cypher(self, table_name, data):
        """
        load with cypher query if the count >= 100000
        load with a small group size can be used to finish the process
        :return:
        """
        total = len(data)
        for index, row in enumerate(data):
            query = "CREATE ({}:{} ".format(table_name, table_name)
            query += "{ "
            query += ", ".join(
                "{}: \"{}\"".format(col, str(row[col]).replace('"', '\'').replace('&', '&.').replace('""', '\'')) for
                col in row)
            query += "});"
            self._progress(index, total, status='Generating cypher query for table: {}'.format(table_name))

            # write to file
            if self.output:
                self._export_file(table_name, [query])
            else:
                pass
            # db.run(query)
            # db.run(query)
            # print(query)

    def _isvalid_load(self, table_name, data):
        """
        check whether data is valid and then load the data into the database
        :param data:
        :return:
        """
        if len(data) == 0:
            raise ValueError("Please insert at least one data in your table")
        if len(data) >= 100000 or self.output:
            self.execute_cypher("MATCH ({}:{})DETACH DELETE {};".format(str(table_name).lower(),
                                                                        table_name, str(table_name).lower()))
            self._load_with_cypher(table_name, data)
        else:
            return self._load_with_csv(table_name, data)

        return None

    def _progress(self, count, total, status=''):
        """
        add the process bar for python
        :param count:
        :param total:
        :param status:
        :return:
        """
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))
        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
        sys.stdout.flush()

    def exporting(self):
        """
        export the table data into csv ready to load into database by using two methods
        1. load csv,
        2. load with cypher query
        :return:
        """

        execute_query = self.execute_mysql if self.db_name != 'psql' else self.execute_psql
        start = time.time()
        export_tables = self.read_relations()
        cypher_query = []
        # to record whether the tables data already converted
        exported = []
        print("Starting export csv files for tables! Please wait for a while ...")
        self.logger.warning("Start exporting the {} database to graph database".format(self.db_name))

        # print(export_tables)
        # table is the key name
        for table in export_tables:
            if export_tables[table] is None:
                data = self.execute_mysql("SELECT * FROM %s;" % table)
                if len(data) == 0:
                    raise ValueError("Please insert at least one data in your table")

                exported.append(table)
                # remove the old data firstly
                self.execute_cypher(
                    "MATCH ({}:{}) DETACH DELETE {};".format(str(table).lower(), table, str(table).lower()))
                # if the dataset is too large then use cypher query to load or output only
                if len(data) >= 100000 or self.output:
                    self._load_with_cypher(table, data)
                else:
                    query = self._load_with_csv(table, data)
                    cypher_query.append(query)
            else:
                # means it should have some relation, key means table name
                src = table
                if src not in exported:
                    src_data = execute_query("SELECT * FROM %s;" % src)
                    result = self._isvalid_load(src, src_data)
                    if result is not None:
                        cypher_query.append(result)
                # to record the converted table
                exported.append(src)

                for t in export_tables[table]:
                    dst = t['dst']
                    src_key = t['src_key']
                    dst_key = t['dst_key']
                    label = t['label']

                    if dst not in exported:
                        dst_data = execute_query("SELECT * FROM %s;" % dst)
                        result = self._isvalid_load(dst, dst_data)
                        if result is not None:
                            cypher_query.append(result)

                    exported.append(dst)
                    query = "MATCH ({}:{}), ".format(str(src).lower(), src)
                    query += "({}:{}) ".format(str(dst).lower(), dst)
                    query += "WHERE {}.{} = {}.{} ".format(src, src_key, dst, dst_key)
                    query += "MERGE ({})-[r:{}]->({})".format(str(src).lower(), label, str(dst).lower())
                    print("ConvertDB query:" , query)

                    cypher_query.append(query)

        if self.output:
            self._export_file("relationship", cypher_query)
        else:
            # add progress bar in the terminal
            total = len(cypher_query)
            for index, cypher in enumerate(cypher_query):
                self._progress(index, total, status='Execute cypher query')
                print("Execute: {}".format(cypher))
                self.execute_cypher(cypher)
        print("Export finished!")
        self.c.warning("Export finished {} database to graph database".format(self.db_name))

        # after exporting the data then delete the csv files which cached in the neo4j directory
        self.logger.warning("Start cleaning the cache file... for {} database".format(self.db_name))
        print("Start cleaning the cache file...")
        for file in self.delete_files:
            os.remove(file)

        end = time.time()
        self.logger.warning("Cost {:2}s to exporting {} database".format(round(float(end - start), 2), self.db_name))
        print("Cost {:2}s to exporting".format(round(float(end - start)), 2))
