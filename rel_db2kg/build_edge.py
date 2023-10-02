import configparser, os
config = configparser.ConfigParser()
from py2neo import Graph
from py2neo.data import Node, Relationship
from py2neo.matching import *
from environs import Env
import sqlite3
import json

config.read('../config.ini')
filenames = config["FILENAMES"]

root = filenames['root']
benchmark = filenames['benchmark']
env_file = os.path.join(root, 'application', '.env')

def get_fk_values(fdb, this_fk, ref_fk, this, ref):
    conn = sqlite3.connect(fdb)  
    conn.text_factory = lambda b: b.decode(errors = 'ignore')
    cursor = conn.cursor() 
    print('sql:', 'SELECT distinct {0}.{1} FROM {0} JOIN {2} ON {0}.{1} = {2}.{3};'.format(this, this_fk, ref, ref_fk))
    fk_values = cursor.execute('SELECT distinct {0}.{1} FROM {0} JOIN {2} ON {0}.{1} = {2}.{3};'.format(this, this_fk, ref, ref_fk)).fetchall()
    
    return [each[0] for each in fk_values]


def create_edge( db_id, this_tb, ref_tb, this_fk, ref_fk, fk_value):
    env = Env()
    env.read_env(env_file)
    graph = Graph(password=env("GRAPH_PASSWORD"))

    cypher = 'MATCH (m:`{0}.{1}` {{{3}: {5}}}),(n:`{0}.{2}` {{{4}:{5}}}) \
                CREATE (m)-[:`{0}.{1}_HAS_{0}.{2}`]->(n);'.format(db_id, this_tb, ref_tb, this_fk, ref_fk, fk_value)

    print(cypher)
    graph.run(cypher)

db_info_fph = os.path.join(root, 'application', 'rel_db2kg', 'data', benchmark,  'KaggleDBQA_tables.json')
with open(db_info_fph, 'r') as f:
    db_info = json.load(f)
for db in db_info:
    db_id = db.get('db_id')
    fks = db.get('foreign_keys')
    # pks = db.get('primary_keys')
    tbs = db.get('table_names_original')
    cols = db.get('column_names_original')
    cols_type = db.get('column_types')
    if fks:
        for pair in fks:
            this = cols[pair[0]] # [tb_id, col_name]
            ref = cols[pair[1]]
            this_tb = tbs[this[0]]
            ref_tb = tbs[ref[0]]
            this_fk = this[1]
            ref_fk =ref[1]
            this_fk_type = cols_type[pair[0]]
            ref_fk_type = cols_type[pair[1]]
            print(db_id)
            print(this_tb, this_fk, ref_tb, ref_fk)
            if db_id not in ['Pesticide', 'WhatCDHipHop']:
                fdb = os.path.join(root, 'application', 'rel_db2kg', 'data', benchmark, 'database', db_id, '{}.sqlite'.format(db_id))
                fk_values = get_fk_values(fdb, this_fk, ref_fk, this_tb, ref_tb)
                for fk_value in fk_values:
                    if this_fk_type=='text' or ref_fk_type=='text':
                        fk_value="'{}'".format(fk_value)             
                    create_edge(db_id, this_tb, ref_tb, this_fk, ref_fk, fk_value)
