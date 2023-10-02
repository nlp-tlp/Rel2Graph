'''
Author: Ziyu Zhao
Affiliation: UWA NLT-TLP GROUP
'''

import os, re, math
from typing import Set
from py2neo import Graph
from py2neo.matching import *
from py2neo.data import Node, Relationship
from environs import Env


import numpy as np
import sqlite3
import pandas as pd
import dill

class DBengine:
    
    """
    An Entity, which takes a sqlite3 database schema.
    Parameters:
    
    """

    def __init__(self, fdb):

        self.conn = sqlite3.connect(fdb)      
        self.conn.text_factory = lambda b: b.decode(errors = 'ignore')

    
    def execute(self, query):
      cursor = self.conn.cursor()  
      return cursor.execute(query)  
    
    def get_table_names(self):
        cursor = self.conn.cursor()  
        return cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")     
    
    def get_table_values(self, table_name):
        cursor = self.conn.cursor()  
        return cursor.execute('SELECT * from "{}"'.format(table_name)) 

    def get_schema(self, table_name):
        cursor = self.conn.cursor()  
        cursor.execute('SELECT sql FROM sqlite_master WHERE type = "table" and name= "{}";'.format(table_name))  
        result = cursor.fetchone()
        if result is None:
            raise ValueError("Table %s does not exist" % table_name)
        return result[0].strip()  
   
    def get_data_type(self, table_name, headers):
        cursor = self.conn.cursor() 
        infos = cursor.execute("PRAGMA table_info('{}')".format(table_name))
        data_types = {}
        '''
        Parameters:  
            cid: Column ID
            name: Column name
            type: Column data type
            notnull: Non-null constraint flag (1 if the column has a NOT NULL constraint, 0 otherwise)
            dflt_value: Default value for the column (if any)
            pk: Primary key flag (1 if the column is part of the primary key, 0 otherwise)
        '''
        for info in infos:
            cid, col_name, dtype, notnull, dflt_value, pk = info
            col_name = headers[cid]
            if dtype.lower() in ['text', 'varchar', 'varchar(20)']:
                dtype=str
            if dtype in ['INTEGER', 'INT']:
                dtype=int
            data_types[col_name] = dtype
        return data_types

    def get_outbound_foreign_keys(self, table_name, headers, db_tabs):
        cursor = self.conn.cursor() 
        infos = cursor.execute("PRAGMA foreign_key_list('{}')".format(table_name)).fetchall()
        constraints_ = {}
        id_fks = {}
        '''
        Parameters:
            id: A unique identifier for the foreign key constraint.
            seq: The sequence number of the column in the foreign key constraint.
            table: The name of the table that contains the foreign key constraint.
            from: The name of the column in the current table that is part of the foreign key relationship.
            to: The name of the column in the referenced table that is being referenced by the foreign key.
            on_update: The action to be performed when the referenced column is updated.
            on_delete: The action to be performed when the referenced column is deleted.
            match: The match type for the foreign key constraint. 
        '''
        for info in infos:
            id, seq, to_table, fk, to_col, on_update, on_delete, match = info 
            print(f'table: {table_name}, fk: {fk}, ref_table: {to_table}, to_col:{to_col}')
            fk_idx = [h.lower() for h in headers].index(fk.lower())
            fk = headers[fk_idx]
            if to_col:
                if to_table and to_table not in db_tabs:
                    db_ = [tab.lower() for tab in db_tabs]
                    if to_table.lower() in db_:
                        to_table = db_tabs[db_.index(to_table.lower())]  
                    else:
                        continue
                to_tab_headers = [desc[0] for desc in self.get_table_values(to_table).description]
                to_ = [h.lower() for h in to_tab_headers]
                if to_col.lower() not in to_: # amend mismatch reference foreign key
                    to_pks = self.get_primay_keys(to_table) 
                    if to_pks and len(to_pks)==1:
                        to_col= to_pks[0]
                    else:
                        continue
                    
                to_col_idx = to_.index(to_col.lower())
                to_col = to_tab_headers[to_col_idx]

                if id not in id_fks:
                    id_fks[id]=[fk]
                else:
                    id_fks[id].append(fk)
                
                if id not in constraints_:
                    constraints_[id]= {  
                            "to_tab": to_table, 
                            "to_col_list": [to_col]
                        }
                else:
                    constraints_[id]["to_col_list"].append(to_col)
        constraints = {}
        for id, _ in constraints_.items():
            if id in id_fks:
                constraints[','.join(id_fks[id])]=constraints_[id]

        if bool(constraints):
            return constraints
    
    def get_primay_keys(self, table_name):
        cursor = self.conn.cursor() 
        pks = cursor.execute('SELECT l.name FROM pragma_table_info("{}") as l WHERE l.pk <> 0;'.format(table_name)).fetchall()
        return [pk[0] for pk in pks]  
    
    def check_compound_pk(self, primary_keys):
        if len(primary_keys)>1: 
            return True
        return False
    
    def get_fk_values(self, this_fk, ref_fk, this, ref):
        cursor = self.conn.cursor() 
        fk_values = cursor.execute('SELECT distinct {0}.{1} FROM {0} JOIN {2} ON {0}.{1} = {2}.{3};'.format(this, this_fk, ref, ref_fk)).fetchall()
        return [each[0] for each in fk_values]

    def close(self):
        self.conn.cursor.close()

class RelTable:
    
    """
    Parameters:
        db_name (Strint):
        table_name (String):
        headers (List)): A list of table headers.
        data_type (Dict): A dictionary of each column's data type. 
        pks (List): A list of primary and foreign keys appearing in this table. 
        fks (Dict):
        is_coumpound_pk (bool):
        rows (List): A list of rows appearing in this table. 
        cols (Dict): A dictionary of each column appearing in this table.
        
    """

    def __init__(self, db_name, table_name):
        self.db_name = db_name
        self.table_name = table_name
        self.headers = []
        self.data_type = {}
        self.pks = []
        self.fks= {} 
        self.is_compound_pk = False
        self.rows = []
        self.cols = {}

    def add_row(self, row_dict):
        """
        Add a row to this table.
        Args:
            table (Table object): The table to add.
        """
        self.rows.append(row_dict)
 
class RelDBDataset(DBengine):
    """dtaset class takes a relational databse `schema` as input. 

    A class that represents the whole relational database dataset, i.e. a dataset that constains its whole tables.
    Parameters:
        rel_dbs (Dict): 
                    {

                    }
        fdb: The schema path of a relational database.
        fk_constraints (Dict): Store the information of each table's fk constraints.        
    """
    def __init__(self, paths, logger):
        self.rel_dbs, self.db_fk_constraints, self.db_data_type, self.db_pks= self.read_dataset(paths)
        self.logger = logger
        
    def fix_reference(self, tbs_fk_constraints, tbs_pks, db_tbs):
        for tb_name, fks in tbs_fk_constraints.items():
            pks = tbs_pks[tb_name]
            for fk, constraint in fks.items():
                to_col_list = constraint['to_col_list']
                to_tb_name = constraint['to_tab']
                if to_tb_name==tb_name:
                    for tb_name_ , pks in tbs_pks.items():
                        pks_  = [pk.lower() for pk in pks]
                        assert ',' not in fk
                        if fk.lower() in pks_:
                            constraint['to_tab'] = tb_name_
                            for i, to_col in enumerate(to_col_list):
                                if to_col.lower() not in pks_:
                                    constraint['to_col_list'][i]= pks[pks_.index(fk.lower())]
                    fks.update({fk:constraint})
                    db_tbs[tb_name].fks.update({fk:constraint})

    def fix_fk(self, tbs_fk_constraints, tbs_data_type, db_tbs):
        for tb_name, fks in tbs_fk_constraints.items():
            for fk_, constraint in fks.items():
                to_col_list = constraint['to_col_list']
                to_tb_name = constraint['to_tab']
                fks = [fk.strip() for fk in fk_.split(',')]
                to_col_types = [tbs_data_type[to_tb_name][col] for col in to_col_list]
                fk_data_type_mapping = dict(zip(fks, to_col_types))
                for i, row_dict in enumerate(db_tbs[tb_name].rows):
                    for fk in fks:
                        row_value = row_dict[fk]
                        to_col_value_type = fk_data_type_mapping[fk]
                        if row_value:
                            if (to_col_value_type== 'int' or to_col_value_type==int) and not isinstance(row_value, int):
                                if type(row_value)==str:
                                    row_value=int(row_value)
                                row_dict[fk] = int(row_value) if not math.isnan(row_value) else None
                            elif (to_col_value_type == 'float' or to_col_value_type==float) and not isinstance(row_value, float):
                                if type(row_value)==str:
                                    row_value=float(row_value)
                                row_dict[fk] = float(row_value) if not math.isnan(row_value) else None
                            elif (to_col_value_type == 'str' or to_col_value_type==str) and not isinstance(row_value, str):
                                row_dict[fk] = "'{}'".format(str(row_value).strip('\'').strip('\"')) if row_value else None
                #update corresponding 'cols'
                db_tbs[tb_name].cols = pd.DataFrame(db_tbs[tb_name].rows).to_dict( orient='list')

            

    def read_dataset(self, paths):
        rel_dbs = {}
        db_fk_constraints = {}
        db_data_type = {}
        db_pks = {}
        for db_path in paths:
            path_compodbnents = db_path.split(os.sep)
            db_name = path_compodbnents[-1].split('.')[0] 
            ###########################################This is for data consistency checking code#######################
            
            # missing_data_file = '/Users/ziyuzhao/Desktop/phd/SemanticParser4Graph/application/rel_db2kg/consistency_check/diff_stat.json'
            # check_dbs = []
            # isFile = os.path.isfile(missing_data_file)
            # if isFile:
            #     if len(read_json(missing_data_file))==1:
            #         missing_dbs = read_json(missing_data_file)[0]['dff_dbs']
            #         for missing in missing_data:
            #             if '.' in missing:
            #                 missing_db, missing_table = missing.split('.')
            #                 if missing_db not in check_dbs:
            #                      check_dbs.append(missing_db)
            #     if db_name not in check_dbs:
            #         pass
            ###########################################to make sure the acutal data is the same as expected data#######################
           # in [ 'car_1',  'department_management', 'musical', 'pets_1', 'real_estate_properties', "local_govt_and_lot", 'concert_singer', ]
            if db_name:
                rel_dbs[db_name]={}               
                db_fk_constraints[db_name] = {}
                db_data_type[db_name]={}
                db_pks[db_name]={}
                engine = DBengine(db_path)
                tab_names = [tab_info[0] for tab_info in engine.get_table_names()]
                drop_flag = False
                print(f'db:{db_name}, tab_names: {tab_names}')
            
                for table_name in tab_names:
                    tb_object = RelTable(db_name, table_name)                     
                    table_records = engine.get_table_values(table_name)
                    tb_object.headers = [desc[0] for desc in table_records.description]     
                    tb_object.data_type= engine.get_data_type(table_name, tb_object.headers)
                    tb_object.fks =  engine.get_outbound_foreign_keys(table_name, tb_object.headers, tab_names) 
                    tb_object.pks = engine.get_primay_keys(table_name) 
                    tb_object.is_compound_pk =  engine.check_compound_pk(tb_object.pks)

                    db_data_type[db_name][table_name] = tb_object.data_type 
                    db_pks[db_name][table_name] = tb_object.pks
        
                    if bool(tb_object.fks):
                        db_fk_constraints[db_name][table_name]=tb_object.fks

                    # Do not drop duplicate rows in place, because this action would affect the query results.
                    df = pd.DataFrame(table_records.fetchall(), columns = tb_object.headers)
                    # df.drop_duplicates(inplace=True)
                    # data = df.transpose().to_dict().values()  # to keep the origial data types.
                    data = [d for d in df.transpose().to_dict().values()  if any(d.values())]

                    tb_object.cols = df.to_dict(orient='list')
                    
                    
                    # if len(data)>4000 or tb_object.headers == None:  # we set a threshold for the experiment
                    #     drop_flag =True
                    #     break


                    rows = data or [ {k: '' for k in tb_object.headers} ]
                    list(map(lambda row: tb_object.add_row(row), rows))
                    rel_dbs[db_name][table_name] = tb_object    
                
                if drop_flag:
                    del rel_dbs[db_name]              
                    del db_fk_constraints[db_name] 
                    del db_data_type[db_name]
                    del db_pks[db_name]
                                                                                                                       
                if db_name in rel_dbs:
                    self.fix_reference(db_fk_constraints[db_name], db_pks[db_name], rel_dbs[db_name])
                    self.fix_fk(db_fk_constraints[db_name], db_data_type[db_name] , rel_dbs[db_name])

        return rel_dbs, db_fk_constraints, db_data_type, db_pks
                    
class RelDB2KGraphBuilder(RelDBDataset):
    
    """
    RelDB2KG graph builder takes, a relational databse `schema` as inputs. 
    and constructs a Neo4J graph via py2neo.

    Parameters:
        paths (string): 
        logger: 
        defined_fields (dict): A dictionary of defined structured fields. 
        node_normaliser (LexicalNormaliser): A LexicalNormaliser whose purpose is to clean the nodes, \\
             resolving issues such as making 'leaks', 'leaking' etc resolve to a single 'leak' node. 
        Note: defined_fields and node_normaliser are placeholders. 
    """

    def __init__( self, paths, 
                        benchmark,
                        logger, 
                        env_file, 
                        if_cased, 
                        node_normaliser=None, 
                        defined_fields=None):
        super().__init__(paths, logger )
        self.paths = paths
        self.benchmark = benchmark
        self.logger = logger
        self.env_file = env_file
        self.if_cased = if_cased
        self.node_normaliser = node_normaliser
        self.defined_fields = defined_fields

        env = Env()
        env.read_env(self.env_file)
        self.graph = Graph(password=env("GRAPH_PASSWORD"))
        self.dataset = RelDBDataset(self.paths, self.logger)

        # with open('data/{}.pkl'.format(self.benchmark), 'wb') as pickle_file:
        #     dill.dump(self.dataset, pickle_file)

    def get_matched_node(self, db_name, to_tab, to_col_list, fks, row_dict, tbs_dict):
        alias = to_tab.lower()
        # assert   isinstance(val, val_type), 'FIX ME'
        match_ =  "match ({}:`{}.{}`) ".format(alias, db_name, to_tab)
        condi = []
        for i, col in enumerate(to_col_list):
            val = row_dict[fks[i]]
            if not tbs_dict[to_tab].cols[col]:
                continue
            val_type = type(tbs_dict[to_tab].cols[col][0])
            if val and isinstance(val, val_type):
                if type(val)==str:
                    val = '"{}"'.format(str(val).strip('\'').strip('\"').strip())
                    # elif not re.findall(r'[\'|\"][a-zA-Z].*', val):
                    #     return None
                    # if isinstance(val, str) and not re.findall(r'[a-zA-Z].*', val):
                    #     return None
                elif type(val)==int:
                    val = int(str(val).strip('\'').strip('\"').strip())     
                condi.append('{}.{}={}'.format( alias, col, val))
            else:
                print(f"Value '{val}' is not an instance of {val_type}")

        if bool(condi):
            where_ = ' where ' + ' and '.join(condi) 
            cypher = match_ + where_ + ' return {}'.format(alias)
            print(f'cypher: {cypher}')
            self.logger.warning(f'cypher_query: {cypher}')
            return self.graph.run(cypher).data()


    def build_schema_nodes(self, tx):
        print("**********TOTEST: Graph Schema  Builder********")
        self.domain_base_node = Node("Domain", name="Schema_Node:Domain")
        tx.create(self.domain_base_node)

        self.table_base_node = Node("Table", name="Schema_Node:Table")
        tx.create(self.table_base_node)

        rel = Relationship(self.table_base_node, "Schema_Reltionship:Part_of", self.domain_base_node)
        tx.create(rel)

    def table2nodes(self, db_name, tbs_dict, tx): 
        print("************START BUILDING GRAPH NODES****************")
        for tb_name, tab in tbs_dict.items():
            if  not tab.fks or  len(tab.fks)!=2 \
                or (len(tab.fks)==2 and bool(tab.pks) and not tab.is_compound_pk):
                # or (len(table.fk_constraints)==2 and not table.check_compound_pk):
                '''
                Case: a whole table is turned into corresponding graph nodes,
                this covers the following cases:
                1) the examples, that exist just one primary key and several forign keys,
                2) the examples, that exist compound primary keys, but no foreign keys. 
                e.g., in musical.db, the table `musical` is the case.
                '''
                print(f'Total rows {len(tab.rows)} of entity table: {tb_name} in {db_name}.')
                for i, row_dict in enumerate(tab.rows):
                    graph_row_dict = dict((k.lower(), v) for k, v in row_dict.items()) if not self.if_cased else row_dict
                    graph_db_name = db_name if self.if_cased else db_name.lower()
                    graph_node_label = tb_name if self.if_cased else tb_name.lower()

                    self.tx = self.graph.begin()
                    r = Node('{}.{}'.format(graph_db_name, graph_node_label), **graph_row_dict)
                    self.tx.create(r)
                    self.graph.commit(self.tx)

    def table2edges(self, db_name, tbs_dict):
        # Note: Open graph transaction again.
        self.logger.warning("************Start Building Graph Edges****************")
        for tb_name, tab in tbs_dict.items():
            if tab.fks:
                if len(tab.fks)==2 and (tab.is_compound_pk or not tab.pks):
                    for row_dict in tab.rows:
                        # and  (not primary_keys or table.check_compound_pk):
                        # class py2neo.data.Relationship(start_node, type, end_node, **properties)
                        self.logger.warning(f'******Building graph edge directly from {tb_name}.*******')   
                        '''
                        NOTE 1: deal with one-to-one by the table itself as edges, aka, a whole table is turned into graph edges,                         
                        NOTE 2: we remove the foreign keys from each graph edge's property set, for the sake of the consistency of graph database.
                            to make sure the update of head or tail nodes would not violate the data consistency.                    
                        '''
                        # update_row_dict = {k: row_dict[k] for k in row_dict if k not in fks}  
                        # self.logger.warning(f'update_row_dict: {update_row_dict}')
                        matched = []
                        for concated_fk, constraint in tab.fks.items():
                            to_tab = constraint['to_tab']
                            to_col_list = constraint['to_col_list']       
                            fks = [fk.strip() for fk in concated_fk.split(',')]     
                            assert tb_name!=to_tab, 'FIX ME'  
                            matched_res = self.get_matched_node(db_name, to_tab, to_col_list, fks, row_dict, tbs_dict) 
                            self.logger.warning(f"matched_res : {matched_res}")   
                            if matched_res:
                                try:
                                    matched.extend(matched_res)         
                                except:
                                    self.logger.warning(f'there is no matched nodes for {tb_name} in {db_name}.')
                      
                        if len(matched)==2:
                            first_node_label, first_node = next(iter(matched[0].items()))
                            second_node_label, second_node = next(iter(matched[1].items()))
                            self.tx = self.graph.begin()
                            rel = Relationship(first_node, '{}.{}'.format(db_name, tb_name), second_node, **row_dict) 
                            self.tx.create(rel)
                            self.graph.commit(self.tx)
                        elif len(matched)>2:
                            self.logger.warning("Hyper edge exist in {db_name}: {tb_name}")
                        else:
                            self.logger.warning("Error: Insufficient matched_labels. Cannot create relationship for {tb_name} in {db_name}.")


    def build_edges(self, db_name, tbs_dict):
        # Note: Open graph transaction again.
        print("************Start Building Curated Graph Edges****************")
        for tb_name, tab in tbs_dict.items():   
            if tab.fks:
                if not (len(tab.fks)==2 and tab.is_compound_pk):
                    self.logger.warning(f"Running {tb_name} in {db_name} with {tab.headers}")
                    for row_dict in tab.rows:
                        for concated_fk, constraint in tab.fks.items():
                            to_tab = constraint['to_tab']
                            to_col_list = constraint['to_col_list']            
                            fks = [fk.strip() for fk in concated_fk.split(',')]     
                            assert tb_name!=to_tab, 'FIX ME'
                            print(f'tb_name: {tb_name}, fks:{fks}, to_tab: {to_tab}, to_col_list: {to_col_list}')
                            these = self.get_matched_node(db_name, tb_name, fks, fks, row_dict, tbs_dict)  
                            refs = self.get_matched_node(db_name, to_tab, to_col_list, fks, row_dict, tbs_dict)
                            if these and refs:
                                for this in these:
                                    for s_label, this_node in this.items():
                                        start_node_label='{}.{}'.format(db_name, s_label)
                                    for ref in refs:
                                        for ref_label, ref_node in ref.items():
                                            end_node_label='{}.{}'.format(db_name, ref_label)
                                        print(f'start_node_labe:{start_node_label}, end_node_label: {end_node_label}')
                                        self.tx = self.graph.begin()
                                        rel = Relationship(this_node, '{}_HAS_{}'.format( start_node_label, end_node_label ), ref_node) 
                                        self.tx.create(rel)
                                        self.graph.commit(self.tx)


    def build_graph(self, restart_flag):
        self.tx = self.graph.begin()
        tx = self.tx
        print('restart?', restart_flag)
        if restart_flag:
            self.graph.delete_all()
            # self.build_schema_nodes(tx)
        for db_name, tbs_dict in self.dataset.rel_dbs.items():      
            print("************Start building graph directly from relational table schema****************")
            self.table2nodes(db_name, tbs_dict, tx)
            self.table2edges(db_name, tbs_dict)
            self.build_edges(db_name, tbs_dict) # this process should run based on table2node process
        
def create_edge(graph, db_id, this_tb, ref_tb, this_fk, ref_fk, fk_value):
    cypher = 'MATCH (m:`{0}.{1}` {{{3}: {5}}}),(n:`{0}.{2}` {{{4}:{5}}}) \
                CREATE (m)-[:`{0}.{1}_HAS_{0}.{2}`]->(n);'.format(db_id, this_tb, ref_tb, this_fk, ref_fk, fk_value)
    print(cypher)
    graph.run(cypher)


def main():
    import json
    import glob, argparse
    import configparser
    from rel2kg_utils import Logger
    config = configparser.ConfigParser()

    config.read('../config.ini')
    filenames = config["FILENAMES"]

    root = filenames['root']
    benchmark = filenames['benchmark']
    env_file = os.path.join(root, 'application', '.env')
    # if not os.path.exists(os.path.dirname('/data')):
    #     try:
    #         os.makedirs(os.path.dirname('/data'))
    #     except:
    #         raise NotImplementedError
        
    db_folder = os.path.join(root, 'application', 'data', benchmark, 'database')
    benchmark_dbs = glob.glob(db_folder + '/**/*.sqlite', recursive = True) 
    print(benchmark_dbs)

    parser = argparse.ArgumentParser(description='relational database to graph database.')
    # parser.add_argument('--consistencyChecking', help='Check the consistency between fields in sql query and schema', action='store_true')
    parser.add_argument('--spider', help='build graph from spider.', action='store_true')
    parser.add_argument('--kaggleDBQA', help='build graph from bird.', action='store_true')
    parser.add_argument('--wikisql', help='build graph from wikisql.', action='store_true')
    parser.add_argument('--create', help='Mannually create graph edges.', action='store_true')
    parser.add_argument('--restart', help='build graph from spider and lowercasing all properties.', action='store_true')
    parser.add_argument('--cased', help='build graph from spider and lowercasing all properties.', action='store_true')
    args = parser.parse_args()


    if args.kaggleDBQA or args.spider:
        RelDB2KGraphBuilder(benchmark_dbs,  benchmark, Logger('/{}_rel_schema2graph.log'.format(benchmark)), env_file, if_cased=args.cased).build_graph(restart_flag = args.restart)

    if args.wikisql:
        raise NotImplementedError
    
    if args.create:
        # To advoid Java heap space issue, in each relational database,
        # we retrive the relational databases to get the corresponding rows aligned with the foreign key constraints.
        db_info_fph = os.path.join(root, 'application', 'rel_db2kg', 'data', benchmark,  'KaggleDBQA_tables.json')
        env = Env()
        env.read_env(env_file)
        graph = Graph(password=env("GRAPH_PASSWORD"))

        with open(db_info_fph, 'r') as f:
            db_info = json.load(f)
            for db in db_info:
                db_id = db.get('db_id')
                fdb = os.path.join(root, 'application', 'rel_db2kg', 'data', benchmark, 'database', db_id, '{}.sqlite'.format(db_id))
                engine = DBengine(fdb)
                fks = db.get('foreign_keys')
                # pks = db.get('primary_keys')
                tbs = db.get('table_names_original')
                cols = db.get('column_names_original')

                if fks:
                    for pair in fks:
                        this = cols[pair[0]] # [tb_id, col_name]
                        ref = cols[pair[1]]
                        this_tb = tbs[this[0]]
                        ref_tb = tbs[ref[0]]
                        this_fk = this[1]
                        ref_fk =ref[1]
                        fk_values = engine.get_fk_values( this_fk, ref_fk, this_tb, ref_tb)
                        print(db_id, this_tb, ref_tb, this_fk, ref_fk)
                        for fk_value in fk_values:
                            if isinstance(fk_value, str):
                                fk_value = "'{}'".format((fk_value.strip("'").strip('"')))
                            create_edge(graph, db_id, this_tb, ref_tb, this_fk, ref_fk, fk_value)
    


if __name__ == "__main__":
    main()
