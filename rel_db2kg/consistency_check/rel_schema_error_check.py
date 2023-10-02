import os, sys, json, csv, re
import pandas as pd
sys.path.append('..')
from schema2graph import DBengine, RelDB, RelTable, TableSchema
from utils import Logger, save2json, read_json



def read_dataset(paths, logger):

    data_stat =[]
    filtered_list = []

    for i, db_path in enumerate(paths):
        path_compodbnents = db_path.split(os.sep)
        db_name = path_compodbnents[-1].split('.')[0]

        # create realational database object.
        rel_db_object = RelDB(fdb = db_path, db_name=db_name)
        # engine = rel_db_object.engine
        table_infos = rel_db_object.engine.get_table_names()
        
        for table_info in table_infos:
            table_name = table_info[0]
            # Export tables to neo4j/import as csv files.                           
            table_records = rel_db_object.engine.get_table_values(table_name)
            table_headers = [desc[0] for desc in table_records.description]    
            print(f'table_name: {table_name}, table_headers {table_headers}' )
            

            df = pd.DataFrame(table_records.fetchall(), columns = table_headers)
            # Do not drop duplicate rows in place, because this action would affect the query results.
            # df.drop_duplicates(inplace=True)
            data = df.transpose().to_dict().values()  # to keep the origial data types.

            #NOTE: for statistics          
            every = {}  
            every['db_name'] = db_name
            every['table_name']= table_name
            every['num_of_rows'] = len(data)

            if table_headers == None:
                logger.error("There is no table headers in {} of {}!".format(table_name, db_name))
                every['empty_table'] = True
            else:
                every['empty_table'] = False
            

            # create table object. Note: one file w.r.t. one table object. 
            table_object = RelTable(table_name=table_name, table_headers=table_headers)

            # Check if relational table. If yes, then we rewrite the table name starting with
            table_constraints, pks_fks_dict =  rel_db_object.engine.get_outbound_foreign_keys(table_name) #R[{"column": from_, "ref_table": table_name, "ref_column": to_}]
            primary_keys = rel_db_object.engine.get_primay_keys(table_name) #R[(pk, )]

            check_compound_pk =  rel_db_object.engine.check_compound_pk(primary_keys)
                        
            every['primary_keys'] = primary_keys
            every['if_compound_pk'] = check_compound_pk
            every['table_headers'] = table_headers
            every['num_of_headers'] = len(table_headers)
            every['table_constraints'] = table_constraints
            if len(table_constraints)>2:
                every['hyper_edge_candidate'] = True
            else:
                 every['hyper_edge_candidate'] = False
            if len(table_constraints)==len(table_headers)==2 and len(primary_keys)==0:
                every['edge_candidate'] = True
            else:
                every['edge_candidate'] = False

            # create table header category in the format of table_header -> table_name -> db_name. 
            values = []
            if table_headers:
                for row in data:
                    table_row = tuple(row.values())
                    print(table_row)
                    values.append(table_row)
            if len(values)> len(set(values)):
                every['has_duplicate_rows'] = True
            else:
                every['has_duplicate_rows'] = False

            data_stat.append(every)
            if len(data)>4000:
                if len(data)>10000:
                    every['zzy']=True
                filtered_list.append(every)

                

    save2json(data_stat, '/Users/ziyuzhao/Desktop/phd/SemanticParser4Graph/application/rel_db2kg/consistency_check/data_stat.json')
    save2json(filtered_list, '/Users/ziyuzhao/Desktop/phd/SemanticParser4Graph/application/rel_db2kg/consistency_check/filtered_list.json')

def check_difference(root, logger):
    # Check whether a path pointing to a file
    file_path = os.path.join(root, 'application', 'rel_db2kg', 'consistency_check', 'data_stat.json')
    isFile = os.path.isfile(file_path)

    if isFile:
        all_db_list = tuple(set([every['db_name'] for every in read_json(file_path)]))
        filtered_list = tuple(set([every['db_name'] for every in read_json(file_path) \
            if every['num_of_rows']>4000]))
        expected_graph_db_list = list(set(all_db_list) - set(filtered_list))
        print(f' num_of_all_dbs: {len(all_db_list)}, num_of_filtered_dbs: {len(filtered_list)}, num_of_expected_graph_dbs: {len(expected_graph_db_list)}')


        all_tables_list = list([every['table_name'] for every in read_json(file_path)])
        expected_graph_tables_list = list(['{}.{}'.format(every['db_name'], every['table_name'] ) \
            for every in read_json(file_path) \
            if every['db_name'] not in filtered_list])
        print(f' num_of_all_tables: {len(all_tables_list)}, num_of_expected_graph_tables: {len(expected_graph_tables_list)}')
       
        schema_map_file = os.path.join(root, 'application', 'rel_db2kg', 'consistency_check', 'records.json')
        isSchemaMap = os.path.isfile(schema_map_file)
        if isSchemaMap:
            actual_stat = {}
            actual_stat['nodes'] = {}
            actual_stat['relationships'] = {}
            data = read_json(schema_map_file)
            
            total_unique_db =[]
            actual_tables = []

            lookup_dict = {'nodes': 'labels', 'relationships':'type'}
            for variant in list(lookup_dict.keys()):
                counter=0
                curated_counter = 0
                db_counter = 0
                variant_data = data[0][variant]
                for line in variant_data:
                    if lookup_dict[variant] in line:
                        name =  line[lookup_dict[variant]]
                        if isinstance(name, list):
                            name = name[0]
                        if  '.' in name:  
                            # print(variant, lookup_dict[variant], name )
                            split_res = name.split('.')
                            db = split_res[0]
                            tb = split_res[1]
                            if tb not in actual_tables:
                                actual_tables.append(tb)
                            if db not in total_unique_db:
                                total_unique_db.append(db)

                            if db not in actual_stat[variant]:   
                                actual_stat[variant][db]=[]
                                db_counter+=1

                            actual_stat[variant][db].append({tb: line['properties']['count']})
                            counter+=line['properties']['count']
                            if len(split_res)==3:
                                curated_counter+=line['properties']['count']

                actual_stat['{}_counter'.format(lookup_dict[variant])]= counter
                actual_stat['curated_edges'] = curated_counter
                actual_stat['db_counter_in_{}'.format(lookup_dict[variant])] =db_counter
            actual_stat['total_unique_db'] = total_unique_db
            actual_stat['num_of_total_unique_db'] = len(total_unique_db)
            save2json(actual_stat, os.path.join(root, 'application', 'rel_db2kg', 'consistency_check', 'actual_stat.json'))
 
            diff_dbs = list(set(tuple(expected_graph_db_list)) - set(tuple(total_unique_db)))
            diff_tbs = list(set(tuple(expected_graph_tables_list)) - set(tuple(actual_tables)))
            belongs_to_missing_dbs = []
            belongs_to_registered_dbs = []
            for reforamt_diff_db in  diff_tbs:
                if '.' in reforamt_diff_db:
                    diff_db, diff_tb = reforamt_diff_db.split('.')
                    if diff_db in diff_dbs:
                        belongs_to_missing_dbs.append(reforamt_diff_db)
                    else:
                        belongs_to_registered_dbs.append(reforamt_diff_db)
            

            
            print(f'num_of_diff_dbs: {len(diff_dbs)}, num_of_diff_tbs: {len(diff_tbs)}, \
                belongs_to_missing_dbs: {len(belongs_to_missing_dbs)}, \
                belongs_to_registered_dbs: {len(belongs_to_registered_dbs)} ')

            check_registered_dbs=[]
            for registed in belongs_to_registered_dbs:
                registed_db, _  = registed.split('.')
                if registed_db not in check_registered_dbs:
                    check_registered_dbs.append(registed_db)
            # print(check_registered_dbs)


            diff_stat = [{'num_of_expected_graph_dbs':len(expected_graph_db_list), 
                        'num_of_expected_graph_tables_list': len(expected_graph_tables_list), 
                        'num_of_actual_graph_dbs':len(total_unique_db), 
                        'num_of_actual_graph_tables_list': len(actual_tables), 
                        'num_of_diff_dbs': len(diff_dbs), 
                        'num_of_diff_tbs': len(diff_tbs),
                        'num_of_missing_tables_in_missing_dbs': len(belongs_to_missing_dbs),
                        'num_of_missing_tables_in_registered_dbs': len(belongs_to_registered_dbs),
                        'diff_dbs': diff_dbs,
                        'belongs_to_missing_dbs': belongs_to_missing_dbs,
                        'belongs_to_registered_dbs': belongs_to_registered_dbs,
                        'diff_tbs':  diff_tbs,
                        'expected_graph_db_list' :expected_graph_db_list, 
                        'expected_graph_tables_list': expected_graph_tables_list}]
                    

            save2json(diff_stat, '/Users/ziyuzhao/Desktop/phd/SemanticParser4Graph/application/rel_db2kg/consistency_check/diff_stat.json')


    else:
        logger.warning("Do not exist {} or {}!".format(file_path, schema_map_file))
        raise NotImplementedError

def read_cypher(root):
    stat_res = {}
    for split in ['train', 'dev']:
        json_file = os.path.join(root, 'semantic_parser', 'data', 'spider', '{}_correct.json'.format(split))
        # print(json_file)
        f = open(json_file)
        data = json.load(f)

        # # test output file
        # cypher_file_musical  = os.path.join(root, '{}_{}_cypher.json'.format('musical', split))
        stat_res[split]=[]
        dbs = []
        db_counter = 0
        node_pattern_counter = 0
        edge_pattern_counter = 0
        filtering_pattern_counter = 0
        aggregation_pattern_counter = 0
        sub_query_pattern_counter = 0
        agg = ['count', 'avg', 'min', 'max', 'sum']

        for i, every in enumerate(data):
            db_name = every['db_id']
            if db_name not in dbs:
                dbs.append(db_name)
                db_counter+=1
            parsed_cypher = every['parsed_cypher']
            # node patterns: 
            if ')-[' and ']-(' not in parsed_cypher['Token_Punctuation']:
                node_pattern_counter+=1
            # edge patterns: len(Token_Punctuation)>6
            if ']-(' and ']-(' in parsed_cypher['Token_Punctuation']:
                edge_pattern_counter+=1
            # filtering patterns
            if 'WHERE' in parsed_cypher['Token_Keyword'] \
                and ('NOT' not in  parsed_cypher['Token_Operator']):
                filtering_pattern_counter+=1
            # aggregation patterns
            if 'Token_Name_Function' in parsed_cypher:
                aggregation_pattern_counter+=1
            # subquery patterns 
            if 'NOT' in parsed_cypher['Token_Operator']:
                sub_query_pattern_counter+=1
            match_clause_counter = 0
            for key in parsed_cypher['Token_Keyword']:
                if key=='MATCH':
                    match_clause_counter+=1
            if match_clause_counter>=2 and 'WHERE' in parsed_cypher['Token_Keyword']:
                sub_query_pattern_counter+=1

        
        stat_res[split].append({ "db_counter": db_counter, 
                'node_pattern_counter':node_pattern_counter,
                "edge_pattern_counter": edge_pattern_counter, 
                "filtering_pattern_counter": filtering_pattern_counter,
                "aggregation_pattern_counter": aggregation_pattern_counter,
                "sub_query_pattern_counter": sub_query_pattern_counter})
    print(stat_res)

def read_sql(root):
    stat_res = {}
    for split in ['train', 'dev']:
        json_file = os.path.join('/Users/ziyuzhao/Desktop/raw_data/data/spider', '{}.json'.format(split))
        # print(json_file)
        f = open(json_file)
        data = json.load(f)

        # # test output file
        # cypher_file_musical  = os.path.join(root, '{}_{}_cypher.json'.format('musical', split))
        stat_res[split]=[]
        dbs = []
        db_counter = 0
        join_counter = 0
        nested_counter =0
        intersect_counter =0
        except_counter =0
        
        def isNested(json ):
            check_str = str(json)
            if len(re.findall(r'FROM|from', check_str)) != 1:
                return True
            else:
                return False
        
        for i, every in enumerate(data):
            db_name = every['db_id']
            if db_name not in dbs:
                dbs.append(db_name)
                db_counter+=1
            tokens = [token.lower() for token in every["query_toks"]]
            # join on statements
            if 'join' in tokens:
                join_counter+=1


            # if db_name in graph_db_list:
            if db_name:
                # print(f'hey db: {db_name}')
                # for evaluate in [ nested_sql, intersect_sql, except_sql]:
                #     if db_name not in evaluate:
                #         evaluate[db_name]=[]
                    
                # 1. Extract database name, questions and SQL queries
                # all_table_fields = lookup_dict[db_name]	
                question = every['question']
                sql_query = every['query']	


                if 'intersect' in sql_query.lower():
                    intersect_counter+=1
                    # intersect_sql[db_name].append(i)
                if 'except' in sql_query.lower():
                    except_counter+=1
                    # except_sql[db_name].append(i)
                if isNested(sql_query):
                    nested_counter+=1
                    # nested_sql[db_name].append(i)

        stat_res[split].append({ "db_counter": db_counter, 
        'join_counter': [join_counter, join_counter/(i+1)],
        "intersect_counter": [intersect_counter, intersect_counter/(i+1)],
        "except_counter": [except_counter, except_counter/(i+1)],
        "nested_counter": [nested_counter, nested_counter/(i+1)]})

        # f = os.path.join(root, 'application', 'rel_db2kg', 'consistency_check', '{}_stat.json'.format(split))   

    print(stat_res)

                      
def main():
    import glob, argparse
    import configparser
    config = configparser.ConfigParser()
    config.read('../../config.ini')
    filenames = config["FILENAMES"]

    raw_folder = filenames['raw_folder']
    root = filenames['root']

    raw_spider_folder = os.path.join(raw_folder, 'spider')
    db_folder = os.path.join(raw_spider_folder,  'database')
    spider_dbs = glob.glob(db_folder + '/**/*.sqlite', recursive = True) 

    parser = argparse.ArgumentParser(description='check the consistency of mapping relational database to graph database.')
    # parser.add_argument('--consistencyChecking', help='Check the consistency between fields in sql query and schema', action='store_true')
    parser.add_argument('--get', help='get statistical data', action='store_true')
    parser.add_argument('--check', help='check the difference between expected data and actual graph data', action='store_true')
    parser.add_argument('--cypherstat', help='check the statistics of cypher queries and nested subqueries', action='store_true')
    parser.add_argument('--sqlstat', help='check the statistics of sql queries regarding complex queries', action='store_true')
    args = parser.parse_args()

    if args.get:
        read_dataset(spider_dbs, Logger())
    
    if args.check:
        check_difference(root, Logger())

    if args.cypherstat:
        read_cypher(root)
    
    if args.sqlstat:
        read_sql(root)


if __name__ == "__main__":
    main()