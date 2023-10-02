import glob, os, json, argparse
from schema2graph import DBengine
from moz_sql_parser import parse
from py2neo import Graph
import configparser
import shutil
from sql2cypher import  Formatter, execution_accuracy
from utils import Logger
import dill


config = configparser.ConfigParser()
config.read('../config.ini')
filenames = config["FILENAMES"]

root = filenames['root']
benchmark = filenames['benchmark']

neo4j_uri = filenames['neo4j_uri']
neo4j_user = filenames['neo4j_user']
neo4j_password = filenames['neo4j_password']
graph = Graph(neo4j_uri, auth = (neo4j_user, neo4j_password))

data_folder = os.path.join(root, 'application', 'rel_db2kg', 'data', benchmark)
db_folder = os.path.join(data_folder, 'database')

logger =Logger('/sql2cypher.log')

with open('data/{}.pkl'.format(benchmark), 'rb') as pickle_file:
    rel_db_dataset=dill.load(pickle_file)



parser = argparse.ArgumentParser(description='text2sql2cypher.')
parser.add_argument('--spider', help='build graph from spider.', action='store_true')
parser.add_argument('--bird', help='build graph from bird.', action='store_true')
parser.add_argument('--wikisql', help='build graph from wikisql.', action='store_true')
parser.add_argument('--restart', help='build graph from spider and lowercasing all properties.', action='store_true')
parser.add_argument('--cased', help='build graph from spider and lowercasing all properties.', action='store_true')
args = parser.parse_args()

db_paths=glob.glob(db_folder + '/**/*.sqlite', recursive = True) 


text2sql_pres_folds = os.path.join(root, 'application', 'rel_db2kg', 'text2sql', 'pricai')

for model in ['CodeT5_base_prefix_spider_with_cell_value', 'CodeT5_base_finetune_spider_with_cell_value', 'T5_base_prefix_spider_with_cell_value','T5_base_finetune_spider_with_cell_value']:


    sp_out_folder = os.path.join(root, 'sp_data_folder','text2sql2cypher', model)
    if not os.path.exists(sp_out_folder):
        os.makedirs(sp_out_folder) 

    json_file = os.path.join(text2sql_pres_folds, model, 'predictions_predict.json')
    print(json_file)

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)


    qa_pairs = {'correct_': [], 'incorrect_':[], 'pairs':[]}
    f_sql = {'invalid_parsed':[], 'intersect': [], 'except':[]}



    for i, every in enumerate(data):
        db_name = every['db_id']
        print(f'db: {db_name}')
        if db_name:       
            question = every['question']
            sql_prediction = every['prediction']
            sql_gold= every['query']
                
            db_path = os.path.join(db_folder, db_name, '{}.sqlite'.format(db_name))  
            engine = DBengine(db_path)
            sql_gold_result = []
            sql_preds_result = []
            try:

                sql_gold_result = engine.execute(sql_gold).fetchall()
                sql_preds_result = engine.execute(sql_prediction).fetchall()
            except:
                logger.error('Attention in {}, exist Invalid sql query:{}'.format(db_name, sql_gold))
                continue


            try:
                # 3. Convert SQL prediction to Cypher query.	
                pred_parsed_sql = parse(sql_prediction)	
                gold_parsed_sql = parse(sql_gold)
                print(f'pred_parsed_sql: {pred_parsed_sql}')

        
                formatter  = Formatter( logger, db_name, rel_db_dataset.rel_dbs[db_name], graph)
                pred_sql2cypher = formatter.format(pred_parsed_sql)
                gold_sql2cypher = formatter.format(gold_parsed_sql)
                print("**************Cypher Query***************")
                print("pred_sql2cypher:")
                print(pred_sql2cypher)
                print("gold_sql_2cypher:")
                print(gold_sql2cypher)
                print("**************Cypher Query***************")

                if pred_sql2cypher and gold_sql2cypher:
                    try:
                        cypher_pred_res = graph.run(pred_sql2cypher).data()
                    except:
                        qa_pairs['incorrect_'].append(
                            {
                                'db_id':db_name, 
                                'index': i,
                                'gold_sql': sql_gold,
                                'gold_sql2cypher': gold_sql2cypher,
                                'pre_sql':sql_prediction,
                                'pred_sql2cypher': pred_sql2cypher,
                                'question':question,
                            })
                        continue
                    cypher_pred_ans = []
                    for dict_ in cypher_pred_res:
                        cypher_pred_ans.append(tuple(dict_.values()))

                
                    cypher_gold_res = graph.run(gold_sql2cypher).data()
                    cypher_gold_ans = []
                    for dict_ in cypher_gold_res:
                        cypher_gold_ans.append(tuple(dict_.values()))

                    
                    if set(cypher_pred_ans)==set(cypher_gold_ans):
                        print(f'correct_ans: {cypher_pred_ans}') 
                        qa_pairs['correct_'].append(
                            {
                                'db_id':db_name, 
                                'gold_sql': sql_gold,
                                'gold_sql2cypher': gold_sql2cypher,
                                'pre_sql':sql_prediction,
                                'pred_sql2cypher': pred_sql2cypher,
                                'question':question,
                                'answers':cypher_pred_ans
                            })
                        qa_pairs['pairs'].append(every)
                        
                    else:
                        print(f'incorrect_ans: {cypher_pred_ans}')
                        qa_pairs['incorrect_'].append(
                            {
                                'db_id':db_name, 
                                'index': i,
                                'gold_sql': sql_gold,
                                'gold_sql2cypher': gold_sql2cypher,
                                'gold_ans': cypher_gold_ans,
                                'pre_sql':sql_prediction,
                                'pred_sql2cypher': pred_sql2cypher,
                                'pred_ans': cypher_pred_ans,
                                'question':question,
                            })
            except:
                every.update({'index':i})
                if 'intersect' in sql_gold.lower():
                    f_sql['intersect'].append(every)
                if 'except' in sql_gold.lower():
                    f_sql['except'].append(every)
                else:
                    f_sql['invalid_parsed'].append(every)
                logger.error('Attention in {}.db. Can not parse sql query:{}'.format(db_name, pred_parsed_sql))
                print(pred_parsed_sql)


    metrics_file = os.path.join(root, 'application', 'rel_db2kg', 'text2sql2cypher_metrics.json')
    metrics = execution_accuracy(metrics_file, 'text2sql2cypher_{}'.format(model), qa_pairs, f_sql)
    print(f'metrics: {metrics}')

    for key, item in qa_pairs.items():
        with open(os.path.join(sp_out_folder, '{}_{}_{}.json'.format('text2sql2cypher', model, key)) , 'a')  as f:
            json.dump(qa_pairs[key], f, indent = 4)
    for key, item in f_sql.items():
        with open(os.path.join(sp_out_folder, '{}.json'.format(key)) , 'a')  as f:
            json.dump(item, f, indent = 4)
