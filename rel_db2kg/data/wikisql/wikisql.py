"""A large crowd-sourced dataset for developing natural language interfaces for relational databases"""
"""https://github.com/salesforce/WikiSQL/blob/master/lib/common.py"""

import json,re, math, os
import datasets
import records
from babel.numbers import parse_decimal, NumberFormatError
from copy import deepcopy 
import pandas as pd
schema_re = re.compile(r'\((.+)\)')
num_re = re.compile(r'[-+]?\d*\.\d+|\d+')
re_whitespace = re.compile(r'\s+', flags=re.UNICODE)

_DATA_URL = "/Users/ziyuzhao/Desktop/phd/SemanticParser4Graph/application/data/wikisql"
_AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
_COND_OPS = ["=", ">", "<", "OP"]

def detokenize(tokens):
    ret = ''
    for g, a in zip(tokens['gloss'], tokens['after']):
        ret += g + a
    return ret.strip()

class Query:
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=', '>', '<', 'OP']
    syms = ['SELECT', 'WHERE', 'AND', 'COL', 'TABLE', 'CAPTION', 'PAGE', 'SECTION', 'OP', 'COND', 'QUESTION', 'AGG',
            'AGGOPS', 'CONDOPS']

    def __init__(self, sel_index, agg_index, conditions=tuple(), ordered=False):
        self.sel_index = sel_index
        self.agg_index = agg_index
        self.conditions = list(conditions)
        self.ordered = ordered

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            indices = self.sel_index == other.sel_index and self.agg_index == other.agg_index
            if other.ordered:
                conds = [(col, op, str(cond).lower()) for col, op, cond in self.conditions] == [
                    (col, op, str(cond).lower()) for col, op, cond in other.conditions]
            else:
                conds = set([(col, op, str(cond).lower()) for col, op, cond in self.conditions]) == set(
                    [(col, op, str(cond).lower()) for col, op, cond in other.conditions])

            return indices and conds
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))

    def __repr__(self):
        rep = 'SELECT {agg} {sel} FROM table'.format(
            agg=self.agg_ops[self.agg_index],
            sel='col{}'.format(self.sel_index),
        )
        if self.conditions:
            rep += ' WHERE ' + ' AND '.join(
                ['{} {} {}'.format('col{}'.format(i), self.cond_ops[o], v) for i, o, v in self.conditions])
        return rep

    def to_dict(self):
        return {'sel': self.sel_index, 'agg': self.agg_index, 'conds': self.conditions}

    def lower(self):
        conds = []
        for col, op, cond in self.conditions:
            conds.append([col, op, cond.lower()])
        return self.__class__(self.sel_index, self.agg_index, conds)

    @classmethod
    def from_dict(cls, d, ordered=False):
        return cls(sel_index=d['sel'], agg_index=d['agg'], conditions=d['conds'], ordered=ordered)

    @classmethod
    def from_tokenized_dict(cls, d):
        conds = []
        for col, op, val in d['conds']:
            conds.append([col, op, detokenize(val)])
        return cls(d['sel'], d['agg'], conds)

    @classmethod
    def from_generated_dict(cls, d):
        conds = []
        for col, op, val in d['conds']:
            end = len(val['words'])
            conds.append([col, op, detokenize(val)])
        return cls(d['sel'], d['agg'], conds)

    @classmethod
    def from_sequence(cls, sequence, table, lowercase=True):
        sequence = deepcopy(sequence)
        if 'symend' in sequence['words']:
            end = sequence['words'].index('symend')
            for k, v in sequence.items():
                sequence[k] = v[:end]
        terms = [{'gloss': g, 'word': w, 'after': a} for g, w, a in
                 zip(sequence['gloss'], sequence['words'], sequence['after'])]
        headers = [detokenize(h) for h in table['header']]

        # lowercase everything and truncate sequence
        if lowercase:
            headers = [h.lower() for h in headers]
            for i, t in enumerate(terms):
                for k, v in t.items():
                    t[k] = v.lower()
        headers_no_whitespcae = [re.sub(re_whitespace, '', h) for h in headers]

        # get select
        if 'symselect' != terms.pop(0)['word']:
            raise Exception('Missing symselect operator')

        # get aggregation
        if 'symagg' != terms.pop(0)['word']:
            raise Exception('Missing symagg operator')
        agg_op = terms.pop(0)['word']

        if agg_op == 'symcol':
            agg_op = ''
        else:
            if 'symcol' != terms.pop(0)['word']:
                raise Exception('Missing aggregation column')
        try:
            agg_op = cls.agg_ops.index(agg_op.upper())
        except Exception as e:
            raise Exception('Invalid agg op {}'.format(agg_op))

        def find_column(name):
            return headers_no_whitespcae.index(re.sub(re_whitespace, '', name))

        def flatten(tokens):
            ret = {'words': [], 'after': [], 'gloss': []}
            for t in tokens:
                ret['words'].append(t['word'])
                ret['after'].append(t['after'])
                ret['gloss'].append(t['gloss'])
            return ret

        where_index = [i for i, t in enumerate(terms) if t['word'] == 'symwhere']
        where_index = where_index[0] if where_index else len(terms)
        flat = flatten(terms[:where_index])
        try:
            agg_col = find_column(detokenize(flat))
        except Exception as e:
            raise Exception('Cannot find aggregation column {}'.format(flat['words']))
        where_terms = terms[where_index + 1:]

        # get conditions
        conditions = []
        while where_terms:
            t = where_terms.pop(0)
            flat = flatten(where_terms)
            if t['word'] != 'symcol':
                raise Exception('Missing conditional column {}'.format(flat['words']))
            try:
                op_index = flat['words'].index('symop')
                col_tokens = flatten(where_terms[:op_index])
            except Exception as e:
                raise Exception('Missing conditional operator {}'.format(flat['words']))
            cond_op = where_terms[op_index + 1]['word']
            try:
                cond_op = cls.cond_ops.index(cond_op.upper())
            except Exception as e:
                raise Exception('Invalid cond op {}'.format(cond_op))
            try:
                cond_col = find_column(detokenize(col_tokens))
            except Exception as e:
                raise Exception('Cannot find conditional column {}'.format(col_tokens['words']))
            try:
                val_index = flat['words'].index('symcond')
            except Exception as e:
                raise Exception('Cannot find conditional value {}'.format(flat['words']))

            where_terms = where_terms[val_index + 1:]
            flat = flatten(where_terms)
            val_end_index = flat['words'].index('symand') if 'symand' in flat['words'] else len(where_terms)
            cond_val = detokenize(flatten(where_terms[:val_end_index]))
            conditions.append([cond_col, cond_op, cond_val])
            where_terms = where_terms[val_end_index + 1:]
        q = cls(agg_col, agg_op, conditions)
        return q

    @classmethod
    def from_partial_sequence(cls, agg_col, agg_op, sequence, table, lowercase=True):
        sequence = deepcopy(sequence)
        if 'symend' in sequence['words']:
            end = sequence['words'].index('symend')
            for k, v in sequence.items():
                sequence[k] = v[:end]
        terms = [{'gloss': g, 'word': w, 'after': a} for g, w, a in
                 zip(sequence['gloss'], sequence['words'], sequence['after'])]
        headers = [detokenize(h) for h in table['header']]

        # lowercase everything and truncate sequence
        if lowercase:
            headers = [h.lower() for h in headers]
            for i, t in enumerate(terms):
                for k, v in t.items():
                    t[k] = v.lower()
        headers_no_whitespcae = [re.sub(re_whitespace, '', h) for h in headers]

        def find_column(name):
            return headers_no_whitespcae.index(re.sub(re_whitespace, '', name))

        def flatten(tokens):
            ret = {'words': [], 'after': [], 'gloss': []}
            for t in tokens:
                ret['words'].append(t['word'])
                ret['after'].append(t['after'])
                ret['gloss'].append(t['gloss'])
            return ret

        where_index = [i for i, t in enumerate(terms) if t['word'] == 'symwhere']
        where_index = where_index[0] if where_index else len(terms)
        where_terms = terms[where_index + 1:]

        # get conditions
        conditions = []
        while where_terms:
            t = where_terms.pop(0)
            flat = flatten(where_terms)
            if t['word'] != 'symcol':
                raise Exception('Missing conditional column {}'.format(flat['words']))
            try:
                op_index = flat['words'].index('symop')
                col_tokens = flatten(where_terms[:op_index])
            except Exception as e:
                raise Exception('Missing conditional operator {}'.format(flat['words']))
            cond_op = where_terms[op_index + 1]['word']
            try:
                cond_op = cls.cond_ops.index(cond_op.upper())
            except Exception as e:
                raise Exception('Invalid cond op {}'.format(cond_op))
            try:
                cond_col = find_column(detokenize(col_tokens))
            except Exception as e:
                raise Exception('Cannot find conditional column {}'.format(col_tokens['words']))
            try:
                val_index = flat['words'].index('symcond')
            except Exception as e:
                raise Exception('Cannot find conditional value {}'.format(flat['words']))

            where_terms = where_terms[val_index + 1:]
            flat = flatten(where_terms)
            val_end_index = flat['words'].index('symand') if 'symand' in flat['words'] else len(where_terms)
            cond_val = detokenize(flatten(where_terms[:val_end_index]))
            conditions.append([cond_col, cond_op, cond_val])
            where_terms = where_terms[val_end_index + 1:]
        q = cls(agg_col, agg_op, conditions)
        return q

class wikisql_DBEngine:
    """
    I changed the code "val = float(parse_decimal(val))"
    to "val = float(parse_decimal(val, locale='en_US'))"
    to prevent bugs due to OS and file encoding.
    (for more details why i did that please review the source code of babel package)
    """

    def __init__(self, fdb):
        self.db = records.Database('sqlite:///{}'.format(fdb))
        self.conn = self.db.get_connection()

    def execute_query(self, table_id, query, *args, **kwargs):
        return self.execute(table_id, query.sel_index, query.agg_index, query.conditions, *args, **kwargs)

    def execute(self, table_id, select_index, aggregation_index, conditions, lower=True):
        if not table_id.startswith('table'):
            table_id = 'table_{}'.format(table_id.replace('-', '_'))
        """
        My lesson is, to make this line perform normally, you must keep an older version of sqlalchemy, 1.3.10 for example.
        """
        table_info = self.conn.query('SELECT sql from sqlite_master WHERE tbl_name = :name', name=table_id).all()[0].sql
        schema_str = schema_re.findall(table_info)[0]
        schema = {}
        for tup in schema_str.split(', '):
            c, t = tup.split()
            schema[c] = t
        select = 'col{}'.format(select_index)
        agg = Query.agg_ops[aggregation_index]
        if agg:
            select = '{}({})'.format(agg, select)
        where_clause = []
        where_map = {}
        for col_index, op, val in conditions:
            if lower and isinstance(val, str):
                val = val.lower()
            if schema['col{}'.format(col_index)] == 'real' and not isinstance(val, (int, float)):
                try:
                    val = float(parse_decimal(val, locale='en_US'))
                    # val = float(parse_decimal(val))
                except NumberFormatError as e:
                    val = float(num_re.findall(val)[0])
            where_clause.append('col{} {} :col{}'.format(col_index, Query.cond_ops[op], col_index))
            where_map['col{}'.format(col_index)] = val
        where_str = ''
        if where_clause:
            where_str = 'WHERE ' + ' AND '.join(where_clause)
        query = 'SELECT {} AS result FROM {} {}'.format(select, table_id, where_str)
        out = self.conn.query(query, **where_map)
        return [o.result for o in out]

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

class RelDBDataset(wikisql_DBEngine):
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
            print(f'db_name:', db_name)
            if db_name:
                rel_dbs[db_name]={}               
                db_fk_constraints[db_name] = {}
                db_data_type[db_name]={}
                db_pks[db_name]={}
                engine = wikisql_DBEngine(db_path)
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
            
        
def convert_to_human_readable(sel, agg, columns, conditions):
    """Make SQL query string. Based on https://github.com/salesforce/WikiSQL/blob/c2ed4f9b22db1cc2721805d53e6e76e07e2ccbdc/lib/query.py#L10"""

    rep = "SELECT {agg} {sel} FROM table".format(
        agg=_AGG_OPS[agg], sel=columns[sel] if columns is not None else "col{}".format(sel)
    )

    if conditions:
        rep += " WHERE " + " AND ".join(["{} {} {}".format(columns[i], _COND_OPS[o], v) for i, o, v in conditions])
    return " ".join(rep.split())

def generate_examples( _DATA_URL, split):
    """Yields examples."""
    data=[]
    main_filepath = os.path.join(_DATA_URL, 'data', '{}.jsonl'.format(split))
    tables_filepath = os.path.join(_DATA_URL, 'data', '{}.tables.jsonl'.format(split))
    db_filepath = os.path.join(_DATA_URL, 'data', '{}.db'.format(split))
    db_engine = wikisql_DBEngine(db_filepath)

    # Build dictionary to table_ids:tables
    with open(tables_filepath, encoding="utf-8") as f:
        tables = [json.loads(line) for line in f]
        id_to_tables = {x["id"]: x for x in tables}

    with open(main_filepath, encoding="utf-8") as f:
        
        for idx, line in enumerate(f):
            sample = {}
            row = json.loads(line)
            row["table"] = id_to_tables[row["table_id"]]
            sample["table_id"]= row["table_id"]
            sample['question']=row['question']
            del row["table_id"]

            # Get the result of the query.
            sample['answers'] = [str(result) for result in db_engine.execute_query(row["table"]["id"], Query.from_dict(row["sql"]))]
        
            # Handle missing data
            row["table"]["page_title"] = row["table"].get("page_title", "")
            row["table"]["section_title"] = row["table"].get("section_title", "")
            row["table"]["caption"] = row["table"].get("caption", "")
            row["table"]["name"] = row["table"].get("name", "")
            row["table"]["page_id"] = str(row["table"].get("page_id", ""))

            # Fix row types
            row["table"]["rows"] = [[str(e) for e in r] for r in row["table"]["rows"]]

            # Get human-readable version
            row["sql"]["human_readable"] = convert_to_human_readable(
                row["sql"]["sel"],
                row["sql"]["agg"],
                row["table"]["header"],
                row["sql"]["conds"],
            )
            

            # Restructure sql->conds
            # - wikiSQL provides a tuple [column_index, operator_index, condition]
            #   as 'condition' can have 2 types (float or str) we convert to dict
            for i in range(len(row["sql"]["conds"])):
                row["sql"]["conds"][i] = {
                    "column_index": row["sql"]["conds"][i][0],
                    "operator_index": row["sql"]["conds"][i][1],
                    "condition": str(row["sql"]["conds"][i][2]),
                }
            sample['query']=row["sql"]["human_readable"]
            del row["sql"]["human_readable"]
            sample['sql']=row['sql']
            data.append(sample)

    with open(os.path.join(_DATA_URL, '{}.jsonl'.format(split)), "w") as json_file:
        for record in data:
            json.dump(record, json_file)
            json_file.write('\n')  
    print(f"Data has been written to {json_file}")

def main():
    from py2neo import Graph
    from environs import Env
    import configparser
    config = configparser.ConfigParser()

    config.read('../../config.ini')
    filenames = config["FILENAMES"]

    root = filenames['root']
    env_file = os.path.join(root, 'application', '.env')
    env = Env()
    env.read_env(env_file)
    graph = Graph(password=env("GRAPH_PASSWORD"))

    for split in ['test']:
        generate_examples(_DATA_URL, split )


if __name__ == "__main__":
    main()