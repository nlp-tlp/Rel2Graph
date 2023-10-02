
'''
Author: Ziyu Zhao
Affiliation: UWA NLT-TLP GROUP
'''

import os, re, math
from typing import Set
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
from fire import Fire
import numpy as np

def calculate_num_attributes(conn):
    # Number of attributes (natt)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()

    total_attributes = 0
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name})")
        attributes = cursor.fetchall()
        total_attributes += len(attributes)
    return total_attributes

def get_sub_graphs(conn, table_name):
        cursor = conn.cursor() 
        infos = cursor.execute("PRAGMA foreign_key_list({})".format(table_name)).fetchall()
        sub_graphs = []
        for info in infos:
            id, seq, to_table, fk, to_col, on_update, on_delete, match = info
            sub_graphs.append(to_table)
        return set(sub_graphs)
        
def calculate_cos(conn):
    # Cohesion of the schema (cos)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    # Get the total number of tables in the schema
    print(f'total tables: {len(tables)}')

    total_ntus = []
    for table in tables:
        table_name = table[0]
        # cursor.execute(f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name!='{table_name}' AND sql LIKE '%{table_name}%'")
        # ntus = cursor.fetchone()[0] # ntus: the number of the table in the related subgraph.
        # print(f'ntus: {ntus}')
        sub_graph_table_list = get_sub_graphs(conn, table_name)
        if sub_graph_table_list:
            ntus = len(sub_graph_table_list)+1
            print(f'ntus: {ntus}, sub_graph: {sub_graph_table_list}')
            total_ntus.append(ntus)



    print(f'total_ntus: {len(total_ntus)}')

    # Get the number of tables in each unrelated subgraph
    us = len(tables)-sum(total_ntus)
    cos = sum([ntus ** 2 for ntus in total_ntus])
    print(f'us: {us}, cos: {cos}')
    return cos

def get_drt(conn):
    # Get list of all foreign keys
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in c.fetchall()]
    foreign_keys = []
    for table in tables:
        c.execute(f"PRAGMA foreign_key_list({table})")
        for row in c.fetchall():
            foreign_keys.append((table, row[3], row[2]))

    # Traverse relationships recursively to find longest path
    def dfs(table, path):
        if table not in path:
            path.append(table)
            max_depth = 0
            for fk in foreign_keys:
                if fk[0] == table:
                    depth = dfs(fk[1], path.copy())
                    max_depth = max(max_depth, depth)
            return max_depth + 1
        else:
            return 0

    max_depth = 0
    for table in tables:
        depth = dfs(table, [])
        max_depth = max(max_depth, depth)

    return max_depth

def calculate_metrics(conn):
    c = conn.cursor()

    natt = calculate_num_attributes(conn)

    # Number of foreign keys (nfk) 
    c.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND sql LIKE '%FOREIGN KEY%';")
    nfk = c.fetchone()[0]
    print(f'nfk: {nfk}')
    
    cos = calculate_cos(conn)

    # Schema size (ss)
    c.execute("SELECT SUM(pgsize) FROM dbstat WHERE name NOT LIKE 'sqlite_%';")
    ss = c.fetchone()[0]
    print(f'schema size: {ss}')

    drt = get_drt(conn)

    return {'natt': natt, 'nfk': nfk, 'cos': cos, 'ss': ss, 'drt': drt}


def data_complexity(spider_dbs):

    for_plot = {}
    
    for i, db_path in enumerate(spider_dbs):
        conn = sqlite3.connect(db_path)
        db_name = db_path.split(os.sep)[-1].split('.')[0] 
        # ['concert_singer', 'department_management', 'musical']
        if db_name in ['concert_singer', 'department_management', 'musical']:
            print(db_name)
            for_plot[db_name] = calculate_metrics(conn)


    print(for_plot)
    csv_file = "spider_data_complexity.csv"
    import csv
    try:
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['db_name', 'natt', 'nfk', 'cos', 'ss', 'drt'])
            for db_name, metrics in for_plot.items():
                metrics = [db_name] + list(metrics.values())
                writer.writerow(metrics)
    except IOError:
        print("I/O error")
    
    return for_plot



def plot(for_plot):
    data = []
    for db_name, metrics in for_plot.items():
        each = [db_name] + list(metrics.values())
        data.append(each)
    
    data = sorted(data, key=lambda x: x[0])

    # Create data frame
    data = pd.DataFrame(data, columns=['db_name', 'natt', 'nfk', 'cos', 'ss', 'drt'])

    # Normalize the metrics
    data['NA'] = data['natt']*100 / sum(data['natt'])
    data['NFK'] = data['nfk'] *100 / sum(data['nfk'])
    data['COS'] = data['cos'] *100 / sum(data['cos'])
    data['SS'] = data['ss'] *100 / sum(data['ss'])
    data['DRT'] = data['drt'] *100 / sum(data['drt'])

    # Create subplots
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))

    # Plot histograms of the metrics
    axs[0].hist(data[['NA', 'NFK', 'COS', 'SS', 'DRT']].values, bins=6, label=['Number of Attributes (NA)', 'Number of Foreign Keys (NFK)', 'Cohesion of the Schema (COS)', 'Schema Size (SS)', 'Depth Referential Tree (DRT)'], color=['red', 'green', 'yellow',  'black', 'blue'])
    axs[0].legend()
    axs[0].set_xlabel('Data Complexity')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Histogram of Data Complexity')

    # Plot scatter diagram of the metrics
    tick_labels = data['db_name'][::1]
    colors = ['red', 'green', 'yellow',  'black', 'blue']
    metrics = ['NA', 'NFK', 'COS', 'SS', 'DRT']
    for i, metric in enumerate(metrics):
        axs[1].scatter(np.arange(len(data))[::1], data[metric][::1], color=colors[i], label=metric)
    axs[1].set_xlabel('Relational Database Name')
    axs[1].set_xticks(np.arange(len(data))[::1])
    axs[1].set_xticklabels(tick_labels, rotation=0)
    axs[1].set_ylabel('Normalized Data Complexity')
    axs[1].set_title('Scatter Diagram of Normalized Data Complexity')
    axs[1].legend()

    fig.tight_layout()
    fig.show()

    # Save the figure
    fig.savefig('my_plot.png', dpi=300)


def main():

    import glob, argparse
    import configparser
    config = configparser.ConfigParser()
    config.read('../config.ini')
    filenames = config["FILENAMES"]

    raw_folder = filenames['raw_folder']
    raw_spider_folder = os.path.join(raw_folder, 'spider')
    db_folder = os.path.join(raw_spider_folder,  'database')
    spider_dbs = glob.glob(db_folder + '/**/*.sqlite', recursive = True)

    metric = data_complexity(spider_dbs)
    print(metric)
    plot(metric)


if __name__ == "__main__":
    Fire(main)