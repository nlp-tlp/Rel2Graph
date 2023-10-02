"""
init the python file
Author: Hongtai Zhang & Shunyang Li 
Date: 10/10/2022
Description:
    Convert SQL to cypher
"""
import os
import sys
import argparse
from view import app
from view.route import main

from CLI import CLI


if __name__ == '__main__':
    # set the command line
 
    parser = argparse.ArgumentParser(description="Welcome to use sql2cypher for managing your database",
                                     epilog="Enjoy the sql2cypher 🙂")
    parser.add_argument('--translate', '-t', action="store_true", help="Translate SQL query to Cypher query")
    parser.add_argument('--migrate', '-m', action="store_true", help="Migrate RDBMS to Graph database")
    parser.add_argument('--output', '-o', action="store_true", help="Output the cypher query without executing")
    parser.add_argument('--web_ui', '-web', action="store_true", help="Start the web ui model")
    parser.add_argument('--clean_cache', '-cc', action="store_true", help="To clean the cache information")
    parser.add_argument('--database', '-db', help="The specific relational database(mysql or psql). "
                                                  "The default value is mysql")
    parser.add_argument('--load_method', '-lm', help="The load method, provide csv and cypher."
                                                     "The default method is csv and cypher together.")

    args = parser.parse_args()

    if len(sys.argv) < 2:
        # invalid command and args
        parser.print_help()
        sys.exit()

    supports = ['mysql', 'psql']
    if args.migrate:
        db = args.database
        db_name = 'mysql'
        if db is not None and db in supports:
            db_name = db
        output = args.output
        cli = CLI(output, db_name=db_name)
        cli.convert_db()
        sys.exit()

    if args.translate:
        output = args.output
        print(output)
        cli = CLI(output)
        cli.transfer_sql()
        sys.exit()

    # if args.clean_cache is not None:
    #     os.remove(os.getcwd() + '/cache/relation.pickle')
    #     os.rmdir(os.getcwd() + '/data')
    #     print("Clean all the cache and temp files")

    if args.web_ui:
        # print(app.static_folder)
        app.run(debug=True)
    if len(sys.argv) > 2:
        print("Please have a look the help: python3 sql2cypher.py --help")
        exit()
    
    command = sys.argv[1]
    cli = CLI()
    # cli.cb.get_tables()
    if command == "--help":
        cli.help()
    elif command == "-t":
        cli.transfer_sql()
    elif command == "-m":
        cli.convert_db()
    else:
        raise ValueError("Invalid command")

