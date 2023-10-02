import os
import pickle
from unsw.AbsSQL import AbsSQL


class From(AbsSQL):

    def handle_sql(self, data):
        """
        handle the from data, because it includes many from values
        :param data: dict data or list
        :return:
             a string contain the match data
        """
        self.check_data(data)

        if type(data) == dict:
            self.cypher += "({}:{}) ".format(data['name'], data['value'])
        elif type(data) == list:
            pass

    def load_relation(self):
        filepath = os.getcwd() + "/cache/relation.pickle"
        try:
            files = open(filepath, "rb")
            data = pickle.load(files)
            if type(data) is list:
                return data
        except FileNotFoundError:
            raise FileNotFoundError("You haven not converted the database into cypher yet.")

    def handle_left_join(self, data):
        pass

    def handle_join(self, data):
        """
        handle the join relation
        :param data: the from data, should be the whole from data
        :return: the string
        """
        tables = []
        for t in data['from']:
            if 'join' not in t:
                tables.append(t['value'])
            else:
                tables.append(t['join']['value'])

        relation = self.load_relation()
        print(relation)
        for r in relation:
            key = str(list(r.keys())[0])
            if key in tables:
                if r[key]['to'] in tables:
                    self.cypher += " ({}:{})-[:{}]->({}:{}) ".format(key.lower(), key,
                                                             r[key]['relation'], str(r[key]['to']).lower(), r[key]['to'])
