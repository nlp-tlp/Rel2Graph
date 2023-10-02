from unsw.AbsSQL import AbsSQL
from unsw.Operator import Operator


class Where(AbsSQL):
    def __init__(self, cypher):
        self.op = Operator()
        super().__init__(cypher)

    def handle_sql(self, data):
        for key in data:
            if str(key).lower() == 'and':
                # handle and case
                self.cypher += self.handle_and_or_case(data[key])
            elif str(key).lower() == 'or':
                # handle or case
                self.cypher += self.handle_and_or_case(data[key], keyword='OR')
            elif str(key).lower() == 'in':
                # handle in case
                self.cypher += self.handle_in_case(data[key])

    def handle_and_or_case(self, data, keyword="AND"):
        """
        handle and case for the where condition
        :param data: list
        :param operators: the operations, eg: >, <, >=, <=, =, !=
        :param keyword: AND, OR
        :return:
            A string of query language
        """
        result = ""
        for case in data:
            if result != "":
                result += keyword
            for key, value in case.items():
                # get the operation first
                ope = self.op.get_operator(key)
                # value is a list type
                values = []
                find_literal = False
                for v in value:
                    if type(v) != dict:
                        values.append(v)
                    else:
                        values.append(v['literal'])
                        find_literal = True
                result += " {} {} {} ".format(values[0], ope, values[1]) if not find_literal \
                    else " {} {} '{}' ".format(values[0], ope, values[1])
        return result

    def handle_in_case(self, data):
        """
        handle in case in where conditions
        eg: {'in': ['p.ProductName', {'literal': ['a', 'b']}]}
        ['p.ProductName', [1, 2]]}
        when index = 0, it is a keyword
        :param data: dict type data
        :return:
            A string
        """
        values = []
        result = "{} IN [".format(data[0])
        is_literal = False
        for i in range(1, len(data)):
            if type(data[i]) != dict:
                values = [i for i in data[i]]
            else:
                is_literal = True
                values = [i for i in data[i]['literal']]

        if is_literal:
            result += ", ".join("'{}'".format(i) for i in values)
        else:
            result += ", ".join(str(i) for i in values)

        return result + "] "
