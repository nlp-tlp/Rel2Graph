
class Operator:
    def __init__(self):
        self.ope = {
            'neq': '!=',
            'eq': '=',
            'gt': '>',
            'gte': '>=',
            'lt': '<',
            'lte': '<='
        }

    def get_operator(self, op):
        if op not in self.ope:
            raise ValueError("Please make sure the operator exist")
        return self.ope[str(op)]
