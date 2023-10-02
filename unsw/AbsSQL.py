from abc import ABC, abstractmethod


class AbsSQL(ABC):
    def __init__(self, cypher):
        self.cypher = cypher
        super().__init__()

    def get_cypher(self):
        return self.cypher + " "

    @staticmethod
    def check_data(data):
        if data is None:
            raise ValueError("Incorrect data")

    @abstractmethod
    def handle_sql(self, data):
        pass
