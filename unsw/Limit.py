from unsw.AbsSQL import AbsSQL


class Limit(AbsSQL):

    def handle_sql(self, data):
        self.check_data(data)
        self.cypher += " {}".format(str(data))
