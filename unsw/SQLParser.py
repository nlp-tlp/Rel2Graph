from unsw.OrderBy import OrderBy
from unsw.Select import Select
from unsw.From import From
from unsw.Where import Where
from unsw.Limit import Limit


class SQLParser:
    def __init__(self):
        self.cypher = ""
        self.Orderby = OrderBy("ORDER BY ")
        self.Select = Select("RETURN ")
        self.From = From("MATCH ")
        self.Where = Where("WHERE ")
        self.Limit = Limit("LIMIT ")

    def generate_cypher(self, data, sql):
        if "from" in data:
            if 'join' in sql:
                self.From.handle_join(data)
                self.cypher += self.From.get_cypher()
            else:
                self.From.handle_sql(data['from'])
                self.cypher += self.From.get_cypher()

        if "where" in data:
            self.Where.handle_sql(data['where'])
            self.cypher += self.Where.get_cypher()

        if "select" in data:
            self.Select.handle_sql(data['select'])
            self.cypher += self.Select.get_cypher()

        if 'orderby' in data:
            self.Orderby.handle_sql(data['orderby'])
            self.cypher += self.Orderby.get_cypher()

        if 'limit' in data:
            self.Limit.handle_sql(data['limit'])
            self.cypher += self.Limit.get_cypher()

    def get_cypher(self):
        return self.cypher + ";"

