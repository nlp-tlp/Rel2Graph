from unsw.AbsSQL import AbsSQL


class OrderBy(AbsSQL):

    def handle_sql(self, data):
        """
        handle order by
        :param data: list or dict which need to be order
        :return:
            A string
        """
        self.check_data(data)

        # means only one order by
        if type(data) == dict:
            self.cypher += " {} {}".format(data['value'], data['sort'].upper())
        else:
            self.cypher += ",".join(" {} {}".format(order['value'], order['sort'].upper()) for order in data)
