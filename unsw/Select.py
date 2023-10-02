from unsw.AbsSQL import AbsSQL


class Select(AbsSQL):

    def handle_sql(self, data):
        """
        handle select form
        :param data: list or dict data
        :return:
        """
        if type(data) == dict:
            if type(data['value']) == dict:
                self.cypher += ",".join(" {} {} ".format(str(key).upper(), str(value).replace('.*', ''))
                                   for key, value in data['value'].items())
            else:
                self.cypher += "{}".format(str(data['value']).replace('.*', ''))
        elif type(data) == list:
            values = [x['value'] for x in data]
            self.cypher += ", ".join(values)
