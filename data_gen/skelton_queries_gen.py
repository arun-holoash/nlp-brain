import csv
from core import project_constants

## Class to generate more skelton queries for the project

__author__ = 'Arun Bhatia'

class SkeltonQuery:

    def __init__(self):
        pass


    def generate_skelton_queries(self, list_sen):
        """
        Generate and add more skelton queries to the existing ones
        :return:
        """

        previous_list_sen = self.__previous_queries_count()

        self.__make_previous_duplicate_copy(previous_list_sen)

        list_sen.extend(previous_list_sen)

        self.__final_queries(list_sen)

    def __final_queries(self, list_sen):
        """
        Make final csv and write all queries into it
        :return:
        """
        list_sen = list(set(list_sen))

        print("New count of queries: {}".format(len(list_sen)))

        with open(project_constants.MAIN_SKELTON_QUERIES_PATH, 'w+') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for sen in list_sen:
                tokens = sen.split(",")
                writer.writerow([tokens[0], tokens[1]])


    def __make_previous_duplicate_copy(self, list_sen):
        """
        Make a duplicate copy for the skelton file
        :return:
        """
        with open(project_constants.DUP_SKELTON_QUERY_PATH, 'w+') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for sen in list_sen:
                tokens = sen.split(",")
                writer.writerow([tokens[0], tokens[1]])

    def add_more_data(self, new_queries, intent):
        """
        Add intents to the new queries
        :param new_queries:
        :param intent:
        :return:
        """
        return [query + "," + intent for query in new_queries]

    def __previous_queries_count(self):
        """
        Extract previous queries and return the list
        :return:
        """
        list_sen = []
        with open(project_constants.MAIN_SKELTON_QUERIES_PATH, 'r+') as skelton_queries_file:
            reader = csv.reader(skelton_queries_file, delimiter=',')
            for row in reader:
                list_sen.append(','.join([token for token in row]))

        print("Previous count of queries: {}".format(len(list_sen)))
        return list_sen


