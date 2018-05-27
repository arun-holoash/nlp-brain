from data_gen.skelton_queries_gen import SkeltonQuery
from data_gen import sentence_generator as sg
from core import project_constants

__author__ = 'Arun Bhatia'

########################
# Use this script to add more data points to the existing file
# Steps:
# 1. add pre-head, head ,body etc. and define an intent
# 2. Define layers in the section below on how to add above configs

########################

sq = SkeltonQuery()

list_sen = []

################################       Manual configuration       ###################################

pre_head = ["please"]

head = ['want to listen', 'in mood to listen']

body = ['music', 'songs', 'song']

tail = []

post_tail = []

intent = project_constants.MUSIC

#################################################################################

# Define layers
list_sen.extend(sq.add_more_data(sg.sentence_gen(pre_head, head, body), intent))
list_sen.extend(sq.add_more_data(sg.sentence_gen(head, body), intent))


######################################################################################
sq.generate_skelton_queries(list_sen)
