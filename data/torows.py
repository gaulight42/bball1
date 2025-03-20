# -*- python -*-

# Real simple. Just load up the columns and save em as rows.
#
# jhndrsn@acm.org

import json
import sys

column_dict = json.load(open(sys.argv[1]))

columns = list(column_dict.keys())

row_index = column_dict[columns[0]].keys()

for row in row_index:
    item = {}
    for column in columns:
        item[column]=column_dict[column][row]
    print(json.dumps(item))
