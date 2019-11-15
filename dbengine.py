# -*- coding:utf-8 -*-
import json
import re

agg_dict = {0:"", 1:"AVG", 2:"MAX", 3:"MIN", 4:"COUNT", 5:"SUM"}
cond_op_dict = {0:">", 1:"<", 2:"==", 3:"!="}
rela_dict = {0:'', 1:' AND ', 2:' OR '}

class DBEngine:


    def execute(self, table_id,select_index, aggregation_index, conditions, condition_relation, header):
        """
        table_id: id of the queried table.
        select_index: list of selected column index, like [0,1,2]
        aggregation_index: list of aggregation function corresponding to selected column, like [0,0,0], length is equal to select_index
        conditions: [[condition column, condition operator, condition value], ...]
        condition_relation: 0 or 1 or 2
        """
        table_id = 'Table_{}'.format(table_id)

        # 条件数>1 而 条件关系为''
        if condition_relation == 0 and len(conditions) > 1:
            return 'Error1'
        # 选择列或条件列为0
        if len(select_index) == 0 or len(conditions) == 0 or len(aggregation_index) == 0:
            return 'Error2'

        condition_relation = rela_dict[condition_relation]

        select_part = ""
        print(header)
        for sel, agg in zip(select_index, aggregation_index):
            #select_str = 'col_{}'.format(sel+1)
            print(sel)
            select_str = header[sel]
            agg_str = agg_dict[agg]
            if agg:
                select_part += '{}({}),'.format(agg_str, select_str)
            else:
                select_part += '({}),'.format(select_str)
        select_part = select_part[:-1]

        where_part = []
        for col_index, op, val in conditions:
            where_part.append('{} {} "{}"'.format(header[col_index], cond_op_dict[op], val))
        where_part = 'WHERE ' + condition_relation.join(where_part)

        query = 'SELECT {} FROM {} {}'.format(select_part, table_id, where_part)
        return query


        # result_set = [tuple(set(i.values())) for i in out]
        #result_set = [tuple(sorted(i.values(), key=lambda x:str(x))) for i in out]
        #return result_set
