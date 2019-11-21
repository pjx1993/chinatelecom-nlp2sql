#!/usr/bin/env python
# coding: utf-8

import sys

sys.path.append('../')
from tool.question_label import *

header=["品牌","出厂时间","是否支持5G","是否支持volte","是否支持全网通","手机","价格","评分"]

#print(question_label("评分超85分且价格在6000以内的手机有哪个，是否支持volte","select 手机, 是否支持volte from table_device where 评分>'85' and 价格<'6000'",header))
#{"question": "评分超85分且价格在6000以内的手机有哪个，是否支持volte", "table_id": "table_device", "sql": {"agg": [0, 0], "cond_conn_op": 1, "sel": [5, 3], "conds": [[7, 0, "85"], [6, 1, "6000"]]}}

#print(question_label("支持volte的三星手机平均价格是","select avg(价格) from table_device where 是否支持volte='支持volte' and 品牌='三星'",header))
#{"question": "支持volte的三星手机平均价格是", "table_id": "table_device", "sql": {"agg": [1], "cond_conn_op": 1, "sel": [6], "conds": [[3, 2, "支持volte"], [0, 2, "三星"]]}}

#print(question_label("2019年3月出厂的全网通手机有哪些，价格是多少？","select 手机, 价格 from table_device where 出厂时间='2019年3月' and 是否支持全网通='支持全网通'",header))
#{"question": "2019年3月出厂的全网通手机有哪些，价格是多少？", "table_id": "table_device", "sql": {"agg": [0, 0], "cond_conn_op": 1, "sel": [5, 6], "conds": [[1, 2, "2019年3月"], [4, 2, "支持全网通"]]}}

#print(question_label("评分超过80的手机最便宜为","select min(价格) from table_device where 评分>'80'",header))
#{"question": "评分超过80的手机最便宜为", "table_id": "table_device", "sql": {"agg": [3], "cond_conn_op": 0, "sel": [6], "conds": [[7, 0, "80"]]}}

#print(question_label("价格超4000的vivo手机有哪些，评分是？","select 手机, 评分 from table_device where 价格>'4000' and 品牌='vivo'",header))
#{"question": "价格超4000的vivo手机有哪些，评分是？", "table_id": "table_device", "sql": {"agg": [0, 0], "cond_conn_op": 1, "sel": [5, 7], "conds": [[6, 0, "4000"], [0, 2, "vivo"]]}}

#print(question_label("2019年3月出厂支持全网通的手机有多少","select count(手机) from table_device where 出厂时间='2019年3月' and 是否支持全网通='支持全网通'",header))
#{"question": "2019年3月出厂支持全网通的手机有多少", "table_id": "table_device", "sql": {"agg": [4], "cond_conn_op": 1, "sel": [5], "conds": [[1, 2, "2019年3月"], [4, 2, "支持全网通"]]}

#print(question_label("支持volte的vivo手机评分最高是多少","select max(评分) from table_device where 品牌='vivo' and 是否支持volte='支持volte'",header))
#{"question": "支持volte的vivo手机评分最高是多少", "table_id": "table_device", "sql": {"agg": [2], "cond_conn_op": 1, "sel": [7], "conds": [[0, 2, "vivo"], [3, 2, "支持volte"]]}}

#print(question_label("支持volte的vivo手机评分最高是多少","select max(评分) from table_device where 品牌='vivo' and 是否支持volte='支持volte'",header))
#{"question": "支持volte的vivo手机评分最高是多少", "table_id": "table_device", "sql": {"agg": [2], "cond_conn_op": 1, "sel": [7], "conds": [[0, 2, "vivo"], [3, 2, "支持volte"]]}}

#print(question_label("支持5G的手机最便宜的是多少钱","select min(价格) from table_device where 是否支持5G='支持5G'",header))
#{"question": "支持5G的手机最便宜的是多少钱", "table_id": "table_device", "sql": {"agg": [3], "cond_conn_op": 0, "sel": [6], "conds": [[2, 2, "支持5G"]]}}
