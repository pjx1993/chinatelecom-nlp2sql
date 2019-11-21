#!/usr/bin/env python
# coding: utf-8

import sys

sys.path.append('../')
from tool.question_label import *

header=["账期","省份","产品","用户数","户均ARPU"]

#print(question_label("四川的用户数总计有多少","select sum(用户数) from table_inter where 省份='四川'",header))
#{"question": "四川的用户数总计有多少", "table_id": "table_inter", "sql": {"agg": [5], "cond_conn_op": 0, "sel": [3], "conds": [[1, 2, "四川"]]}}

#print(question_label("江苏的互联网专线平均ARP有多少","select avg(户均ARPU) from table_inter where 省份='江苏' and 产品='互联网专线'",header))
#{"question": "江苏的互联网专线平均ARP有多少", "table_id": "table_inter", "sql": {"agg": [1], "cond_conn_op": 1, "sel": [4], "conds": [[1, 2, "江苏"], [2, 2, "互联网专线"]]}}

#print(question_label("2019年6月组网专线的最大户均ARPU是多少","select max(户均ARPU) from table_inter where 账期='2019年6月' and 产品='组网专线'",header))
#{"question": "2019年6月组网专线的最大户均ARPU是多少", "table_id": "table_inter", "sql": {"agg": [2], "cond_conn_op": 1, "sel": [4], "conds": [[0, 2, "2019年6月"], [2, 2, "组网专线"]]}}

#print(question_label("2019年7月互联网专线各省最大发展用户数为","select max(用户数) from table_inter where 账期='2019年7月' and 产品='互联网专线'",header))
#{"question": "2019年7月互联网专线各省最大发展用户数为", "table_id": "table_inter", "sql": {"agg": [2], "cond_conn_op": 1, "sel": [3], "conds": [[0, 2, "2019年7月"], [2, 2, "互联网专线"]]}}

#print(question_label("2019年7月组网专线的最小户均ARPU是多少","select min(户均ARPU) from table_inter where 账期='2019年7月' and 产品='组网专线'",header))
#{"question": "2019年7月组网专线的最小户均ARPU是多少", "table_id": "table_inter", "sql": {"agg": [3], "cond_conn_op": 1, "sel": [4], "conds": [[0, 2, "2019年7月"], [2, 2, "组网专线"]]}}

#print(question_label("2019年6月组网专线各省最少发展用户数为","select min(用户数) from table_inter where 账期='2019年6月' and 产品='组网专线'",header))
#{"question": "2019年6月组网专线各省最少发展用户数为", "table_id": "table_inter", "sql": {"agg": [3], "cond_conn_op": 1, "sel": [3], "conds": [[0, 2, "2019年6月"], [2, 2, "组网专线"]]}}

