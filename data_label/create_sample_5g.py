#!/usr/bin/env python
# coding: utf-8

import sys

sys.path.append('../')
from tool.question_label import *

header=["省份","总计","终端销量","新增","日期"]

print(question_label("上海11月11日的终端销量是多少","select 终端销量 from table_5g where 省份= '上海' and 日期='11月11日'",header))
#{"question": "上海11月11日的终端销量是多少", "table_id": "table_5g", "sql": {"agg": [0], "cond_conn_op": 1, "sel": [2], "conds": [[0, 2, "上海"], [4, 2, "11月11日"]]}}

#print(question_label('11月14日各省平均新增是多少','select avg(新增) from table_5g where  日期="11月14日"',header))
#{"question": "11月14日各省平均新增是多少", "table_id": "table_5g", "sql": {"agg": [1], "cond_conn_op": 0, "sel": [3], "conds": [[4, 2, "11月14日"]]}}

#print(question_label('11月14日发展总计超过5000人有多少省份','select count(省份) from table_5g where  日期="11月14日" and 总计>"5000"',header))
#{"question": "11月14日发展总计超过5000人有多少省份", "table_id": "table_5g", "sql": {"agg": [4], "cond_conn_op": 1, "sel": [0], "conds": [[4, 2, "11月14日"], [1, 0, "5000"]]}}

#print(question_label('11月11日终端销量超过2000的省份中,最大新增了多少用户','select max(新增) from table_5g where  日期="11月11日" and 终端销量>"2000"',header))
#{"question": "11月11日终端销量超过2000的省份中,最大新增了多少用户", "table_id": "table_5g", "sql": {"agg": [2], "cond_conn_op": 1, "sel": [3], "conds": [[4, 2, "11月11日"], [2, 0, "2000"]]}}

#print(question_label('江苏平均新增用户量为','select avg(新增) from table_5g where 省份="江苏"',header))
#{"question": "江苏平均新增用户量为", "table_id": "table_5g", "sql": {"agg": [1], "cond_conn_op": 0, "sel": [3], "conds": [[0, 2, "江苏"]]}}

#print(question_label('四川省最小新增用户量为','select min(新增) from table_5g where 省份="四川"',header))
#{"question": "四川省最小新增用户量为", "table_id": "table_5g", "sql": {"agg": [3], "cond_conn_op": 0, "sel": [3], "conds": [[0, 2, "四川"]]}}

#print(question_label('江西省和安徽省总计终端销量为','select sum(终端销量) from table_5g where 省份="山西" or 省份="安徽" ',header))
#{"question": "江西省和安徽省总计终端销量为", "table_id": "table_5g", "sql": {"agg": [5], "cond_conn_op": 2, "sel": [2], "conds": [[0, 2, "山西"], [0, 2, "安徽"]]}}

#print(question_label('11月11和11月14日新增的用户总量为','select sum(新增) from table_5g where 日期="11月11日" or 日期="11月14日" ',header))
#{"question": "11月11和11月14日新增的用户总量为", "table_id": "table_5g", "sql": {"agg": [5], "cond_conn_op": 2, "sel": [3], "conds": [[4, 2, "11月11日"], [4, 2, "11月14日"]]}}

#print(question_label('浙江新增用户小于5000有哪几天','select 日期 from table_5g where 省份="浙江" and 新增<"5000"',header))
#{"question": "浙江新增用户小于5000有哪几天", "table_id": "table_5g", "sql": {"agg": [0], "cond_conn_op": 1, "sel": [4], "conds": [[0, 2, "浙江"], [3, 1, "5000"]]}}

#print(question_label('11月14日终端销量小于一万的有哪些省份,终端销量是多少','select 省份, 终端销量 from table_5g where 日期="11月14日"  and 终端销量<"10000"',header))
#{"question": "11月14日终端销量小于一万的有哪些省份,终端销量是多少", "table_id": "table_5g", "sql": {"agg": [0, 0], "cond_conn_op": 1, "sel": [0, 2], "conds": [[4, 2, "11月14日"], [2, 1, "10000"]]}}

