#!/usr/bin/env python
# coding: utf-8

import sys

sys.path.append('../')
from tool.question_label import *

header=["账期","流量（Gb）","语音（分钟）","话费（元）"]


#print(question_label("流量超过20G的账期是？","select 账期 from table_telbill where 流量（Gb）> 20 ",header))
#{"question": "流量超过20G的账期是？", "table_id": "table_telbill", "sql": {"agg": [0], "cond_conn_op": 0, "sel": [0], "conds": [[1, 0, 20]]}}

#print(question_label("2019年3月和2019年4月的话费总和是？","select sum(话费（元）) from table_telbill where 账期='2019年3月' or 账期='2019年4月' ",header))
#{"question": "2019年3月和2019年4月的话费总和是？", "table_id": "table_telbill", "sql": {"agg": [5], "cond_conn_op": 2, "sel": [3], "conds": [[0, 2, "2019年3月"], [0, 2, "2019年4月"]]}

#print(question_label("2018年7月的语音通话是多少分钟？","select 语音（分钟） from table_telbill where 账期='2018年7月' ",header))
#{"question": "2018年7月的语音通话是多少分钟？", "table_id": "table_telbill", "sql": {"agg": [0], "cond_conn_op": 0, "sel": [2], "conds": [[0, 2, "2018年7月"]]}}

#print(question_label("语音通话超过100分钟的账期是？","select 账期 from table_telbill where 语音（分钟）> 100 ",header))
#{"question": "语音通话超过100分钟的账期是？", "table_id": "table_telbill", "sql": {"agg": [0], "cond_conn_op": 0, "sel": [0], "conds": [[2, 0, 100]]}}

#print(question_label("2018年3月和2018年4月的流量总和是？","select sum(流量（Gb）) from table_telbill where 账期='2018年3月' or 账期='2018年4月' ",header))
#{"question": "2018年3月和2018年4月的流量总和是？", "table_id": "table_telbill", "sql": {"agg": [5], "cond_conn_op": 2, "sel": [1], "conds": [[0, 2, "2018年3月"], [0, 2, "2018年4月"]]}}

print(question_label("2018年3月的流量是多少G？","select 流量（Gb） from table_telbill where 账期='2018年3月' ",header))
#{"question": "2018年7月的语音通话是多少分钟？", "table_id": "table_telbill", "sql": {"agg": [0], "cond_conn_op": 0, "sel": [2], "conds": [[0, 2, "2018年7月"]]}}
