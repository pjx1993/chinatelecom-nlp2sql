#!/usr/bin/env python
# coding: utf-8

import sys

sys.path.append('../')
from tool.question_label import *

header=["任务名称", "子任务", "开始时间", "需求处室"]


#print(question_label("应用处10点的任务有多少个","select count(任务名称) from work_timeline where 启始时间='10点' and 需求处室='应用处'",header))
#{"question": "应用处10点的任务有多少个", "table_id": "work_timeline", "sql": {"agg": [4], "cond_conn_op": 1, "sel": [0], "conds": [[2, 2, "10点"], [3, 2, "应用处"]]}}

#print(question_label("管理处任务数量超过10的任务名称为，开始时间是？","select 任务名称, 启始时间 from work_timeline where 任务数量>'10' and 需求处室='管理处'",header))
#{"question": "管理处任务数量超过10的任务名称为，开始时间是？", "table_id": "work_timeline", "sql": {"agg": [0, 0], "cond_conn_op": 1, "sel": [0, 2], "conds": [[1, 0, "10"], [3, 2, "管理处"]]}}


#print(question_label("平台处的单个任务的平均任务数据量为多少个","select avg(任务数量)  from work_timeline where 需求处室='平台处' ",header))
#{"question": "平台处的单个任务的平均任务数据量为多少个", "table_id": "work_timeline", "sql": {"agg": [1], "cond_conn_op": 0, "sel": [1], "conds": [[3, 2, "平台处"]]}}

#print(question_label("11点开始子任务超过5的任务名称为，需求处室是？","select 任务名称, 需求处室 from work_timeline where 子任务>'5' and 开始时间='11点'",header))
#{"question": "11点开始子任务超过5的任务名称为，需求处室是？", "table_id": "worktimeline", "sql": {"agg": [0, 0], "cond_conn_op": 1, "sel": [0, 3], "conds": [[1, 0, "5"], [2, 2, "11点"]]}}

#print(question_label("管理处11点的任务有多少个","select count(任务名称) from work_timeline where 开始时间='11点' and 需求处室='管理处'",header))
#{"question": "管理处11点的任务有多少个", "table_id": "work_timeline", "sql": {"agg": [4], "cond_conn_op": 1, "sel": [0], "conds": [[2, 2, "11点"], [3, 2, "管理处"]]}}

print(question_label("平台处子任务超过5的任务名称为，开始时间是？","select 任务名称, 开始时间 from work_timeline where 子任务>'5' and 需求处室='平台处'",header))
#{"question": "管理处任务数量超过10的任务名称为，开始时间是？", "table_id": "work_timeline", "sql": {"agg": [0, 0], "cond_conn_op": 1, "sel": [0, 2], "conds": [[1, 0, "10"], [3, 2, "管理处"]]}}
