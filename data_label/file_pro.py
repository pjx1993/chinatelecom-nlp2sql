#!/usr/bin/env python
# coding: utf-8


import os
import re

row_line=[]
rows_line=[]
#file = open('raw_data/5g_result.csv', 'r')
#file = open('raw_data/result_device.csv', 'r')
#file = open('raw_data/inter_product.csv', 'r')
#file = open('raw_data/work_timeline.csv', 'r')
file = open('raw_data/tel_result.csv', 'r')
lines = file.readlines()
for line in lines:
  row_line=[]
  vals = line.strip().split(',')
  for val in vals:
    print(val)
    row_line.append(val)
  rows_line.append(row_line)

#header=["品牌","出厂时间","是否支持5G","是否支持volte","是否支持全网通","手机","价格","评分"]

#inter_json = {'rows':rows_line, "name": "table_inter", "title": "", "header": ["账期","省份","产品","用户数","户均ARPU"], "common": "", "id": "inter", "types": ["text", "text", "text", "real", "real"]}
#g_json = {'rows':rows_line, "name": "table_5G", "title": "", "header": ["省份","总计","新发展","终端销量" ,"日期"], "common": "", "id": "5g", "types": ["text","real","real", "real", "text"]}
#device_json = {'rows':rows_line, "name": "table_deivce", "title": "", "header": ["品牌","出厂时间","是否支持5G","是否支持volte","是否支持全网通","手机","价格","评分"], "common": "", "id": "device", "types": ["text", "text", "text", "text", "text","real","real"]}
#work_json = {'rows':rows_line, "name": "table_worktimeline", "title": "", "header": ["任务名称","任务数量","启始时间","需求处室"], "common": "", "id": "worktimeline", "types": ["text","real","text", "text"]}
telbill_json = {'rows':rows_line, "name": "table_telbill", "title": "", "header": ["账期","流量（Gb）","语音（分钟）","话费（元）"], "common": "", "id": "telbill", "types": ["text","real","real", "real"]}
import json
#print(json.dumps(g_json,ensure_ascii=False))
print(json.dumps(telbill_json,ensure_ascii=False))

#print(json.dumps(inter_json,ensure_ascii=False))
