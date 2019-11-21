from flask import Flask, jsonify
from flask_cors import CORS
from flask import request,render_template
import os
import re
import json
import math
import numpy as np
from tqdm import tqdm_notebook as tqdm

from keras_bert import load_vocabulary, load_trained_model_from_checkpoint, Tokenizer, get_checkpoint_paths

import keras.backend as K
from keras.layers import Input, Dense, Lambda, Multiply, Masking, Concatenate
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils.data_utils import Sequence
from keras.utils import multi_gpu_model
import tensorflow as tf


import sys

sys.path.append('../')


from nl2sql.utils import read_data, read_line,read_tables, SQL, MultiSentenceTokenizer, Query, Question, Table
from nl2sql.utils.optimizer import RAdam
from dbengine import DBEngine
from build_model import construct_model,construct_model2,outputs_to_sqls,SqlLabelEncoder,DataSequence,QuestionCondPairsDataset,QuestionCondPairsDataseq,merge_result,CandidateCondsExtractor,FullSampler



app = Flask(__name__)

app.config['JSON_AS_ASCII'] = False
CORS(app)

headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36'}

@app.route('/')
def api_root():
  return render_template('index.html')

@app.route('/parse_sql', methods=['POST'])
def parse_sql():
  table_id = request.json['table_id']
  question = request.json['question']

  flag_childfind = 0
  if table_id == 'device' :
    matchObj = re.search( r'最(.*)手机是', question, re.M|re.I)
    if matchObj:
      str_match = matchObj.group()
      str_mat_tmp = str_match.replace('手机','')+'多少'
      question = question.replace(str_match,str_mat_tmp)
      flag_childfind = 1
      key_col_index = 5

  if table_id == 'telbill' :
    matchObj = re.search( r'最(.*)账期是', question, re.M|re.I)
    if matchObj:
      str_match = matchObj.group()
      str_mat_tmp = str_match.replace('账期','')+'多少'
      question = question.replace(str_match,str_mat_tmp)
      flag_childfind = 1
      key_col_index = 0

  test_json_line = '{\"question\": \"'+ question+'\",\"table_id\": \"'+table_id+'\"}'
  test_data = read_line(test_json_line, test_tables)
  print(test_json_line)
  test_dataseq = DataSequence(
      data=test_data,
      tokenizer= query_tokenizer,
      label_encoder=label_encoder,
      is_train=False,
      shuffle_header=False,
      max_len=160,
      shuffle=False,
      batch_size=1
  )

  header_lens = np.sum(test_dataseq[0]['input_header_mask'], axis=-1)
  model = models['stage1']
  model2 = models['stage2']
  with graph.as_default():
    preds_cond_conn_op, preds_sel_agg, preds_cond_op = model.predict_on_batch(test_dataseq[0])
    sql = outputs_to_sqls(preds_cond_conn_op, preds_sel_agg, preds_cond_op,header_lens, test_dataseq.label_encoder)
    te_qc_pairs = QuestionCondPairsDataset(test_data,candidate_extractor=CandidateCondsExtractor(share_candidates=True),has_label=False,model_1_outputs=sql)

    te_qc_pairs_seq = QuestionCondPairsDataseq(te_qc_pairs, tokenizer,sampler=FullSampler(), shuffle=False, batch_size=1)
    te_result = model2.predict_generator(te_qc_pairs_seq, verbose=1)

  task2_result = merge_result(te_qc_pairs, te_result, threshold=0.995)
  cond = list(task2_result.get(0, []))
  sql[0]['conds'] = cond

  engine = DBEngine()
  #table_id = json.loads(test_json_line)['table_id']
  header = test_tables.__getitem__(table_id)._df.columns.values.tolist()
  sql_gen = engine.execute(table_id, sql[0]['sel'], sql[0]['agg'], sql[0]['conds'], sql[0]['cond_conn_op'],header)

  if flag_childfind==1 and sql[0]['agg'][0]>0:
    #print(sql[0]['sel'])
    header_index= int(sql[0]['sel'][0])
    
    childcol = header[header_index]
    key_col = header[key_col_index]
    sql_gen = 'select '+key_col + ' from Table_' + table_id + ' where '+childcol+'=( '+sql_gen+' )'

  return jsonify({'task': sql_gen})

def load():
    global test_tables
    test_table_file = '../data/val.tables.json'
    bert_model_path = '../model'
    test_tables = read_tables(test_table_file)
    paths = get_checkpoint_paths(bert_model_path)
    global label_encoder
    label_encoder = SqlLabelEncoder()
    global query_tokenizer
    model,query_tokenizer = construct_model(paths)
    model_path = '../task1_best_model.h5'
    model.load_weights(model_path)
    global tokenizer
    model2, tokenizer = construct_model2(paths)
    model2.load_weights('../model_best_weights.h5')
    global models
    models = {}
    models['stage1'] = model
    models['stage2'] = model2
    global graph
    graph = tf.get_default_graph()



label_encoder = SqlLabelEncoder()

if __name__ == "__main__":
    load()
    app.run(host='0.0.0.0', port=19015)
