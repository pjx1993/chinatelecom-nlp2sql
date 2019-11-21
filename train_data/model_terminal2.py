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

from nl2sql.utils import read_data, read_line,read_tables, SQL, MultiSentenceTokenizer, Query, Question, Table
from nl2sql.utils.optimizer import RAdam
from dbengine import DBEngine
from build_model import construct_model,construct_model2,outputs_to_sqls,SqlLabelEncoder,DataSequence,QuestionCondPairsDataset,QuestionCondPairsDataseq,merge_result,CandidateCondsExtractor,FullSampler




test_table_file = './data/val.tables.json'
bert_model_path = './model'
test_tables = read_tables(test_table_file)
paths = get_checkpoint_paths(bert_model_path)
model,query_tokenizer = construct_model(paths)
model_path = 'task1_best_model.h5'
model.load_weights(model_path)
model2, tokenizer = construct_model2(paths)
model2.load_weights('model_best_weights.h5')

label_encoder = SqlLabelEncoder()

test_json_line = '{"question": "长沙2011年平均每天成交量是3.17，那么近一周的成交量是多少", "table_id": "69cc8c0c334311e98692542696d6e445", "sql": {"agg": [0], "cond_conn_op": 1, "sel": [5], "conds": [[1, 2, "3.17"], [0, 2, "长沙"]]}}'



test_data = read_line(test_json_line, test_tables)

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
preds_cond_conn_op, preds_sel_agg, preds_cond_op = model.predict_on_batch(test_dataseq[0])
sql = outputs_to_sqls(preds_cond_conn_op, preds_sel_agg, preds_cond_op,header_lens, test_dataseq.label_encoder)
te_qc_pairs = QuestionCondPairsDataset(test_data,
                                       candidate_extractor=CandidateCondsExtractor(share_candidates=True),
                                       has_label=False,
                                       model_1_outputs=sql)

te_qc_pairs_seq = QuestionCondPairsDataseq(te_qc_pairs, tokenizer,
                                           sampler=FullSampler(), shuffle=False, batch_size=1)
te_result = model2.predict_generator(te_qc_pairs_seq, verbose=1)

task2_result = merge_result(te_qc_pairs, te_result, threshold=0.995)
cond = list(task2_result.get(0, []))
sql[0]['conds'] = cond

engine = DBEngine()
table_id = json.loads(test_json_line)['table_id']
header = test_tables.__getitem__(table_id)._df.columns.values.tolist()
print(engine.execute(table_id, sql[0]['sel'], sql[0]['agg'], sql[0]['conds'], sql[0]['cond_conn_op'],header))
#print(engine.execute(sql_json['table_id'], sql_json['sql']['sel'], sql_json['sql']['agg'], sql_json['sql']['conds'], sql_json['sql']['cond_conn_op']))

