CREATE TABLE table_5g (
   5g_id INT UNSIGNED AUTO_INCREMENT,
   省份 VARCHAR(40) NOT NULL,
   总计 INTEGER NOT NULL,
   终端销量 INTEGER NOT NULL,
   新增 INTEGER NOT NULL,
   日期 VARCHAR(40) NOT NULL,
   PRIMARY KEY ( 5g_id )
)ENGINE=InnoDB DEFAULT CHARSET=utf8;


LOAD DATA INFILE '/Users/zhuangzhuanghuang/Code/superset/5g_result.csv'
INTO TABLE table_5g
FIELDS TERMINATED BY '\,'
LINES TERMINATED BY '\n'
(省份,总计,终端销量,新增,日期)
SET 5g_id = NULL;

CREATE TABLE table_device (
   device_id INT UNSIGNED AUTO_INCREMENT,
   品牌 VARCHAR(40) NOT NULL,
   出厂时间 VARCHAR(40) NOT NULL,
   是否支持volte VARCHAR(40) NOT NULL,
   是否全网通 VARCHAR(40) NOT NULL,
   手机 VARCHAR(40) NOT NULL,
   价格 INTEGER NOT NULL,
   评分 DOUBLE NOT NULL,
   PRIMARY KEY ( device_id )
)ENGINE=InnoDB DEFAULT CHARSET=utf8;

LOAD DATA INFILE '/Users/zhuangzhuanghuang/Code/superset/result_device.csv'
INTO TABLE table_device
FIELDS TERMINATED BY '\,'
LINES TERMINATED BY '\n'
(品牌,出厂时间,是否支持volte,是否全网通,手机,价格,评分)
SET device_id = NULL;

CREATE TABLE table_inter (
   inter_id INT UNSIGNED AUTO_INCREMENT,
   账期 VARCHAR(40) NOT NULL,
   省份 VARCHAR(40) NOT NULL,
   产品 VARCHAR(40) NOT NULL,
   用户数 INTEGER NOT NULL,
   户均ARPU DOUBLE NOT NULL,
   PRIMARY KEY ( inter_id )
)ENGINE=InnoDB DEFAULT CHARSET=utf8;

LOAD DATA INFILE '/Users/zhuangzhuanghuang/Code/superset/inter_product.csv'
INTO TABLE table_inter
FIELDS TERMINATED BY '\,'
LINES TERMINATED BY '\n'
(账期,省份,产品,用户数,户均ARPU)
SET inter_id = NULL;


CREATE TABLE work_timeline (
   work_id INT UNSIGNED AUTO_INCREMENT,
   任务名称 VARCHAR(40) NOT NULL,
   任务数量 INTEGER(40) NOT NULL,
   启始时间 VARCHAR(40) NOT NULL,
   需求处室 VARCHAR(40) NOT NULL,
   PRIMARY KEY ( work_id )
)ENGINE=InnoDB DEFAULT CHARSET=utf8;

LOAD DATA INFILE '/Users/zhuangzhuanghuang/Code/superset/work_timeline.csv'
INTO TABLE work_timeline
FIELDS TERMINATED BY '\,'
LINES TERMINATED BY '\n'
(任务名称,任务数量,启始时间,需求处室)
SET work_id = NULL;
