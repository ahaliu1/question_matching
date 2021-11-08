'''
@Descripttion: 
@version: 
@Author: Tingyu Liu
@Date: 2020-07-18 16:11:11
LastEditors: Tingyu Liu
LastEditTime: 2021-11-08 14:40:22
'''
# 百度通用翻译API,不包含词典、tts语音合成等资源，如有相关需求请联系translate_api@baidu.com
# coding=utf-8

import http.client
import hashlib
import urllib
import random
import json
import pymysql
import time


def translate(zh_word: str) -> list:
    appid = '20200709000516066'  # 填写你的appid
    secretKey = 'FXakDKRr6kdgdTtMuoDd'  # 填写你的密钥

    httpClient = None
    myurl = '/api/trans/vip/translate'

    fromLang = 'zh'  # 原文语种
    toLang = 'en'  # 译文语种
    salt = random.randint(32768, 65536)
    q = zh_word
    sign = appid + q + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    # myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
    #     salt) + '&sign=' + sign

    try:
        httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
        # httpClient.request('GET', myurl)
        # use post
        body = 'appid=' + appid + '&q=' + urllib.parse.quote(
            q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(salt) + '&sign=' + sign
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"}

        httpClient.request(method='POST', url=myurl,
                           body=body, headers=headers)

        # httpClient.request('POST', myurl)
        # Content-Type,application/x-www-form-urlencoded
       # response是HTTPResponse对象
        response = httpClient.getresponse()
        result = response.read().decode("utf-8")
        res = json.loads(result)['trans_result']
        en_list = []
        for r in res:
            en_list.append(r['dst'])
        return en_list

    except Exception as e:
        print("翻译出现错误:"+str(e))
    finally:
        if httpClient:
            httpClient.close()
    pass

def read_text_pair(data_path, is_test=False):
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            if is_test == False:
                if len(data) != 3:
                    continue
                yield {'query1': data[0], 'query2': data[1], 'label': data[2]}
            else:
                if len(data) != 2:
                    continue
                yield {'query1': data[0], 'query2': data[1]}


if __name__ == '__main__':

    train_data = read_text_pair("/home/lty/code/question_matching/data/train.txt")
    request_str = ''
    count = 0
    id_store = []
    with open("./data/en/train_en.txt", 'ab+',buffering=0) as f:

        for i,line in enumerate(train_data):
            if i < 490700:
                continue
            q1 = line['query1']
            q2 = line['query2']
            request_str = request_str + "\n" +q1
            id_store.append(i)
            count += 1
            # test
            # request_str = ['今天天气很好，我非常开心','今天天气很好，我非常开心']
            if count == 100:
                en = translate(request_str)
                assert len(en) == count
                for id,line in zip(id_store,en):
                    f.write(str(id+1).encode('utf-8'))
                    f.write(b'\t')
                    f.write(line.encode('utf-8'))
                    f.write(b'\n')
                count = 0
                request_str=''
                id_store.clear()
                # 降低访问频率
                time.sleep(1)


        f.close()
        






    # db = pymysql.connect("193.112.77.221", "yxl",
    #                      "123456", "pcl_artical_search")
    # cursor = db.cursor()
    # cursor_insert = db.cursor()
    # trans_list = []
    # record_list = []
    # insert_list = []
    # start = 0
    # batch_count = 0
    # # 大概有35w条三元组，每次查1w条，循环35次
    # for i in range(35):

    #     try:
    #         sql = "select id,subject,predicate,object,subgraph from openkg_covid19 where id not in (select zh_id from zh_en) limit %s,10000" % start
    #         cursor.execute(sql)
    #         db.commit()
    #         start += 10000  # 放到最后
    #     except:
    #         db.rollback()
    #         break

    #     row = cursor.fetchone()

    #     while(row):
    #         triplet_id = row[0]
    #         sub = row[1]
    #         predicate = row[2]
    #         obj = row[3]
    #         subgraph = row[4]
    #         record_list.append(
    #             [str(triplet_id), sub, predicate, obj, subgraph])

    #         # 三元组内的分割
    #         trans_list.append("\"\n\"".join([sub, predicate, obj]))

    #         row = cursor.fetchone()

    #         # 每n条拼接到一个请求里，使用"\n"分割，分割方法来自api。
    #         trans_batch_num = 40
    #         if len(trans_list) == trans_batch_num:
    #             time.sleep(1)
    #             trans_str = "\"\n\"".join(trans_list)

    #             # en_list trans_batch_num*3*1
    #             en_list = translate(trans_str)

    #             try:
    #                 # 准备插入的元组列表
    #                 for i in range(len(en_list)):
    #                     insert_list.append(
    #                         (en_list[i][0], en_list[i][1], en_list[i][2], record_list[i][4], "baidu_trans", "baidu_trans", "baidu_trans"))
    #                 sql = "insert into openkg_covid19_en (subject,predicate,object,subgraph,subject_uri,predicate_uri,object_uri) values (%s,%s,%s,%s,%s,%s,%s);"
    #                 cursor_insert.executemany(sql, insert_list)
    #                 db.commit()

    #                 # 获取该批插入的第一个id
    #                 batch_1st_id = cursor_insert.lastrowid

    #                 insert_list.clear()
    #                 en_id = batch_1st_id
    #                 for i in range(trans_batch_num):
    #                     insert_list.append((record_list[i][0], en_id))
    #                     en_id += 1

    #                 sql = "insert into zh_en(zh_id,en_id) values(%s,%s);"
    #                 cursor_insert.executemany(sql, insert_list)
    #                 db.commit()
    #                 batch_count += 1
    #                 print(str(batch_count) +
    #                       "batch insert success,last id"+str(en_id))

    #                 trans_list.clear()
    #                 record_list.clear()
    #                 insert_list.clear()
    #             except Exception as e:
    #                 print(e)
    #                 db.rollback()
    #                 trans_list.clear()
    #                 record_list.clear()
    #                 insert_list.clear()