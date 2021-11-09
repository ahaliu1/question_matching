'''
@Descripttion: 
@version: 
@Author: Tingyu Liu
@Date: 2020-07-18 16:11:11
LastEditors: Tingyu Liu
LastEditTime: 2021-11-09 11:55:23
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


def translate(zh_word: str,fromLang='zh',toLang='en') -> list:
    appid = '20200709000516066'  # 填写你的appid
    secretKey = 'FXakDKRr6kdgdTtMuoDd'  # 填写你的密钥

    httpClient = None
    myurl = '/api/trans/vip/translate'

    # fromLang = 'zh'  # 原文语种
    # toLang = 'en'  # 译文语种
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
    """
    Reads data.
    is_test True 测试数据，也可以用于两列的数据
    """
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
    # 中文翻英文
    # train_data = read_text_pair("/home/lty/code/question_matching/data/train.txt")
    # request_str = ''
    # count = 0
    # id_store = []
    # with open("./data/en/train_en.txt", 'ab+',buffering=0) as f:

    #     for i,line in enumerate(train_data):
    #        # if i < 769500:
    #        #     continue
    #         q1 = line['query1']
    #         q2 = line['query2']
    #         request_str = request_str + "\n" +q1
    #         id_store.append(i)
    #         count += 1
    #         # test
    #         # request_str = ['今天天气很好，我非常开心','今天天气很好，我非常开心']
    #         if count == 66:
    #             en = translate(request_str)
    #             assert len(en) == count
    #             for id,line in zip(id_store,en):
    #                 f.write(str(id+1).encode('utf-8'))
    #                 f.write(b'\t')
    #                 f.write(line.encode('utf-8'))
    #                 f.write(b'\n')
    #             count = 0
    #             request_str=''
    #             id_store.clear()
    #             # 降低访问频率
    #             time.sleep(1)
    #     f.close()

    # 英翻译中文
    train_data = read_text_pair("/home/lty/code/question_matching/data/en/train_en.txt",is_test=True)
    request_str = ''
    count = 0
    id_store = []
    with open("./data/en/train_en_ch.txt", 'ab+',buffering=0) as f:
        for i,line in enumerate(train_data):
            q = line['query2']
            request_str = request_str + "\n" +q
            id_store.append(i)
            count += 1
            # test
            # request_str = ['今天天气很好，我非常开心','今天天气很好，我非常开心']
            if count == 50:
                en = translate(request_str,fromLang='en',toLang='zh')
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
