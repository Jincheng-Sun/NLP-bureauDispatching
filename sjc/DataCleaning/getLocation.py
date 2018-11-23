import pandas as pd
import time
import re
import requests
from bs4 import BeautifulSoup

def get_location(sentence):
    ###
    # input：    sentence（诉求内容）      type：str
    # output：   地名                     type：list
    # example:
    #
    # >>> sentence = '美兰区三江镇苏寻三村委会仙山村村民'
    # >>> location = get_location(sentence)
    # >>> location
    # ['美兰区', '三江镇', '仙山村']
    ###
    payload = {}
    payload['s'] = sentence
    payload['f'] = 'xml'
    payload['t'] = 'pos'
    response = requests.post("http://127.0.0.1:12345/ltp", data=payload)
    # docker run -d -p 12345:12345 ce0140dae4c0 ./ltp_server --last-stage all
    soup = BeautifulSoup(response.text, 'html.parser')
    # print(soup)
    word_tags = soup.findAll('word')

    ns_bag=[]
    for word in word_tags:
        loc = word['cont']
        if (loc == '海南'):
            if not loc in ns_bag:
                ns_bag.append(loc+'省')
                continue
        if (loc == '海口'):
            if not loc in ns_bag:
                ns_bag.append(loc+'市')
                continue
        if (word['pos'] == 'ns'):

            if ('省' in loc or '市'in loc or '区'in loc or '县'in loc or '镇'in loc or '村'in loc or '岛'in loc or '路' in loc):
                if not loc in ns_bag:
                    ns_bag.append(loc)
    return ns_bag

# csvfile=pd.read_csv('../datas/testset_with_label.csv',encoding='utf-8',skiprows=1)
# new=[]
# for i in range(0,200):
#     extract_location = str(get_location(csvfile.loc[i]['诉求内容']))
#     real_location = csvfile.loc[i]['地点']
#     new.append([extract_location,real_location])
#
#     if (i%100==0):
#         print(i)
#
#
# new=pd.DataFrame(new,columns=['提取地点','真实地点'])
# new.to_csv('compare.csv',encoding='utf-8')