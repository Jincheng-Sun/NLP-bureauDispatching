import csv
import requests
from bs4 import BeautifulSoup
from itertools import islice
import pickle

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
    str = ""
    for word in word_tags:

        loc = word['cont']
        if (loc == '海南'):
            if not loc in ns_bag:
                ns_bag.append(loc+'省')
                str=str+loc+'省'
                continue
        if (loc == '海口'):
            if not loc in ns_bag:
                ns_bag.append(loc+'市')
                str = str + loc + '市'
                continue
        if (word['pos'] == 'ns'):

            if ('省' in loc or '市'in loc or '区'in loc or '县'in loc or '镇'in loc or '村'in loc or '岛'in loc or '路' in loc):
                if not loc in ns_bag:
                    ns_bag.append(loc)
                    str = str + loc

    return ns_bag,str

def create_pickle():
    dict={}
    csvfile=csv.reader(open('../datas/4w_trainset_all.csv',encoding='gb18030'))
    pkfile=open('../datas/4w_locations.pickle','wb')
    i=0
    for line in islice(csvfile, 1, None):
        vec_id = line[0]
        sentense=line[6]
        _,sentense=get_location(sentense)
        dict[vec_id]=sentense
        i+=1
        if i %100==0:
            print(i)

    pickle.dump(dict,pkfile,pickle.HIGHEST_PROTOCOL)
    pkfile.close()


def read(id,filepath):

    with open(filepath, 'rb') as itf:
        pkfile = pickle.load(itf)

        a=pkfile['6']



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
create_pickle()
# sentence='龙昆南海德路昌茂花园旁边工地在超时施工，产生噪音，严重影响居民休息，请城管核实处理！谢谢！（请职能局按规定在30分钟内联系市民，响应处置）'
# _,a=get_location(sentence)
# print(a)