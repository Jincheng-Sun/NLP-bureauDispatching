import math
import pickle
import numpy as np
import time
import csv
import pandas as pd
#from synonyms import synonyms

index_file = "tfidf_inverse_index.pickle"
instance_tokens_file = "tfidf_instance_tokens.pickle"

global tfidf_inverse_index
global tfidf_log_inversed_df
global tfidf_dimen_tokens

print("[INFO] tfidf model loading...")
with open(index_file, 'rb') as iif:
    tfidf_inverse_index = pickle.load(iif)

with open(instance_tokens_file, 'rb') as itf:
    instance_tokens = pickle.load(itf)

# tf-idf = log（1 + F(t,d)) * log(N / df) : N is number of total instances

tfidf_dimen_tokens = list(tfidf_inverse_index.keys())
num_instances = len(instance_tokens.keys())

tfidf_log_inversed_df = [0] * len(tfidf_dimen_tokens)  # cache log(N / df)

for i in range(0, len(tfidf_dimen_tokens)):
    token = tfidf_dimen_tokens[i]
    x = math.log10(num_instances / len(tfidf_inverse_index[token]))
    tfidf_log_inversed_df[i] = x
    pass

del num_instances

print("[INFO] tfidf model havev loaded...")


def get_instance_tfidf_vector(instance_id):
    # tf-idf = log（1 + F(t,d)) * log(N / df) : N is number of total instances
    vector = [0] * len(tfidf_dimen_tokens)

    for i in range(0, len(tfidf_dimen_tokens)):
        idf = tfidf_log_inversed_df[i]  # get log (N / df) for cache
        # get F(t,d)
        tf = 0
        inverse_index_value = tfidf_inverse_index[tfidf_dimen_tokens[i]]
        for ele in inverse_index_value:
            if ele[0] == instance_id:
                tf = ele[1]
                break
        # calculate tf-idf
        # vector[i] = tf
        vector[i] = math.log(1 + tf) * idf

    return vector


def get_single_tfidf(word, instance_id):
    tfidf = 0
    tf = 0
    for i in range(0, len(tfidf_dimen_tokens)):
        idf = tfidf_log_inversed_df[i]
        if (tfidf_dimen_tokens[i] == word):
            inverse_index_value = tfidf_inverse_index[word]
            for ele in inverse_index_value:
                if ele[0] == str(instance_id):
                    tf = ele[1]
                    break
            tfidf = math.log(1 + tf) * idf
    return tfidf

# num=[]
#
# print(len(instance_tokens.values()))
# for line in instance_tokens:
# 	# print (line)
# 	# print(instance_tokens[line])
#
# 	num.append(len(instance_tokens[line]))
#
#
# num.sort(reverse=True)
# print(num[0])


# try:
# 	word = synonyms.v(num[0][2])
# 	print(word)
# except KeyError:
# 	word = 0
# 	pass

# def scale(sentence):
# 	w2vs=np.zeros([247,100])
# 	count=0
# 	for word in sentence:
# 		try:
# 			vector=synonyms.v(word)
#
# 		except KeyError:
# 			vector=np.zeros([1,100])
#
# 		w2vs[count]=vector
# 		count+=1
# 	return w2vs
#
# mmm=scale(instance_tokens['76523'])
# print(mmm)


# if __name__ == '__main__':
# 	#
# 	# list=[]
# 	# i=0
# 	# csvfile = csv.reader(open('./datas/testset_with_label.csv'))
# 	# with open('./datas/train_label.csv', 'w', newline='') as f:
# 	# 	writer = csv.writer(f)
# 	# 	writer.writerow(('tfidf', 'label'))
# 	# 	for line in csvfile:
# 	# 		i+=1
# 	# 		if(line[10]=='None'):
# 	# 			line[10]=5
# 	# 		writer.writerow([get_instance_tfidf_vector(line[0]),line[10]])
# 	# 		if(i%1000==0):
# 	# 			print(i)
# 	# 			if(i==5000):
# 	# 				break


# csvfile = pd.read_csv('./datas/testset_with_label.csv', skiprows=1)
# csvfile = pd.DataFrame(csvfile, columns=['Unnamed: 0', 'Label标签','tfidf_value'])
# for i in range(0,100):
# 	raw=get_instance_tfidf_vector(csvfile.loc[i]['Unnamed: 0'])
# 	label=csvfile.loc[i]['Label标签']
# 	list.append([raw,label])
#
# csvfile.to_csv('./datas/train_label.csv')
