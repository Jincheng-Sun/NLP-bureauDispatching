import math
import pickle


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
	tfidf_log_inversed_df[i] = math.log10( num_instances / len(tfidf_inverse_index[token]) )

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
		vector[i] = math.log( 1 + tf) * idf

	return vector




if __name__ == '__main__':
	# test
	result  = get_instance_tfidf_vector("113508")
	print(result)








