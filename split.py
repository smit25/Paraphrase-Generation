"""
Take the .tsc quora suplicate question dataset and split it into train, val and test data ans store it in json format.
This script will generate:
- quora_raw_train.json
- quora_raw_val.json 
- quora_raw_test.json

"""

import csv
import json
import os

train_len = 50000
test_len = 30000
val_len = 20000

def main():
	out = []
	outtest = []
	outval = []
	with open('./data/quora_duplicate_questions.tsv','rb') as tsvin:
		tsvin = csv.reader(tsvin, delimiter='\t')#read the tsv file of quora question pairs
		count0 = 1
		count1 = 1
		counter = 1
		for row in tsvin:
			counter += 1
			#the 6th entry in every row has value 0 or 1 and it represents paraphrases if that value is 1
			if row[5]=='0' and row[4][-1:]=='?':
				count0 += 1
			elif row[5]=='1' and row[4][-1:]=='?':
				count1 += 1
				# Only considering duplicate pairs for our dataset
				if count1>1 and count1< train_len +2:
					# get the question and unique id from the tsv file
					quesid = row[1] #first question id
					ques = row[3] #first question
					img_id = row[0] #unique id for every pair
					ques1 = row[4] #paraphrase question
					quesid1 =row[2] #paraphrase question id				
					
					# set the parameters of json file for writing 
					jimg = {}

					jimg['question'] = ques
					jimg['question1'] = ques1
					jimg['ques_id'] =  quesid
					jimg['ques_id1'] =  quesid1
					jimg['id'] =  img_id

					out.append(jimg)

				elif count1>train_len + 1 and count1<train_len + test_len +2:
					quesid = row[1] 
					ques = row[3] 
					img_id = row[0] 
					ques1 = row[4]
					quesid1 =row[2]

					jimg = {}

					jimg['question'] = ques
					jimg['question1'] = ques1
					jimg['ques_id'] =  quesid
					jimg['ques_id1'] =  quesid1
					jimg['id'] =  img_id

					outtest.append(jimg)

				else : # Validation set
					quesid = row[1] 
					ques = row[3] 
					img_id = row[0] 
					ques1 = row[4]
					quesid1 =row[2]
				
					jimg = {}
					jimg['question'] = ques
					jimg['question1'] = ques1
					jimg['ques_id'] =  quesid
					jimg['ques_id1'] =  quesid1
					jimg['id'] =  img_id
					
					outval.append(jimg)
	outval = outval[:val_len]
	#write the json files for train test and val
	print(len(out))
	json.dump(out, open('../data/quora_raw_train.json', 'w'))
	print(len(outtest))
	json.dump(outtest, open('../data/quora_raw_test.json', 'w'))
	print(len(outval))
	json.dump(outval, open('../data/quora_raw_val.json', 'w'))

if __name__ == "__main__":
	main()