"""
Preprocessing for PAWS wiki_labeled_swap.tar
Preprocessing for PAWS wiki_labeled_final.tar
Convert tar files to tsv files
"""
import csv
import json

def main():
  train_swap = []
  train_final = []
  with open('data/PAWS_train_swap.tsv', 'r', encoding= 'utf8') as tsvfile:
    read = csv.reader(tsvfile, delimiter = "\t")

    next(read)

    for row in read:
      if row[3] == '1':
        obj ={}
        obj['sentence'] = row[1]
        obj['paraphrase'] = row[2]
        train_swap.append(obj)
    tsvfile.close()

  with open('data/PAWS_train_labeled.tsv', 'r', encoding = 'utf8') as tsvfile:
    reader = csv.reader(tsvfile, delimiter = "\t")

    next(reader)

    for row in reader:
      if row[3] == '1':
        obj ={}
        obj['sentence'] = row[1]
        obj['paraphrase'] = row[2]
        train_final.append(obj)

    tsvfile.close()

    print(len(train_final))
    print(len(train_swap))

    json.dump(train_final, open('./data/PAWS_train_final.json', 'w'))
    json.dump(train_swap, open('./data/PAWS_train_swap.json', 'w'))
    print('JSON files created!')


if __name__ == "__main__":
	main()