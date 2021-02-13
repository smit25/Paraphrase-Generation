import h5py
import copy
import json
import torch
import torch.utils.data as data

class Dataloader(data.Dataset):
    def __init__(self, input_json_path, input_h5py_path):
        super(Dataloader).__init__()

        print('JSON file')
        with open(input_json_path) as input_file:
            data_dict = json.load(input_file)

        self.index_to_word = {}

        # copying the dictionary    
        self.index_to_word = data_dict['index_to_word'].copy()
        
        if 0 not in self.index_to_word:
            self.index_to_word[0] = 'UNK'
        else:
            raise Exception
        
        dict_len = len(self.index_to_word)
        self.EOS, self.PAD, self.SOS, = dict_len, dict_len+1, dict_len +2
        self.index_to_word[self.EOS] = '<EOS>'
        self.index_to_word[self.PAD] = '<PAD>'
        self.index_to_word[self.SOS] = '<SOS>'
        dict_len += 3

        print('H5PY file will open')

        qa_data = h5py.File(input_h5py_path, 'r')

        ques_id_train = torch.from_numpy(qa_data['ques_dup_id_train'][...].astype(int))

        ques_train, ques_len_train = self.process_data(torch.from_numpy(qa_data['ques_train'][...].astype(int)), torch.from_numpy(qa_data['ques_length_train'][...].astype(int)))
        print('ques_train_shape: ', ques_train.shape)

        label_train, label_len_train = self.process_data(torch.from_numpy(qa_data['ques1_train'][...].astype(int)), torch.from_numpy(qa_data['ques1_length_train'][...].astype(int)))

        self.train_id = 0
        self.seq_length = ques_train.size()[1]

        print('Training dataset length : ', ques_train.size()[0])


        ques_test, ques_len_test = self.process_data(torch.from_numpy(qa_data['ques_test'][...].astype(int)), torch.from_numpy(qa_data['ques_length_test'][...].astype(int)))

        label_test, label_len_test = self.process_data(torch.from_numpy(qa_data['ques1_test'][...].astype(int)), torch.from_numpy(qa_data['ques1_length_test'][...].astype(int)))

        ques_id_test = torch.from_numpy(qa_data['ques_dup_id_test'][...].astype(int))

        self.test_id = 0

        print('Test dataset length : ', ques_test.size()[0])
        #close the h5py file
        qa_data.close()

        self.ques = torch.cat([ques_train, ques_test])
        self.len = torch.cat([ques_len_train, ques_len_test])
        self.label = torch.cat([label_train, label_test])
        self.label_len = torch.cat([label_len_train, label_len_test])
        self.id = torch.cat([ques_id_train, ques_id_test])

    def process_data(self, data, data_len):
        N = data.size()[0]
        new_data = torch.zeros(N, data.size()[1] + 2, dtype=torch.long) + self.PAD
        for i in range(N):
            new_data[i, 1:data_len[i]+1] = data[i, :data_len[i]] #shifting the data rightwards by a col
            new_data[i, 0] = self.SOS #adding SOS token
            new_data[i, data_len[i]+1] = self.EOS # adding EOS token
            data_len[i] += 2 # increase len of matrix for compensating the SOS and EOS
        return new_data, data_len

    def __len__(self):
        return self.len.size()[0]

    def __getitem__(self, idx):
        return (self.ques[idx], self.len[idx], self.label[idx], self.label_len[idx], self.id[idx])

    def getVocabSize(self):
        return len(self.index_to_word)

    def getDataNum(self, split):
        if split == 1:
            return 100000

        if split == 2:
            return 30000
            
    def getSeqLength(self):
        return self.seq_length
        