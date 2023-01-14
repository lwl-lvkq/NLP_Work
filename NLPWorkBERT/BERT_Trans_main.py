import torch
import datasets
import torch.utils.data as Data
import transformers

import warnings
warnings.filterwarnings("ignore")


class BERT_Model(torch.nn.Module):
    def __init__(self, BERT_model):
        super().__init__()
        self.BERT = BERT_model
        self.fc1 = torch.nn.Linear(768, 256)
        self.fc2 = torch.nn.Linear(256, 2)

    def forward(self, ids, masks, tokenids):

        # with torch.no_grad():
        output = self.BERT(ids, masks, tokenids)
        
        output = self.fc2(self.fc1(output[0][:, 0]))
        output = output.softmax(dim=1)

        return output


def collate_fn(data):
    
    sentences = [tuple_x['text'] for tuple_x in data]
    labels = [tuple_x['label'] for tuple_x in data]
    
    token = transformers.BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='models')
    
    data = token.batch_encode_plus(batch_text_or_text_pairs=sentences,
                                   truncation=True,
                                   max_length=500,
                                   padding='max_length',
                                   return_tensors='pt',
                                   return_length=True)
    input_ids = data['input_ids']
    attention_mask = data['attention_mask'] 
    token_type_ids = data['token_type_ids'] 
    labels = torch.LongTensor(labels)

    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")
        token_type_ids = token_type_ids.to("cuda")
        labels = labels.to("cuda")

    return input_ids, attention_mask, token_type_ids, labels



def train(model, dataset):

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    lossfun = torch.nn.CrossEntropyLoss()

    loader_train = Data.DataLoader(dataset=dataset,
                                   batch_size=32,
                                   collate_fn=collate_fn,
                                   shuffle=True,
                                   drop_last=True) 
    model.train()
    total_acc_num = 0
    train_num = 0
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader_train):
        output = model(input_ids, attention_mask, token_type_ids)

        loss = lossfun(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.argmax(dim=1)
        accuracy_num = (output == labels).sum().item()
        total_acc_num += accuracy_num
        train_num += loader_train.batch_size

    return total_acc_num / train_num



def test(model, dataset):
    correct_num = 0
    test_num = 0
    loader_test = Data.DataLoader(dataset=dataset,
                                  batch_size=32,
                                  collate_fn=collate_fn,
                                  shuffle=True,
                                  drop_last=True)
    model.eval()
    for t, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader_test):
        with torch.no_grad():
            output = model(input_ids, attention_mask, token_type_ids)

        output = output.argmax(dim=1)
        correct_num += (output == labels).sum().item()
        test_num += loader_test.batch_size

    return correct_num / test_num


def main():
    preBERT_model = transformers.BertModel.from_pretrained('bert-base-uncased', cache_dir='models')

    model = BERT_Model(preBERT_model)

    if torch.cuda.is_available():
        model.to("cuda")

    my_dataset_all = datasets.load_dataset("csv", data_files={
        "train": "ChnSentiCorp_csv/train.csv", 
        "validation": "ChnSentiCorp_csv/dev.csv", 
        "test": "ChnSentiCorp_csv/test.csv"}, 
        cache_dir='ChnSentiCorp_csv')

    train_data = my_dataset_all['train']
    test_data = my_dataset_all['test']

    epochs = 3

    for i in range(0, epochs):
        print("==================== epoch={} ====================".format(i))

        epoch_loss = train(model, train_data)
        print("epoch_loss =", epoch_loss)

        test_acc = test(model, test_data)
        print("test_acc =", test_acc)


main()


        