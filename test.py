# #!/usr/local/bin/python3.7
# pylint=disable(maybe-no-member)

import csv
import pickle
import keras
import random
from gregarious.data import io
from gregarious.network import Gregarious
from contextlib import redirect_stdout
import langdetect
from langdetect import detect

from tqdm import tqdm

from keras.optimizers import Adam


dd = io.DataDescription()
du = io.DataFile('corpora/datasets/BotBonanza-2019-balanced.csv', dd, name="botbonanza-1000-balanced-v2")
# encoder = df.make_encoder()
# df.compile(encoder=encoder, target_lang=None)
# breakpoint()

# df.save()
# # # print(df.imported_data)

with open("botbonanza-1000-balanced.gregariousdata", "rb") as data:
    df = pickle.load(data)
with open("rtbust-1000-enhanced.gregariousdata", "rb") as data:
    df_sm = pickle.load(data)
# breakpoint()

# dd = io.DataDescription()
# d = io.DataFile('corpora/datasets/BotBonanza-2019-balanced.csv', dd, name="botbonanza-1000-revised")
# encoder = d.make_encoder(vocab_size=12000)
# d.compile(encoder, target_lang=None)
# breakpoint()

# isbots = df.importedData["isBot"]
# humans = 0
# bots = 0
# for i in isbots:
    # if i == [0, 1]
        # bots+=1
    # elif i == [1, 0]:
        # humans+=1

# breakpoint()
# net = Gregarious(df, optimizer=Adam(lr=1e-4))
# net = Gregarious(df, optimizer=Adam(lr=3e-4))
# net = Gregarious(df, optimizer=Adam(lr=1e-3))
# net = Gregarious(df_sm, optimizer=Adam(lr=3e-3)) 
# net = Gregarious(df, seed_model="trained_networks/BB-R5.h5")
# net = Gregarious(df_sm, seed_model="trained_networks/BB-R1.5.h5")
# net = Gregarious(df_sm, df_seed=df, seed_model="trained_networks/BB-R1.5.h5")
# net = Gregarious(df, seed_model="trained_networks/BB-R1.5.h5")
net = Gregarious(df, seed_model="trained_networks/BB-R6.h5")
net_sm = Gregarious(df_sm, seed_model="trained_networks/CRESTI-R3.h5")
# net.recompile(optimizer=Adam(lr=7e-3))
# net = Gregarious(df, optimizer=Adam(lr=1.15e-2))
# net = Gregarious(df, df_seed=df, seed_model="trained_networks/BB-R3-HeavyValidation.h5")
# net = Gregarious(df, optimizer=Adam(lr=2e-3))
# net = Gregarious(df, optimizer=Adam(lr=2e-2))
# net = Gregarious(df, optimizer=Adam(lr=0.1))
# breakpoint(# )

with open('modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        net.model.summary()

def lang_check(desc, status, target="en"):
        try:
            dd = detect(str(desc)) == target
            ss = detect(str(status)) == target
        except langdetect.lang_detect_exception.LangDetectException or TypeError:
            return False
        return dd and ss

def checkycheck():
    a = input("Username: ")
    b = input("User's Name: ")
    c = input("User's Description: ")
    d = input("Status: ")
    pred = net.predict([a], [b], [c], [d])
    r = pred[0]
    val = "Bot" if r[0] > r[1] else "Human" 
    print("Prediction:", val, "| Confidence:", max(r))

def checkycheck_cli(usrname, name, desc, status, net):
    pred = net.predict([usrname], [name], [desc], [status])
    r = pred[0]
    val = 1 if r[0] > r[1] else 0 
    return val, max(r)

def evaluate(d, net, sample_size):
    assert not d.isCompiled, "A compiled database cannot be eval'd"
    preds = []
    checks = []
    for _ in range(sample_size):
        item = random.randint(1, len(d.importedData['handle']))
        while (item in checks) or (not lang_check(d.importedData['description'][item], d.importedData['status'][item])):
            item = random.randint(1, len(d.importedData['handle']))
        checks.append(item)
    # checks = random.sample(range(len(d.importedData['handle'])), sample_size)
    # for handle, name, desc, status in tqdm(zip(d.importedData['handle'][:sample_size], d.importedData['name'][:sample_size], d.importedData['description'][:sample_size], d.importedData['status'][:sample_size]), total=len(d.importedData['name'][:sample_size])):
    
    for i in tqdm(checks):
        handle, name, desc, status = d.importedData['handle'][i], d.importedData['name'][i], d.importedData['description'][i], d.importedData['status'][i]
        pdData = net.predict([handle], [name], [desc], [status])[0]
        preds.append(pdData)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    yTrue = [d.importedData['isBot'][i] for i in checks]
    for hat, orig in tqdm(zip(preds, yTrue)):
        val = 1 if hat[0] > hat[1] else 0 
        if val == orig:
            if val == 1:
                tp += 1
            elif val == 0:
                tn += 1
        elif val > orig:
            fp += 1
        elif orig < val:
            fn += 1
    precision = (tp)/(tp+fp)
    recall = (tp)/(tp+fn)
    acc = (tp+tn)/(tp+tn+fp+fn)
    return precision, recall, acc, (tp, tn, fp, fn)
    

while True:
    checkycheck()
# dd = io.DataDescription()
# du = io.DataFile('corpora/datasets/cresci-rtbust-2019.csv', dd, name="cresci-rtbust-enhanced")
# print("Model performance {:.0%}".format(evaluate(d)))
# print(evaluate(du, net_sm, 5000))
breakpoint()
# net.train(epochs=6, batch_size=128, validation_split=0.2, callbacks=[keras.callbacks.TensorBoard(log_dir="./training_tb_logs/CRESTI-R3", update_freq="batch"), keras.callbacks.EarlyStopping(monitor="val_acc", patience=4, restore_best_weights=True)], save="trained_networks/CRESTI-R3.h5")
# breakpoint()

# from gregarious.data.encoding import BytePairEncoder

# ec = BytePairEncoder()

# tokens = ec.encode(["Reader reads the reading read to him by another reader.", "Ron reads a reading as well, for he is loving his kissings of the reading."], factor=10)
# print(tokens)

# breakpoint()
# # # print(ec.combine(tokens[0], ('r', 'a')))
# bp_a = ec._BytePairEncoder__make_bytepair(tokens[0])
# bp_b = ec._BytePairEncoder__make_bytepair(tokens[1])

# bpct_a = ec._BytePairEncoder__return_bp_count(bp_a)
# bpct_b = ec._BytePairEncoder__return_bp_count(bp_b)
# print(ec._BytePairEncoder__two_counting_dicts_to_one(bpct_a, bpct_b))
# breakpoint()
