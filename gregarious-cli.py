import warnings
print("Setting up...")
with warnings.catch_warnings():
    from gregarious.network import Gregarious
    from keras.optimizers import Adam
    from gregarious.data import io
    from tqdm import tqdm
    import argparse
    import pickle
    import keras

def checkycheck(net):
    a = input("Username: ")
    b = input("User's Name: ")
    c = input("User's Description: ")
    d = input("Status: ")
    pred = net.predict([a], [b], [c], [d])
    r = pred[0]
    val = "Bot" if r[0] > r[1] else "Human" 
    print("Prediction:", val, "| Confidence:", max(r))
    
parser = argparse.ArgumentParser("(Easy) Gregarious-CLI by @jemoka")
parser.add_argument("command", help="[create] corpus, [train] model, [tag] data, [interactive] discrimination")
parser.add_argument("-i", "--input", help="input file (csv)")
parser.add_argument("-s", "--seed", help="network seed (h5)")
parser.add_argument("-d", "--handler", help="data handler seed (gregariousdata)")
parser.add_argument("-o", "--output", help="output file (corpus name, h5, csv)")
parser.add_argument("-m", help="manual parametre input", action="store_true")

args = parser.parse_args()

if args.command == "create":
    dd = io.DataDescription()
    df = io.DataFile(args.input, dd, name=args.output)
    if args.m:
        encoder = df.make_encoder(vocab_size=int(input("BPE Vocab Size: ")))
    else:
        encoder = df.make_encoder(vocab_size=12000)
    df.compile(encoder, target_lang=None)
elif args.command == "train":
    with open(args.handler, "rb") as data:
        df = pickle.load(data)
    net = Gregarious(df, optimizer=Adam(3e-3))
    if args.m:
        print("Let's train a model...")
        print("======================")
        e = int(input("Epochs: "))
        bs = int(input("Batch Size: "))
        vs = int(input("Validation Split: "))
    else:
        e = 6
        bs = 128
        vs = 0.2
    net.train(epochs=e, batch_size=bs, validation_split=vs, callbacks=[keras.callbacks.EarlyStopping(monitor="val_acc", patience=4, restore_best_weights=True)], save=args.output)
elif args.command == "tag":
    with open(args.handler, "rb") as data:
        df = pickle.load(data)
    net = Gregarious(df, df_seed=df, seed_model=args.seed)
    net.predict_csv(args.input, args.output)
elif args.command == "interactive":
    with open(args.handler, "rb") as data:
        df = pickle.load(data)
    net = Gregarious(df, df_seed=df, seed_model=args.seed)
    while True:
        checkycheck(net)
        r = input("Quit (Q) or Continue (enter) => ")
        if "q" in r.lower():
            break
print("Closing down...")

