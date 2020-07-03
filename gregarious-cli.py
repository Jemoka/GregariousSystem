import warnings

with warnings.catch_warnings():
    from gregarious.network import Gregarious
    from keras.optimizers import Adam
    from gregarious.data import io
    from tqdm import tqdm
    import argparse
    import pickle
    import keras
    
parser = argparse.ArgumentParser("(Easy) Gregarious-CLI by @jemoka")
parser.add_argument("command", help="[create] corpus, [train] model, [tag] data, [interactive]")
parser.add_argument("-i", "--input", help="input file (csv, gregariousdata)")
parser.add_argument("-s", "--seed", help="network seed (h5)")
parser.add_argument("-o", "--output", help="output file (corpus_name, h5, csv)")
parser.add_argument("-m", help="manual parametre input", action="store_true")

args = parser.parse_args()

if args.command == "create":
    dd = io.DataDescription()
    df = io.DataFile(args.input, dd, name=args.output)
    if args.m:
        encoder = df.make_encoder(vocab_size=int(input("BPE Vocab Size: ")))
    else:
        encoder = df.make_encoder(vocab_size=15000)
    df.compile(encoder, target_lang=None)
elif args.command == "train":
    with open(args.input, "rb") as data:
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

