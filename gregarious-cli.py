import warnings

with warnings.catch_warnings():
    import argparse
    from gregarious.data import io
    from gregarious.network import Gregarious
    from tqdm import tqdm

parser = argparse.ArgumentParser("(Easy) Gregarious-CLI: Bot Detection")
parser.add_argument("command", help="[create] corpus, [train] model, [tag] data, [interactive]")
parser.add_argument("-i", "--input", help="input file (csv, gregariousdata)")
parser.add_argument("-s", "--seed", help="network seed (h5)")
parser.add_argument("-o", "--output", help="output file (corpus_name, h5, csv)")

args = parser.parse_args()

if args.command == "create":
    dd = io.DataDescription()
    df = io.DataFile(args.input, dd, name=args.output)
    encoder = df.make_encoder(vocab_size=15000)
    df.compile(encoder, target_lang=None)


