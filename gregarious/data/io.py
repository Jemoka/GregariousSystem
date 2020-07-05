import os
import csv
import uuid
import tqdm
import pickle
import langdetect
from langdetect import detect

from bpe import Encoder

class DataDescription(object):
    """
    Describes a fields in a CSV
    """

    def __init__(self, header_index:int=0, header_keys_special:dict={}) -> None:
        """__init__

        :param header_index: the index of the header row, usually 0
        :type header_index: int
        :param header_keys_special: {"fileid": "name_in_header", ...}
        :type header_keys_special: dict
        :rtype: None
        """
        header_keys_default = {"handle":"handle", "name": "name", "description":"description", "followers_count":"followers_count", "friends_count":"friends_count", "status":"status", "isBot":"isBot"}
        
        header_keys = {}
        for key, vale in header_keys_default.items():
            hks = header_keys_special.get(str(key))
            if hks:
                header_keys[str(hks)] = str(key)
            else:
                header_keys[str(vale)] = str(key)
        self.header_index = header_index
        self.header_keys = header_keys
        self.ignore_list = []

    def ignore(self, ignore_list:list=None, ignore_str:str=None) -> None:
        assert (ignore_list is not None and ignore_str is None) or (ignore_list is None and ignore_str is not None), "Supply ONLY ignore_list or ignore_str."
        if ignore_str:
            self.ignore_list.append(ignore_list)
        elif ignore_list:
            self.ignore_list = self.ignore_list+ignore_list 
        
class DataFile(object):
    """
    Reads a CSV, gets some fields, serializes them
    """
    
    @staticmethod
    def __optimistically_cast(d):
        """__optimistically_cast
        Cast to the most likely type
        :param d: input 
        """
        
        try:
            res = float(d)
            if res == int(res):
                res = int(res)
        except ValueError:
            if d.upper() == "T" or d.upper() == "TRUE":
                res = True
            elif d.upper() == "F" or d.upper() == "FALSE":
                res = False
            else:
                res = str(d)
        return res

    def __init__(self, directory:str, description:DataDescription, name:str=str(uuid.uuid4())[-8:], save_dir:str="") -> None:
        """__init__

        :param directory: the directory of your lovely CSV
        :type directory: str
        :param description: the DataDescription object
        :type description: DataDescription
        :param delimiter: CSV delimiter
        :type delimiter: str
        :param quotechar: CSV quotecahr
        :type quotechar: str
        :rtype: None
        """

        self.dataDescription = description 
        with open(directory, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for indx, line in enumerate(reader):
                if indx == self.dataDescription.header_index:
                    header_items = list(line) 
                    break
            data_raw = {}
            for item in header_items:
                data_raw[item] = []
            for line in reader:
                for indx, item in enumerate(line):
                    try:
                        data_raw[header_items[indx]].append(self.__optimistically_cast(item))
                    except IndexError:
                        pass
        self.importedData = {}
        for key, val in data_raw.items():
            if str(key) not in list(self.dataDescription.header_keys.keys()):
                pass
            else:
                self.importedData[self.dataDescription.header_keys[str(key)]] = list(val)
        self.directory = os.path.join(save_dir, name+".gregariousdata")
        self.isCompiled = False

    @staticmethod
    def __lang_check(desc, status, target="en"):
        try:
            dd = detect(str(desc)) == target
            ss = detect(str(status)) == target
        except langdetect.lang_detect_exception.LangDetectException or TypeError:
            return False
        return dd and ss

    def make_encoder(self, vocab_size=2500, pct_bpe=0.2):
        assert not self.isCompiled, "Only raw corpuses can make encoders."
        encode_train = []
        for i in self.importedData["description"]:
            encode_train.append(str(i))
        for i in self.importedData["name"]:
            encode_train.append(str(i))
        for i in self.importedData["handle"]:
            encode_train.append(str(i))
        for i in self.importedData["status"]:
            encode_train.append(str(i))
        enc = Encoder(vocab_size, pct_bpe=pct_bpe)
        enc.fit(encode_train)
        return enc

    def compile(self, encoder, target_lang='en'):
        assert not self.isCompiled, "DataFile compiled already!"
        self.encoder = encoder
        if target_lang:
            print("Language conforming...")
            id_desc_conf = []
            id_status_conf = []
            id_handle_conf = []
            id_name_conf = []
            id_friends_conf = []
            id_followers_conf = []
            id_oup_conf = []

            for desc, status, handle, name, friends, followers, isBot in tqdm.tqdm(zip(self.importedData["description"], self.importedData["status"], self.importedData["handle"], self.importedData["name"], self.importedData["friends_count"], self.importedData["followers_count"], self.importedData["isBot"]), total=len(self.importedData["description"])):
                if self.__lang_check(desc, status, target_lang):
                    id_desc_conf.append(desc)
                    id_status_conf.append(status)
                    id_handle_conf.append(handle)
                    id_name_conf.append(name)
                    id_friends_conf.append(friends)
                    id_followers_conf.append(followers)
                    id_oup_conf.append(isBot)
            
            self.importedData["description"], self.importedData["status"], self.importedData["handle"], self.importedData["name"], self.importedData["friends_count"], self.importedData["followers_count"], self.importedData["isBot"] = id_desc_conf, id_status_conf, id_handle_conf, id_name_conf, id_friends_conf, id_followers_conf, id_oup_conf

        bot_indx = []
        human_indx = []
        for indx, i in enumerate(self.importedData["isBot"]):
            if i == 0:
                human_indx.append(indx)
            elif i == 1:
                bot_indx.append(indx)
        if len(bot_indx) > len(human_indx):
            bot_indx = bot_indx[:len(human_indx)-1]
        if len(bot_indx) < len(human_indx):
            human_indx = human_indx[:len(bot_indx)-1]
        id_desc_conf = []
        id_status_conf = []
        id_handle_conf = []
        id_name_conf = []
        id_friends_conf = []
        id_followers_conf = []
        id_oup_conf = []
        for i in bot_indx:
            id_desc_conf.append(self.importedData["description"][i])
            id_status_conf.append(self.importedData["status"][i])
            id_handle_conf.append(self.importedData["handle"][i])
            id_name_conf.append(self.importedData["name"][i])
            id_friends_conf.append(self.importedData["friends_count"][i])
            id_followers_conf.append(self.importedData["followers_count"][i])
            id_oup_conf.append(self.importedData["isBot"][i])
        for i in human_indx:
            id_desc_conf.append(self.importedData["description"][i])
            id_status_conf.append(self.importedData["status"][i])
            id_handle_conf.append(self.importedData["handle"][i])
            id_name_conf.append(self.importedData["name"][i])
            id_friends_conf.append(self.importedData["friends_count"][i])
            id_followers_conf.append(self.importedData["followers_count"][i])
            id_oup_conf.append(self.importedData["isBot"][i])
        self.importedData["description"], self.importedData["status"], self.importedData["handle"], self.importedData["name"], self.importedData["friends_count"], self.importedData["followers_count"], self.importedData["isBot"] = id_desc_conf, id_status_conf, id_handle_conf, id_name_conf, id_friends_conf, id_followers_conf, id_oup_conf
        print("Encoding...")
        self.importedData["handle"] = list(encoder.transform(self.importedData["handle"]))
        self.importedData["name"] = list(encoder.transform(self.importedData["name"]))
        self.importedData["description"] = list(encoder.transform(self.importedData["description"]))
        self.importedData["status"] = list(encoder.transform(self.importedData["status"]))
        print("Columizing...")
        na = []
        for point in self.importedData["isBot"]:
            if point == 0:
                na.append([0, 1])
            elif point == 1:
                na.append([1, 0])
        self.importedData["isBot"] = na
        self.isCompiled = True
        print("Done!")
        self.save()
        print("Saved!")

    def save(self):
        assert self.isCompiled, "You can't save an uncompiled dataset."
        with open(self.directory, "wb") as df:
            pickle.dump(self, df)

    def _recompile(self, target_lang='en'):
        input("I don't think you should be calling this. Would you like to continue? ")
        self.isCompiled = False
        self.compile(target_lang)

    class csv_parser(object):
        @staticmethod
        def __optimistically_cast(d):
            """__optimistically_cast
            Cast to the most likely type
            :param d: input 
            """
            
            try:
                res = float(d)
                if res == int(res):
                    res = int(res)
            except ValueError:
                if d.upper() == "T" or d.upper() == "TRUE":
                    res = True
                elif d.upper() == "F" or d.upper() == "FALSE":
                    res = False
                else:
                    res = str(d)
            return res

        def parse(self, inp, descriptor):
            with open(inp, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for indx, line in enumerate(reader):
                    if indx == descriptor.header_index:
                        header_items = list(line) 
                        break
                data_raw = {}
                for item in header_items:
                    data_raw[item] = []
                for line in reader:
                    for indx, item in enumerate(line):
                        try:
                            data_raw[header_items[indx]].append(self.__optimistically_cast(item))
                        except IndexError:
                            pass
            self.importedData = {}
            for key, val in data_raw.items():
                if str(key) not in list(descriptor.header_keys.keys()):
                    pass
                else:
                    self.importedData[descriptor.header_keys[str(key)]] = list(val)
            return [self.importedData["handle"], self.importedData["name"], self.importedData["description"], self.importedData["status"]]

        def write(self, oup, predictions, descriptor):
            with open(oup, 'w') as opf:
                opfWriter = csv.writer(opf)
                self.importedData["isBot"] = predictions
                newHeader = list(descriptor.header_keys.values())
                opfWriter.writerow(newHeader)
                opfWriter.writerows([list(i) for i in zip(*list(self.importedData.values()))])

class CorpusManager(object):
    def __init__(self, datafile):
        self.df = datafile

    def __pad(self, seqs, char=0, to=None):
        longest = len(max(seqs, key=len))
        if to:
            assert to >= longest, "You know, the pad length has to be bigger than or equal to the biggest length!"
            longest = to
        padded = []
        for i in seqs:
            new = i 
            while len(new) < longest:
                new.append(0)
            padded.append(new)
        return padded, longest
    
    def generate_predict_object(self, data, conform_lengths=None, f=False):
        if conform_lengths:
            _, handles_len = self.__pad(self.df.importedData["handle"], to=conform_lengths[0])
            _, names_len = self.__pad(self.df.importedData["name"], to=conform_lengths[1])
            _, desc_len = self.__pad(self.df.importedData["description"], to=conform_lengths[2])
            _, status_len = self.__pad(self.df.importedData["status"], to=conform_lengths[3])
        else:
            _, handles_len = self.__pad(self.df.importedData["handle"])
            _, names_len = self.__pad(self.df.importedData["name"])
            _, desc_len = self.__pad(self.df.importedData["description"])
            _, status_len = self.__pad(self.df.importedData["status"])

        handle = list(self.df.encoder.transform(data["handle"]))
        name = list(self.df.encoder.transform(data["name"]))
        description = list(self.df.encoder.transform(data["description"]))
        status = list(self.df.encoder.transform(data["status"]))
        if conform_lengths:
            if f:
                handles_enc, _ = self.__pad(handle[:conform_lengths[0]], to=conform_lengths[0])
                names_enc, _ = self.__pad(name[:conform_lengths[1]], to=conform_lengths[1])
                descs_enc, _ = self.__pad(description[:conform_lengths[2]], to=conform_lengths[2])
                statuses_enc, status_len = self.__pad(status[:conform_lengths[3]], to=conform_lengths[3])

            else:
                handles_enc, _ = self.__pad(handle, to=conform_lengths[0])
                names_enc, _ = self.__pad(name, to=conform_lengths[1])
                descs_enc, _ = self.__pad(description, to=conform_lengths[2])
                statuses_enc, status_len = self.__pad(status, to=conform_lengths[3])
        else:
            handles_enc, _ = self.__pad(handle)
            names_enc, _ = self.__pad(name)
            descs_enc, _ = self.__pad(description)
            statuses_enc, _ = self.__pad(status)
        return [handles_enc, names_enc, descs_enc, statuses_enc]

    def compute(self, maximum=None, conform_lengths=None):
        if conform_lengths:
            handles, handles_len = self.__pad(self.df.importedData["handle"][:maximum], to=conform_lengths[0])
            names, names_len = self.__pad(self.df.importedData["name"][:maximum], to=conform_lengths[1])
            description, desc_len = self.__pad(self.df.importedData["description"][:maximum], to=conform_lengths[2])
            status, status_len = self.__pad(self.df.importedData["status"][:maximum], to=conform_lengths[3])
        else:
            handles, handles_len = self.__pad(self.df.importedData["handle"][:maximum])
            names, names_len = self.__pad(self.df.importedData["name"][:maximum])
            description, desc_len = self.__pad(self.df.importedData["description"][:maximum])
            status, status_len = self.__pad(self.df.importedData["status"][:maximum])
        followers = self.df.importedData["followers_count"][:maximum]
        friends = self.df.importedData["friends_count"][:maximum]
        friends_and_follwers = []
        for fl, fr in zip(followers, friends):
            friends_and_follwers.append([fl, fr])
        isBot = self.df.importedData["isBot"][:maximum]
        # return {"meta":{"lengths":[handles_len, names_len, desc_len, status_len, 2]}, "ins":[handles, names, description, status, friends_and_follwers], "out":[isBot]} 
        return {"meta": {"lengths": [handles_len, names_len, desc_len, status_len, 2]}, "ins": [handles, names, description, status], "out": [isBot]} 
        # return {"meta":{"lengths":[desc_len, status_len, 2]}, "ins":[description, status], "out":[isBot]} 
