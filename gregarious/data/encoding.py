import tqdm
import collections

class BytePairEncoder(object):
    """
    Does bytepair encoding
    """

    def __init__(self):
        self.int_tokens = {}

    @staticmethod
    def __init_charlist_generation(sents:list) -> list:
        """__init_charlist_generation
        Makes the initial charlist
        :param sents: sentences
        :type sents: list
        :rtype: list
        """
        product_array = []
        for sent in sents:
            sent = str(sent)+"\0"
            product_array.append(sent)
        oup = []
        for sent in product_array:
            sentList = []
            for word in sent.split(" "):
                sentList = sentList+list(word.lower())
            oup.append(sentList)
        return oup

    @staticmethod
    def __make_bytepair(arr: list) -> list:
        """__make_bytepair
        Makes bytepairs out of an array of tokens
        :param arr: array of tolkens
        :type arr: list
        :rtype: list
        """

        result = [(arr[i], arr[i+1]) for i in range(len(arr)-1)] 
        return result

    @staticmethod
    def __two_counting_dicts_to_one(dicta: dict, dictb: dict) -> dict:
        """__two_counting_dicts_to_one
        Makes two lovely counting dictionaries {key: int count, key: int count, ...} into one lovely counting dictionary
        :param dicta: dictionary a
        :type dicta: dict
        :param dictb: dictionary b
        :type dictb: dict
        :rtype: dict
        """
        oup = collections.defaultdict(int) 
        for key, val in dicta.items():
            oup[key] = oup[key]+val
        for key, val in dictb.items():
            oup[key] = oup[key]+val
        return dict(oup)

    @staticmethod
    def __return_bp_count(arr: list) -> dict:
        """__return_bp_count
        Returns counting dictionary of bytepairs
        :param arr: the bytepairs
        :type arr: list
        :rtype: dict
        """
        bp_count = collections.defaultdict(int)
        for i in arr:
            bp_count[i] += 1
        return dict(bp_count)

    @staticmethod
    def __combine(tokens: list, bp: tuple) -> list:
        tk_gen = []
        i = 0
        while i <= len(tokens) - 1:
            if i == len(tokens) - 1:
                tk_gen.append(tokens[i])
                i+=1
            elif (tokens[i], tokens[i+1]) == bp:
                tk_gen.append(tokens[i]+tokens[i+1])
                i+=2
            else:
                tk_gen.append(tokens[i])
                i+=1
        return tk_gen
    
    def encode(self, sents:list, factor:int=5, returns_string_tokens:bool=False) -> list:
        sents_tokenized = self.__init_charlist_generation(sents)
        print("BPEing...")
        for i in range(factor):
            print("Encoding Round", i+1, "of", factor)
            print("Tokenizing...")
            count_dicts = []
            for tokens in tqdm.tqdm(sents_tokenized):
                count_dicts.append(self.__return_bp_count(self.__make_bytepair(tokens)))
            print("Analyzing...")
            cd_master = {} 
            for d in tqdm.tqdm(count_dicts):
                cd_master = self.__two_counting_dicts_to_one(cd_master, d)
            try:
                maxbp = max(cd_master, key=cd_master.get)
            except ValueError:
                return sents_tokenized
            sents_tokenized = [self.__combine(i, maxbp) for i in sents_tokenized]
        if not returns_string_tokens:
            key = 0            
            sents_int = []
            print("Intergerizing...")
            for sent in tqdm.tqdm(sents_tokenized):
                sent_int = []
                for token in sent:
                    t_int = self.int_tokens.get(token)
                    if not t_int:
                        self.int_tokens[token] = key
                        t_int = key
                        key += 1
                    sent_int.append(t_int) 
                sents_int.append(sent_int)
            sents_tokenized = sents_int
        return sents_tokenized


