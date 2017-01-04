import logging
logging.basicConfig(level = logging.DEBUG)


class Word_index:
    """
    Dictionary mapping words (str) to their rank/index (int).
    This is the same as Keras Tokenizer, but with an already tokenized input.
    The class begins by expecting to set indices, until finialized() is called,
    when this happens, this class will return word indexes for known values,
    or the UNK's index, otherwise.
    TODO: Keras' tokenizer has a maximum number of words features -
    Should we add it?
    """
    def __init__(self):
        """
        Initialize the inner member dictionary
        """
        self.reset()

    def __getitem__(self, word):
        """
        Get (and possibly also set) the index for word
        """
        if word not in self.dic:
            if not self.finalized:
                # Add value to dic
                self.last_index += 1
                self.dic[word] = self.last_index
            else:
                # Not adding new values, revert word to UNK
                # This exists by default in the dictionary
                word = UNK_SYMBOL

        # By now, word is in self.dic for sure
        return self.dic[word]

    def __len__(self, include_UNK = False):
        """
        Returns the length of this dictionary, by default omitting UNK
        """
        return len(self.dic) - (0 if include_UNK else 1)


    def reset(self):
        """
        Reset this mapping.
        """
        self.dic = {UNK_SYMBOL : 1}
        self.last_index = 1
        self.finalized = False

    def finalize(self):
        """
        Finalize this mapping
        """
        self.finalized = True

    def iteritems(self, return_UNK = False):
        """
        Return an item generator.
        By default, will not include UNK
        """
        min_index = 1 if return_UNK else 2
        return iter([(k - ((min_index) - 1), v) for k, v in   # scale indices to [1, n]
                     self.dic.iteritems() if k >= min_index])




# CONSTANTS
UNK_SYMBOL = "UNK"


if __name__ == "__main__":
    m = Word_index()
    for word in "my dog ate my homework".split(" "):
        logging.debug((word, m[word]))
    m.finalize()
    for word in "my llama ate my football".split(" "):
        logging.debug((word, m[word]))
