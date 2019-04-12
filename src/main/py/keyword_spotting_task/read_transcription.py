import os
import re


def read_transcription(file_name = "transcription.txt", output = "ID_dict"):
    """Function reading transcription or keyword file, returns dictionary.
    Options:
        - "ID_dict": dictionary key is positional ID (page-line-word position; for transcription file) or running number (for keyword file)
                dictionary entries are tuples of word with and without special characters as well as literal word
        - "word_dict": dictionary key is literal word
                dictionary entries are tuples of running ID and word with and without special characters
        Examples:
            - transcription file with option ID_dict:
                e.g. transcr_dict["270-01-02"] --> ('L-e-t-t-e-r-s-s_cm', 'L-e-t-t-e-r-s', 'Letters').
            - keyword file with option word_dict:
                e.g. keyword_dict["Commissary"] = (5, 'C-o-m-m-i-s_s-s-a-r-y', 'C-o-m-m-i-s-s-a-r-y')"""
    if output == "ID_dict" or output == "word_dict":
        word_dict = dict()
        counter = 0
        with open(file_name, "r") as data:
            for line in data:
                # print(line)
                if len(line.split()) == 2:
                    ID, word = line.split()
                else:
                    word = line.rstrip("\n")
                    ID = counter
                word_no_special_char = re.sub("s_s-", "s-", word)  # replaces the strong "s" (s_s-s) with s-s
                word_no_special_char = re.sub("-s_.*$", "", word_no_special_char)  # removes trailing special characters
                word_literal = ''.join(re.split("-", word_no_special_char))  # the literal word, without hyphens
                if output == "ID_dict":
                    word_dict[ID] = word, word_no_special_char, word_literal
                elif output == "word_dict":
                    word_dict[word_literal] = ID, word, word_no_special_char
                counter += 1
        return word_dict
    else:  # output specification is not valid
        print("\tPlease select output argument from 'ID_dict' or 'word_dict'.")
        print("\tWill return dictionary with ID (position or running number) or literal word as key, respectively.")
        return

test_dict= dict()
test_dict[2] = ["abc"]
test_dict = test_dict.update((2, "def"))
