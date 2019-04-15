#!/usr/bin/python

import re


def read_transcription(file_name = "transcription.txt", output = "ID_dict"):
    """Function reading transcription or keyword file, returns dictionary.
    Options:
        - "ID_dict": dictionary key is positional ID (page-line-word position; for transcription file) or running number (for keyword file)
                dictionary entries are tuples of word with and without special characters as well as literal word
        - "word_dict": dictionary key is literal word
                dictionary entries are lists of tuples of running ID and word with and without special characters
                For words occurring multiple times, the list has multiple entries.
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
                word_no_special_char = re.sub("s_s-", "s-", word)  # special case: replaces the strong "s" (s_s-s) with s-s
                word_no_special_char = re.sub("-s_.*$", "", word_no_special_char)  # removes trailing special characters
                word_literal = ''.join(re.split("-", word_no_special_char))  # the literal word, without hyphens
                if output == "ID_dict":
                    word_dict[ID] = word, word_no_special_char, word_literal
                elif output == "word_dict":
                    new_entry = tuple([ID, word, word_no_special_char])  # tuple
                    if word_literal not in word_dict:  # new word
                        word_dict[word_literal] = [new_entry]  # list containing tuple
                    else:  # word is already in dictionary
                        existing_entry = word_dict.pop(word_literal)
                        existing_entry.append(new_entry)
                        word_dict[word_literal] = existing_entry
                counter += 1
        return word_dict
    else:  # output specification invalid
        print("\tPlease select output argument from 'ID_dict' or 'word_dict'.")
        print("\tWill return dictionary with ID (position or running number) or literal word as key, respectively.")
        return

# test_dict = read_transcription(output = "word_dict")


def retrieve_IDs(word, dictionary):
    """Function returning list of positional IDs (page-line-word position) from checking word_dictionary
    (key: word; value: tuple of positional ID, word with and without special characters)"""
    IDs = [dictionary[word][i][0] for i in range(len(dictionary[word]))]
    return IDs

# retrieve_IDs("General", test_dict)


def generate_word_ID_csv(word_dict, file_name = "words_positions.txt", sep = ", "):
    """Given a word-dictionary (key = word, values = positions), this function
    generates a csv-file of words and respective positions in the images."""
    with open(file_name, "w") as file:
        for word in word_dict:
            out = str(word) + sep
            file.write(out)
            if len(retrieve_IDs(word, word_dict)) == 1:
                out = str(retrieve_IDs(word, word_dict)[0]) +"\n"
                file.write(out)
            else:
                for i in range(len(retrieve_IDs(word, word_dict))):
                    out = str(retrieve_IDs(word, word_dict)[i])
                    if i < len(retrieve_IDs(word, word_dict))-1:
                        file.write("".join([out, sep]))
                    else:
                        file.write("".join([out, "\n"]))
    return


# generate_word_ID_csv(word_dict = test_dict, file_name = "test_word_ID.txt", sep = ",")
