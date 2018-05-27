from data_gen.errors import InvalidArgCount, InvalidArgType


FIRST_ELEMENT = 0
EXCLUDED_ITEM_INDEX = 1


def sentence_combination(list_1, list_2):
    """
    Creates combinations of all possible sentences from list_1 and list_2
    :param list_1: list
    A list of sentence-parts to precede.

    :param list_2: list
    A list of sentence-parts to succeed

    :rtype: list
    """
    return [(term_1 + ' ' + term_2) for term_1 in list_1 for term_2 in list_2]


def sentence_gen(*sentence_parts):
    """=====================================================================================
    :param sentence_parts: list
    List of sentence parts.

    :rtype: list
    ----------------------------------------------------------------------------------------
    Objective: Recursive function to extend sentence combination.
    ----------------------------------------------------------------------------------------
    Use as:

        sentence_gen(list1, list2, ..., listN)
    ----------------------------------------------------------------------------------------
    Code hints:

    1  ) Take the first element as the first part of the sentence.
    2  ) Slice the sentence_parts to exclude the first element.
    3.a) If the sentence_parts contains only one list, return the combination with prime.
    3.b) Else unpack the sentence_parts tuple to provide the modified list of sentences.
    ======================================================================================"""
    for i, part in enumerate(sentence_parts):
        if type(part) is not list:
            raise InvalidArgType(part, i)

    if len(sentence_parts) < 2:
        raise InvalidArgCount(sentence_parts)

    prime = sentence_parts[FIRST_ELEMENT]
    sentence_parts = sentence_parts[EXCLUDED_ITEM_INDEX:]

    if len(sentence_parts) == 1:
        return sentence_combination(prime, sentence_parts[FIRST_ELEMENT])
    else:
        return sentence_combination(prime, sentence_gen(*sentence_parts))
