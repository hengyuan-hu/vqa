import os
import sys
import json
import numpy as np
import re
import cPickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import Dictionary
import utils


contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
    "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
    "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
    "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
    "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
    "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
    "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
    "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
    "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
    "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
    "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
    "youll": "you'll", "youre": "you're", "youve": "you've"
}

manualMap = { 'none': '0',
              'zero': '0',
              'one': '1',
              'two': '2',
              'three': '3',
              'four': '4',
              'five': '5',
              'six': '6',
              'seven': '7',
              'eight': '8',
              'nine': '9',
              'ten': '10'}

articles = ['a', 'an', 'the']


periodStrip  = re.compile("(?!<=\d)(\.)(?!\d)")
commaStrip   = re.compile("(\d)(\,)(\d)")
punct        = [';', r"/", '[', ']', '"', '{', '}',
                     '(', ')', '=', '+', '\\', '_', '-',
                     '>', '<', '@', '`', ',', '?', '!']


def processPunctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(commaStrip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub("", outText, re.UNICODE)
    return outText


def processDigitArticle(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText


def multiple_replace(text, wordDict):
    for key in wordDict:
        text = text.replace(key, wordDict[key])
    return text


def preprocess_answer(answer):
    # answer = processDigitArticle(processPunctuation(answer))
    answer.replace('&', 'and')
    return answer


def filter_answers(answers_dset, min_occurence):
    """This will change the answer to preprocessed version
    """
    occurence = {}
    for ans_entry in answers_dset:
        answers = ans_entry['answers']
        answer2score = {}
        for answer in answers:
            # if answer['answer_confidence'] != 'yes':
            #     continue
            answer_ = answer['answer']
            answer_ = preprocess_answer(answer_)
            if answer_ not in answer2score:
                answer2score[answer_] = 0
            answer2score[answer_] += 1

            # this will change the content of answer
            answer['answer'] = answer_
        assert ans_entry['multiple_choice_answer'] in answer2score

        for answer in answer2score:
            # filter out 'incorrect' answers
            if answer2score[answer] < 3:
                continue

            if answer not in occurence:
                occurence[answer] = []
            occurence[answer].append(ans_entry['question_id'])

            # occurence[answer].add(ans_entry['question_id'])
    for answer in occurence.keys():
        if len(occurence[answer]) < min_occurence:
            occurence.pop(answer)

    print 'Num of answers that appear >= %d times: %d' % (
        min_occurence, len(occurence))
    return occurence


def create_ans2label(occurence, name, cache_root='data/cache'):
    """
    occurence: dict {answer -> whatever}
    name: prefix of the output file
    cache_root: str
    """
    ans2label = {}
    label = 0
    for answer in occurence:
        ans2label[answer] = label
        label += 1
    cache_file = os.path.join(cache_root, name+'_ans2label.pkl')
    cPickle.dump(ans2label, open(cache_file, 'wb'))
    return ans2label


def compute_target(answers_dset, ans2label, name, cache_root='data/cache'):
    """Augment answers_dset with soft score as label

    ***answers_dset should be preprocessed***

    Write result into a cache file
    """
    target = []
    for ans_entry in answers_dset:
        answers = ans_entry['answers']
        answer_count = {}
        for answer in answers:
            answer_ = answer['answer']
            answer_count[answer_] = answer_count.get(answer_, 0) + 1

        labels = []
        scores = []
        for answer in answer_count:
            if answer not in ans2label:
                continue
            labels.append(ans2label[answer])
            count = float(min(answer_count[answer], 3))
            scores.append(count / 3.0)

        target.append({
            'question_id': ans_entry['question_id'],
            'image_id': ans_entry['image_id'],
            'labels': labels,
            'scores': scores
        })

    cache_file = os.path.join(cache_root, name+'_target.pkl')
    cPickle.dump(target, open(cache_file, 'wb'))
    return target


def get_answer(qid, answers):
    for ans in answers:
        if ans['question_id'] == qid:
            return ans


def get_question(qid, questions):
    for question in questions:
        if question['question_id'] == qid:
            return question


# if __name__ == '__main__':
#     filename = 'data/v2_mscoco_train2014_annotations.json'
#     answers = json.load(open(filename))['annotations']
#     occurence = filter_answers(answers, 9)
