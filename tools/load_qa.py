import json


if __name__ == '__main__':
    answer_file = 'data/v2_mscoco_train2014_annotations.json'
    answers = json.load(open(answer_file))['annotations']

    question_file = 'data/v2_OpenEnded_mscoco_train2014_questions.json'
    questions = json.load(open(question_file))['questions']

    # occurence = filter_answers(answers, 9)
