import json

q_list = []
a_list = []
s = "Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame"
print(len(s))
with open('./SQuAD2.0/train-v2.0.json', 'r') as file:
    str = file.read()
    row_data = json.loads(str)

    # 442个主题,逐一个主题检视
    title_count = 0
    while (title_count < len(row_data['data'])):
        title_data = row_data['data'][title_count]

        # 66个段落,逐一个段落检视
        paragraphs_count = 0
        while (paragraphs_count < len(title_data['paragraphs'])):
            paragraphs_data = title_data['paragraphs'][paragraphs_count]

            # 15个问题,逐一个问题检视
            qas_count = 0
            while (qas_count < len(paragraphs_data['qas'])):
                qas_data = paragraphs_data['qas'][qas_count]

                # 取出question, answers组成数列

                # 假若没有question则自己补上no question
                if qas_data['question'] == []:
                    question_data_clean = qas_data['question'] + [' no question']
                else:
                    question_data_clean = qas_data['question']
                q_list = q_list + [question_data_clean]

                # 假若没有answers则自己补上no answers
                if qas_data['answers'] == []:
                    answers_data_clean = qas_data['answers'] + [' no answers']
                else:
                    answers_data_clean = qas_data['answers'][0]['text']
                a_list = a_list + [answers_data_clean]

                qas_count = qas_count + 1

            paragraphs_count = paragraphs_count + 1

        title_count = title_count + 1

# question和answers存成json
file_name_1 = 'q_list.json'
file_name_2 = 'a_list.json'
with open(file_name_1, 'w') as file_object:
    json.dump(q_list, file_object)
with open(file_name_2, 'w') as file_object:
    json.dump(a_list, file_object)