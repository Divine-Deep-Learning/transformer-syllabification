import re
import pickle
import numpy as np


def check_male_caesura(formed_verse):
    pos = 0
    for i, (acc, end) in enumerate(formed_verse):
        if (acc == 1) and (i == 5 or i == 3) and (end == 1):
            if formed_verse[i + 1][0] == 1:
                pos = i
                break
    if pos == 0:
        for i, (acc, end) in enumerate(formed_verse):
            if (acc == 1) and (i == 5 or i == 3) and (end == 1):
                pos = i
                break
    return pos


def check_liric_caesura(formed_verse):
    if (formed_verse[2][0] == 1) and (formed_verse[2][1] == 0) \
            and (formed_verse[3][0] == 0) and (formed_verse[3][1] == 1):
        return 3
    return 0


def check_female_caesura(formed_verse):
    pos = 0
    for i, (acc, end) in enumerate(formed_verse):
        if (acc == 1) and (i == 5 or i == 3) and (end == 0) \
                and (formed_verse[i + 1][1] == 1) and (formed_verse[i + 1][0] == 0):
            pos = i + 1
            break
    return pos


def clean_text(text):
    new_text = []
    for line in text:
        new_line = re.sub(r'á', 'à', line)
        new_line = re.sub(r'í', 'ì', new_line)
        new_line = re.sub(r'ó', 'ò', new_line)
        new_line = re.sub(r'ú', 'ù', new_line)
        new_line = re.sub(r'(\w)è(\w)', r'\1e\2', new_line)
        new_line = re.sub(r'(\w)à(\w)', r'\1a\2', new_line)
        new_line = re.sub(r'(\w)ù(\w)', r'\1u\2', new_line)
        new_line = re.sub(r'(\w)ì(\w)', r'\1i\2', new_line)
        new_line = re.sub(r'(\w)ò(\w)', r'\1o\2', new_line)
        new_line = re.sub(r'(’)|(‘)', '\'', new_line)
        new_text.append(new_line)
    return new_text


def main():
    file_X = open('resources/X.csv', 'r+', encoding='utf-8')
    file_y = open('resources/y.csv', 'r+', encoding='utf-8')
    file_dest_y = open('resources/y_cesura.csv', 'w+', encoding='utf-8')
    file_dest_X = open('resources/X_cesura.csv', 'w+', encoding='utf-8')

    with open('./resources/orig/dantes_dictionary.pkl', 'rb') as f:
        dictionary = pickle.load(f)

    # Latin words not considered in dictionary
    dictionary['aegypto'] = [((1, 2, -1, 0), 'a|egyp|to', 1)]
    dictionary['voil'] = [((1, 1, 0, 0), 'voil', 1)]
    dictionary['gausen'] = [((0, 2, 0, 0), 'gau|sen', 1)]
    dictionary['goi'] = [((0, 1, 0, 0), 'goi', 1)]

    lines_X = file_X.readlines()
    lines_y = file_y.readlines()
    new_lines_y = []
    new_lines_X = []

    for i, syl_line in enumerate(lines_X):
        ##### STRIPPING DOWN VERSE #####
        syl_line = re.sub(r'<start>|<end>|\n', '', syl_line)
        # using default " ’ " type in X and y, reconvert at the end
        words = syl_line.split('<s>')
        pos_acc_array = []
        abs_pos = -1
        for w in words:
            rel_pos, n_syl = dictionary[w][0][0][2], dictionary[w][0][0][1]
            abs_pos += n_syl
            pos_acc_array.append((rel_pos, n_syl, abs_pos + rel_pos))
        #  pos_acc_array contiene una tripla per ogni parola:
        #  (pos accento dal fondo, numero sillabe della parola, posizione assoluta dell'accento)

        words_syl = re.sub(r'<start>|<end>|\n', '', lines_y[i])
        words_syl = re.sub(r'<syl>', '', words_syl, count=1)
        words_syl = words_syl.split('<s><syl>')
        # words_syl è una lista di parole contenenti <syl> e <s> in caso di sinalefe

        formed_verse = []
        num_snl = 0  # recalculates for each line the number of synalefa
        for j, w in enumerate(words_syl):
            word_syls = w.split('<syl>')
            for rel_syl_index in range(len(word_syls)):
                prop = [0, 0]
                num_snl += len(re.findall(r'<s>', word_syls[rel_syl_index]))
                if len(word_syls) + pos_acc_array[j + num_snl][0] - 1 == rel_syl_index:
                    prop[0] = 1
                if rel_syl_index == len(word_syls) - 1:
                    prop[1] = 1
                formed_verse.append(prop)
        # formed_verse è una lista di 11 elementi
        # [accento sulla sillaba corrente?, ultima sillaba della parola?]

        pos = check_male_caesura(formed_verse)
        if pos < 3:
            pos = check_liric_caesura(formed_verse)
        if pos < 3:
            pos = check_female_caesura(formed_verse)

        if pos > 2:
            ##### REBUILDING VERSE ####
            new_line = '<start><syl>'
            counter = -1
            for j, w in enumerate(words_syl):
                counter += len(w.split('<syl>'))
                new_line += w
                if counter != pos:
                    new_line += '<s>'
                else:
                    new_line += '<c><s>'
                if j < len(words_syl) - 1:
                    new_line += '<syl>'
            new_line += '<end>\n'
            new_line = re.sub(r'<s><end>', '<end>', new_line)
            new_lines_y.append(new_line)
            new_lines_X.append(re.sub(r'<syl>|<c>', '', new_line))
        else:
            new_line = '<unl><start><syl>'
            counter = -1
            for j, w in enumerate(words_syl):
                counter += len(w.split('<syl>'))
                new_line += w + '<s>'
                if j < len(words_syl) - 1:
                    new_line += '<syl>'
            new_line += '<end>\n'
            new_line = re.sub(r'<s><end>', '<end>', new_line)
            new_lines_X.append(re.sub(r'<syl>', '', new_line))
            new_lines_y.append('<unl>\n')

    # Troubleshooting
    print(f"Added {sum([len(re.findall(r'<c>', line)) for line in new_lines_y])} "
          f"caesuras instead of {len(lines_X)}")

    new_lines_X = clean_text(new_lines_X)
    new_lines_y = clean_text(new_lines_y)

    file_dest_y.writelines(new_lines_y)
    file_dest_X.writelines(new_lines_X)
    file_y.close()
    file_X.close()
    file_dest_y.close()
    file_dest_X.close()


if __name__ == '__main__':
    main()
