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
    if (formed_verse[2][0] == 1) and (formed_verse[2][1] == 0) and (formed_verse[3][0] == 0) and (formed_verse[3][1] == 1):
        return 3
    return 0


def check_female_caesura(formed_verse):
    pos = 0
    for i, (acc, end) in enumerate(formed_verse):
        if (acc == 1) and (i == 5 or i == 3) and (end == 0) and (formed_verse[i+1][1] == 1) and (formed_verse[i+1][0] == 0):
            pos = i + 1
            break
    return pos


def main():
    file_X = open('resources/X.csv', 'r+', encoding='utf-8')
    file_y = open('resources/y.csv', 'r+', encoding='utf-8')
    file_dest = open('resources/y_cesura.csv', 'w+', encoding='utf-8')
    with open('./resources/orig/dantes_dictionary.pkl', 'rb') as f:
        dictionary = pickle.load(f)

    lines_X = file_X.readlines()
    lines_y = file_y.readlines()

    new_lines = []
    for i, syl_line in enumerate(lines_X[:10]):
        syl_line = re.sub(r'<start>|<end>|\n', '', syl_line)
        syl_line = re.sub(r"'", "’", syl_line)
        words = syl_line.split('<s>')
        pos_acc_array = []
        abs_pos = -1
        for w in words:
            rel_pos, n_syl = dictionary[w][0][0][2], dictionary[w][0][0][1]
            abs_pos += n_syl
            pos_acc_array.append((rel_pos, n_syl, abs_pos + rel_pos))

        #  pos_acc_array contiene una tripla per ogni parola: pos accento dal fondo, numero sillabe della parola e posizione assoluta dell'accento

        syl_words = re.sub(r'<start>|<end>|\n', '', lines_y[i])
        syl_words = re.sub(r"'", "’", syl_words)
        syl_words = re.sub(r'<syl>', '', syl_words, count=1)
        syl_words = syl_words.split('<s><syl>')
        # syl_words è una lista di parole contenenti <syl> e <s> in caso di sinalefe
        formed_verse = np.zeros((11, 2))
        index = 0
        for j, w in enumerate(syl_words):
            syls = w.split('<syl>')
            for k in range(len(syls)):
                prop = [0, 0]
                if len(syls) + pos_acc_array[j][0] - 1 == k:
                    prop[0] = 1
                if k == len(syls) - 1:
                    prop[1] = 1
                formed_verse[index + k] = np.array(prop)
            index += len(syls)
        # formed_verse è un vettore di 11 elementi [accento?, ultima sillaba della parola?]
        pos = check_male_caesura(formed_verse)
        if pos == 0:
            pos = check_liric_caesura(formed_verse)
        if pos == 0:
            pos = check_female_caesura(formed_verse)
        new_line = '<start><syl>'
        counter = -1
        for j, w in enumerate(syl_words):
            counter += len(w.split('<syl>'))
            new_line += w
            if counter != pos:
                new_line += '<s>'
            else:
                new_line += '<c><s>'
            if j < len(syl_words) - 1:
                new_line += '<syl>'
        new_line += '<end>\n'
        new_lines.append(new_line)

    file_dest.writelines(new_lines)
    file_y.close()
    file_X.close()
    file_dest.close()


if __name__ == '__main__':
    main()
