import re


def main(do_x):
    file_orig = open('resources/orig/full_divina.txt', 'r+', encoding='utf-8')
    if do_x:
        file_dest = open('resources/X.csv', 'w+', encoding='utf-8')
    else:
        file_dest = open('resources/y.csv', 'w+', encoding='utf-8')

    lines_orig = file_orig.readlines()
    lines_dest = re.sub(r'([,])|([.])|([;])|([:])|([!])|([?])|'
                        r'([-])|([«])|([»])|(["])|([“])|([‟])|'
                        r'([”])|(\()|(\))|(\[)|(\])|(—)',
                        "", ''.join(lines_orig).lower())
    lines_dest = re.sub(r'(’)|(‘)', '\'', lines_dest)
    lines_dest = re.sub(r' \' ', ' i ', lines_dest)
    lines_dest = re.sub(r'.*\d', '', lines_dest)
    lines_dest = re.sub(r'.*• canto.*', '', lines_dest)
    # lines_dest = re.sub(r'ä', 'a', lines_dest)
    # lines_dest = re.sub(r'ë', 'e', lines_dest)
    # lines_dest = re.sub(r'ï', 'i', lines_dest)
    # lines_dest = re.sub(r'ö', 'o', lines_dest)
    # lines_dest = re.sub(r'ü', 'u', lines_dest)
    lines_dest = re.sub(r'á', 'à', lines_dest)
    lines_dest = re.sub(r'í', 'ì', lines_dest)
    lines_dest = re.sub(r'ó', 'ò', lines_dest)
    lines_dest = re.sub(r'ú', 'ù', lines_dest)
    lines_dest = re.sub(r'k', 'c', lines_dest)
    lines_dest = re.sub(r'j', 'g', lines_dest)
    # lines_dest = re.sub(r'x', 'cs', lines_dest)
    lines_dest = re.sub(r'y', 'i', lines_dest)

    # lines_dest = re.sub(r'(\w+) \'(\w+)', r"\1'\2", lines_dest)  # !!!
    # lines_dest = re.sub(r'(\w+)\' (\w+)', r"\1'\2", lines_dest)  # !!!
    lines_dest = re.sub(r'\n\n\n', '\n', lines_dest)
    lines_dest = re.sub(r'\n\n\n', '\n', lines_dest)
    lines_dest = re.sub(r'\n\n', '\n', lines_dest)
    lines_dest = re.sub(r'\n\n', '\n', lines_dest)
    lines_dest = re.sub(r'\n ', '\n', lines_dest)
    lines_dest = re.sub(r'\n ', '\n', lines_dest)
    lines_dest = re.sub(r'\n ', '\n', lines_dest)
    lines_dest = re.sub(r'\n ', '\n', lines_dest)
    lines_dest = re.sub(r'^\n', '', lines_dest)
    # lines_dest = re.sub(r'\n\|', '\n', lines_dest)
    # lines_dest = re.sub(r'^\|', '', lines_dest)
    lines_dest = re.sub(r'\n', ' \n', lines_dest)

    lines_dest = re.sub(r'(\w)è(\w)', r'\1e\2', lines_dest)
    lines_dest = re.sub(r'(\w)à(\w)', r'\1a\2', lines_dest)
    lines_dest = re.sub(r'(\w)ù(\w)', r'\1u\2', lines_dest)
    lines_dest = re.sub(r'(\w)ì(\w)', r'\1i\2', lines_dest)
    lines_dest = re.sub(r'(\w)ò(\w)', r'\1o\2', lines_dest)
    # lines_dest = re.sub(r'(\w+i)\'', r'\1i', lines_dest)
    lines_dest = re.sub(r'[ ]+', r' ', lines_dest)

    lines_dest = re.sub(r' $', r'\n', lines_dest)

    if do_x:
        lines_dest = re.sub(r'\|', r'', lines_dest)
        lines_dest = re.sub(r' \n', r'\n', lines_dest)
        lines_dest = re.sub(r' ', r'<s>', lines_dest)
        lines_dest = re.sub(r'^', r'<start>', lines_dest)
        lines_dest = re.sub(r'\n', r'\n<start>', lines_dest)
        lines_dest = re.sub(r'<start>$', r'', lines_dest)
        lines_dest = re.sub(r'\n', r'<end>\n', lines_dest)

    else:
        lines_dest = re.sub(r'\|', r'<syl>', lines_dest)
        lines_dest = re.sub(r' \n', r'\n', lines_dest)
        lines_dest = re.sub(r' ', r'<s>', lines_dest)
        # lines_dest = re.sub(r'(^-)|(\n-)', r'\n', lines_dest)

    file_dest.writelines(lines_dest)

    file_dest.close()
    file_orig.close()


if __name__ == '__main__':
    main(do_x=False)
    main(do_x=True)
