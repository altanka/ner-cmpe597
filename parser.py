

def get_dataset():
    starting_tag = '<b_enamex'
    ending_tag = '<e_enamex>'
    all_words = []

    with open('data/ner_data.txt') as f:
        ner_data = f.read()
        for line in ner_data.split('\n'):
            line_has_tag = ending_tag in line
            if line_has_tag:
                tagged_lines = line.split(ending_tag)
                words = []
                tags = []
                for tagged_line in tagged_lines:
                    if starting_tag in tagged_line:
                        starting_tag_index = tagged_line.index('<')
                        ending_tag_index = tagged_line.index('>')

                        untagged_line = tagged_line[0:starting_tag_index]
                        untagged_words = untagged_line.split()
                        words.extend(untagged_words)
                        tags.extend(['O'] * len(untagged_words))

                        tagged_word = tagged_line[ending_tag_index + 1:]
                        tag_type_index = tagged_line.index('TYPE=')
                        tag_type = tagged_line[tag_type_index +
                                               6: ending_tag_index - 1]
                        if ' ' in tagged_word:
                            tagged_words = tagged_word.split()
                            words.extend(tagged_words)
                            tags.append(tag_type + 'B')
                            tags.extend([tag_type + 'I'] *
                                        (len(tagged_words) - 1))
                        else:
                            words.append(tagged_word)
                            tags.append(tag_type + 'B')
                    else:
                        untagged_words = tagged_line.split()
                        words.extend(untagged_words)
                        tags.extend(['O'] * len(untagged_words))
                all_words.append((words, tags))
    return all_words
