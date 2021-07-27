def levenshtein(string_1: str, string_2: str) -> int:
    if len(string_1) < len(string_2):
        return levenshtein(string_2, string_1)

    if len(string_2) == 0:
        return len(string_1)

    previous_row = range(len(string_2) + 1)
    for i, c1 in enumerate(string_1):
        current_row = [i + 1]
        for j, c2 in enumerate(string_2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]
