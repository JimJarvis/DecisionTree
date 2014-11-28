def predict(example):
    if example[13] == 0:
        if example[12] == 0:
            if example[8] == 0:
                return 'shellfish'
            elif example[8] == 1:
                return 'reptile'
        elif example[12] == 1:
            if example[3] == 0:
                return 'mammal'
            elif example[3] == 1:
                return 'fish'
    elif example[13] == 2:
        if example[1] == 0:
            return 'bird'
        elif example[1] == 1:
            return 'mammal'
    elif example[13] == 4:
        if example[1] == 0:
            if example[6] == 0:
                return 'reptile'
            elif example[6] == 1:
                if example[8] == 0:
                    return 'shellfish'
                elif example[8] == 1:
                    return 'amphibian'
        elif example[1] == 1:
            return 'mammal'
    elif example[13] == 5:
        return 'shellfish'
    elif example[13] == 6:
        if example[6] == 0:
            return 'insect'
        elif example[6] == 1:
            return 'shellfish'
    elif example[13] == 8:
        return 'shellfish'
