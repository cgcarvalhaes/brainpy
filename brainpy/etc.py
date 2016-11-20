from collections import defaultdict


def regroup_labels(labels, group_size):
    groups, new_labels = [], []
    pool = defaultdict(list)
    for index, lab in enumerate(labels):
        pool[lab].append(index)
        if len(pool[lab]) == group_size:
            groups.append(pool.pop(lab))
            new_labels.append(lab)
    for lab, indexes in pool.iteritems():
        groups.append(indexes)
        new_labels.append(lab)
    return groups, new_labels
