# coding:utf-8

def compare():
    pred = 'output.txt'
    right = 'test.raw.tgt'
    wrong = "test.raw.src"
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    with open(pred, 'r', encoding='utf-8') as f1:
        with open(right, 'r', encoding='utf-8') as f2:
            with open(wrong, 'r', encoding='utf-8') as f5:
                sentence_list = []
                for sentence in f1.readlines():
                    sen = "".join(sentence.split(" "))
                    sentence_list.append(sen)
                for (pred, wrong, right) in zip(sentence_list, f5.readlines(), f2.readlines()):
                    if len(wrong.strip()) != len(right.strip()):
                        continue
                    if len(pred.strip()) != len(wrong.strip()):
                       # fp = fp + abs(len(pred.strip())-len(wrong.strip()))
                       pass
                    else:
                        for i in range(len(wrong.strip())):
                #    print(pred, wrong, right)
                            if wrong.strip()[i] != right.strip()[i] and wrong.strip()[i] != pred.strip()[i] and right.strip()[i] == pred.strip()[i]:
                                tp = tp + 1
                                print(wrong.strip()[i],right.strip()[i],pred.strip()[i])
                            if wrong.strip()[i] == right.strip()[i] and wrong.strip()[i] != pred.strip()[i] and pred.strip()[i] != right.strip()[i]:
                                fp = fp + 1
                                #print(pred.strip())
                                #print(right.strip())
                            if wrong.strip()[i] == right.strip()[i] and pred.strip()[i] == wrong.strip()[i]:
                                tn = tn + 1
                            if wrong.strip()[i] != right.strip()[i] and pred.strip()[i] == wrong.strip()[i]:
                                fn = fn + 1
    print(tp,fp,fn,tn)
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * p * r / (r + p)
    print("pre:", p, "rec:", r, "F1:", f1)


if __name__ == "__main__":
     compare()
