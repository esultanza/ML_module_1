DATA_FOLDER = "data"

def getData(file: str):
    file_name = f"{DATA_FOLDER}/{file}"
                #data_elephants_rhinos_100.txt"

    with(open(file_name, encoding="utf-8")) as f:
        text = f.read()
    # print(text)

    # Извлекаем данные о животных в привычном для питона виде
    animals= []
    lines = text.split("\n")
    for line in lines:
        splitted = line.split("\t")
        # print(splitted)
        if len(splitted) == 3:
            animals.append({
                "type": splitted[0],
                "age": int(splitted[1]),
                "weight": int(splitted[2])
            })

    #print(animals)

    # алгоритмы МО обычно ждут данные в своем стиле
    # 1. Целевой вектор (targets, labels)
    labels = [a["type"] for a in animals]
    #print(labels)
    # 2. Признаки (features)
    features = [[a["age"], a["weight"]] for a in animals]
    #print(features)
    # 3. Классы (для классификации) = неповторяющиеся лейблы
    classes = sorted(list(set(labels)))
    #print(classes)
    return animals, labels, features, classes

if __name__ == "__main__":
    animals, labels, features, classes = getData("data_elephants_rhinos_100.txt")
    print(animals)
    print(labels)
    print(features)
    print(classes)
