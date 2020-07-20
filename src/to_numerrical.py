from sklearn import preprocessing


def make_it_numerical(name_data):
    le = preprocessing.LabelEncoder()
    name_data["cut"] = le.fit_transform(name_data["cut"])
    name_data["color"] = le.fit_transform(name_data["color"])
    name_data["clarity"] = le.fit_transform(name_data["clarity"])
    return name_data