import numpy as np
import h5py

names = {
    0: "Abdissa Degefu",
    1: "Abdurahman Muhammed",
    2: "Abraham Wendmeneh",
    3: "Amanuel Beyene",
    4: "Amir Ahmedin",
    5: "Ananiya_Tesfahun",
    6: "Betelhem Yimam",
    7: "Bethelhem Yemane",
    8: "Biniyam Haile",
    9: "Dagmawi_Tensay",
    10: "Dawit Getahun",
    11: "Dawit_Abebe",
    12: "Deribew_Shimels",
    13: "Ephrem_Shimels",
    14: "Esayas Nigussie",
    15: "Etsubdink Awoke",
    16: "Fasika_Fikadu",
    17: "Feven Tesfaye",
    18: "Fraol Mulugeta",
    19: "Gedion Ezra",
    20: "Geleta Daba",
    21: "Gelila Moges",
    22: "Gelila Tefera",
    23: "Husen Yusuf",
    24: "Kidus Hunegnaw",
    25: "Leul Degarege",
    26: "Leul Wujira",
    27: "Mariam Yohannes",
    28: "Melkishi Tesfaye",
    29: "Metsakal Zeleke",
    30: "Milion Tolesa",
    31: "Milka Fasika",
    32: "Naol Taye",
    33: "Nathnael Dereje",
    34: "Olyad Temesgen",
    35: "Sahib Semahegn",
    36: "Semir Hamid",
    37: "Shemsu Nurye",
    38: "Sosina Esayas",
    39: "Tewodros Berhanu",
    40: "Tinsae Shemalise",
    41: "Tiruzer Tsedeke",
    42: "Yanet Mekuria",
    43: "Yohannes Ahunm",
    44: "Yohannes Dessie",
    45: "Yohannes Desta",
    46: "Yonas Engedu",
    47: "Yosef Aweke",
    48: "Yosef Ayele",
    49: "Yosef Muluneh",
}


class fisherFaces:
    def __init__(self):
        with h5py.File("model.h5", "r") as file:
            self.D = file["D"][:]
            self.W = file["W"][:]
            self.mu = file["mu"][:]
            self.train_label = file["train_label"][:]
            self.projections = file["projection"][:]

    def project(self, W, data, mu=None):
        if mu is None:
            return np.dot(data, W)
        return np.dot(data - mu, W)

    def EuclideanDistance(self, p, q):
        p = np.asarray(p).flatten()
        q = np.asarray(q).flatten()
        return np.sqrt(np.sum(np.power((p - q), 2)))

    def predict(self, test_data):
        minDist = np.finfo("float").max
        minClass = -1

        Q = self.project(self.W, test_data.reshape(1, -1), self.mu)

        for i in range(len(self.projections)):
            dist = self.EuclideanDistance(self.projections[i], Q)
            if dist < minDist:
                minDist = dist
                minClass = self.train_label[i]
        return names[minClass]
