import numpy as np
class EigenFaces:
    def __init__(self, eigenfaces, A=None, labels=None, num_components=0):
        self.num_components = 0
        self.projections = []
        self.threshold = 0
        self.U = eigenfaces
        self.mean_face = []
        if (A is not None) and (labels is not None):
            self.compute(A, labels)
    def distance(self, p, q):
        """
        Euclidean distance between two faces
        """
        p = np.asarray(p).flatten()
        q = np.asarray(q).flatten()
        return np.sqrt(np.sum(np.power((p - q), 2)))
    
    def compute(self, A, labels):
        [eigen_faces, weights, mean_face] = PCA(A)
        self.labels = labels
        self.mean_face = mean_face
        self.U = weights
        # print(self.mean_face.shape)
        self.projections = np.dot(self.U.T, A.T - self.mean_face[:, np.newaxis])
        diffs = self.projections[:, np.newaxis] - self.projections
    
        # Calculate norms for all differences
        norms =  np.linalg.norm(diffs, axis=-1)
    
        # Find the maximum norm
        self.threshold = 0.5 * np.max(norms)
        print("threshold", self.threshold)
    def reconstruct_faces(self, weights, Q):
        """
        Reconstruct the image vector a from face space
        """
        return np.dot(weights, Q)

    def predict(self, A):
        minDist = np.finfo('float').max
        minClass = -1
        predicted_face = []
        status = ""
        print("A.t shape in perdict", A.T.shape)
        Q = project_to_face_space(self.U, A.T, self.mean_face)
        for i in range(len(self.projections)):
            dist = self.distance(self.projections[i], Q)
            if dist < minDist:
                minDist = dist
                minClass = self.labels[i]
                predicted_face = self.projections[i]
        
        reconsturcted_face = self.reconstruct_faces(self.U, Q)
        e = LA.norm(np.abs(A.T - self.mean_face - reconsturcted_face))
        if e >= self.threshold:
            status = "Not a face"
            print(status)
        elif e < self.threshold and minDist < self.threshold:
            status = "New face"
            print(status)
        elif e < self.threshold and minDist >= self.threshold:
            status = "Known face"
            print(status)
        
        return minClass, status
    
# eigenFaces = EigenFaces()