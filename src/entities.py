import numpy as np

class Vector:
    # def __init__(self, startpoint, endpoint):
        
    @staticmethod
    def scalar_mul(a: np.ndarray, b: np.ndarray) -> float:
        Line._validate_entity(a)
        Line._validate_entity(b)

        return a[0]*b[0] + a[1]*b[1]

    @staticmethod
    def vector_mul(a: np.ndarray, b: np.ndarray) -> float:

        Line._validate_entity(a)
        Line._validate_entity(b)

        return b[1]*a[0] - b[0]*a[1]

    @staticmethod
    def magnitude(a: np.ndarray) -> float:
        Line._validate_entity(a)

        return np.sqrt(a[0]**2 + a[1]**2)

    @staticmethod
    def angle(a: np.ndarray, b: np.ndarray) -> float:
        """Returns angle in degrees"""
        Line._validate_entity(a)
        Line._validate_entity(b)

        scalar = Vector().scalar_mul(a, b)
        x = scalar / (Vector.magnitude(a) * Vector.magnitude(b))
        if x > 1:
            x = 1
        return np.arccos(x) * 180/np.pi

    @staticmethod
    def rotate(a: np.ndarray, phi: float) -> np.ndarray:
        """Input phi in degrees"""
        phi = phi/180 * np.pi
        Line._validate_entity(a)

        out = np.zeros(2)
        out[0] = np.cos(phi) * a[0] - np.sin(phi) * a[1]
        out[1] = np.sin(phi) * a[0] + np.cos(phi) * a[1]

        return out

class Line:
    def __init__(self, startpoint: np.ndarray, endpoint: np.ndarray):
        self._validate_entity(startpoint)
        self._validate_entity(endpoint)
        self.startpoint: np.ndarray = startpoint
        self.endpoint: np.ndarray = endpoint
        self.coeffs: dict = self.count_coeffs(endpoint - startpoint, startpoint, direct=True)
        self.range: np.ndarray = np.sort(np.array([startpoint[0], endpoint[0]]))

    def __repr__(self):
        return f"line: {self.coeffs['a']}x + {self.coeffs['b']}y + {self.coeffs['c']} == 0"
    
    @staticmethod
    def _validate_entity(entity: np.ndarray) -> bool:
        if not isinstance(entity, np.ndarray):
            raise TypeError('ENTITY must be an (2,) numpy array.')
        if len(entity.shape) == 1 and len(entity) == 2:
            return True
        else:
            raise ValueError("Unexpected length of ENTITY array.")

    def normal_vector(self, direct_vector: np.ndarray):
        # VALIDATION
        self._validate_entity(direct_vector)
        
        x = direct_vector[0]
        y = direct_vector[1]

        return np.array([-y, x]) #/ np.sqrt(x**2 + y**2)

    def count_coeffs(self, vector: np.ndarray, point_on_line: np.ndarray, direct: bool=True) -> dict:
        # VALIDATION
        self._validate_entity(vector)
        self._validate_entity(point_on_line)

        if direct:
            vector = self.normal_vector(vector)

        a = vector[0].copy()
        b = vector[1].copy()
        c = -a * point_on_line[0] - b*point_on_line[1]

        return {'a': a, 'b': b, 'c': c}

    def in_range(self, point: np.ndarray) -> bool:
        self._validate_entity(point)

        normal_vector = self.startpoint - self.endpoint
        if normal_vector[1] == 0:
            x_range = np.sort(np.array([self.startpoint[0], self.endpoint[0]]))
            return x_range[0] <= point[0] and point[0] <= x_range[1]

        c1 = self.count_coeffs(normal_vector,self.startpoint, direct=False)
        y1 = lambda x: - c1['a'] / c1['b'] * x - c1['c'] / c1['b']

        c2 = self.count_coeffs(normal_vector,self.endpoint, direct=False)
        y2 = lambda x: - c2['a'] / c2['b'] * x - c2['c'] / c2['b']

        # del c1, c2

        # ktera z primek je vetsi
        if y1(0) > y2(0):
            return y2(point[0]) <= point[1] and point[1] <= y1(point[0])
        else:
            return y1(point[0]) <= point[1] and point[1] <= y2(point[0])

    def distance(self, point: np.ndarray):
        self._validate_entity(point)

        return abs(self.coeffs['a']*point[0] + self.coeffs['b']*point[1] + self.coeffs['c'])/ np.sqrt(self.coeffs['a']**2 + self.coeffs['b']**2)

    @staticmethod
    def distance_points(p1:np.ndarray, p2: np.ndarray):
        Line._validate_entity(p1)
        Line._validate_entity(p2)

        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def distance_segment(self, point: np.ndarray):
        self._validate_entity(point)

        if self.in_range(point):
            return self.distance(point)
        else:
            d1 = self.distance_points(self.startpoint, point)
            d2 = self.distance_points(self.endpoint, point)
            return min(d1, d2)