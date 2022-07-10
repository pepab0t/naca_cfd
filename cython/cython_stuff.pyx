import numpy as np
import concurrent.futures
from entities import Line
cimport numpy

cpdef double vector_mul_cy(numpy.ndarray[numpy.float64_t, ndim=1] a, numpy.ndarray[numpy.float64_t, ndim=1] b):
    return b[1]*a[0] - b[0]*a[1]

cpdef double magnitude_cy(numpy.ndarray[numpy.float64_t, ndim=1] v):
    return np.sqrt(v[0]**2 + v[1]**2)

cpdef bint is_outside_cy(
    numpy.ndarray[numpy.float64_t, ndim=2] profile,
    numpy.ndarray[numpy.float64_t, ndim=1] point
):
    cdef double sum_angle = 0.0
    cdef int i
    cdef double alpha

    for i in range(1,len(profile)):
        a = profile[i-1, :] - point
        b = profile[i, :] - point
        # print(profile_points[i-1, :], profile_points[i, :])
        alpha = np.arcsin( vector_mul_cy(a, b) / (magnitude_cy(a) * magnitude_cy(b))) * 180/np.pi
        # print(alpha)
        # self.disp(camber=False, show=False)
        # plt.plot([point[0], point[0]+a[0]], [point[1], point[1]+a[1]], 'r')
        # plt.plot([point[0], point[0]+b[0]], [point[1], point[1]+b[1]], 'b')
        # plt.show()
        sum_angle += alpha

    return sum_angle < 180

cpdef double shortest_distance_cy(
    numpy.ndarray[numpy.float64_t, ndim=2] profile,
    numpy.ndarray[numpy.float64_t, ndim=1] point
):


    cdef double d = 1e6
    cdef double d_new

    for i in range(1, len(profile)):
        l = Line(profile[i-1,:], profile[i,:])
        d_new = l.distance_segment(point)
        if d_new < d:
            d = d_new 

    d = (-1)**int(not is_outside_cy(profile, point)) * d

    return d


cpdef numpy.ndarray[numpy.float64_t, ndim=2] createDistField(
    numpy.ndarray[numpy.float64_t, ndim=2] profile,
    # numpy.ndarray[numpy.float64_t, ndim=1] point,
    numpy.ndarray[numpy.float64_t, ndim=2] field):

    with concurrent.futures.ProcessPoolExecutor() as executor:
        procs = [executor.submit(shortest_distance_cy, profile, point) for point in field]

    cdef numpy.ndarray[numpy.float64_t, ndim=2] out
    out = np.zeros(shape=(field.shape[0], 3), dtype=np.float)
    out[:,0:2] = field.copy()

    cdef int i

    for i in range(len(procs)):
        out[i,2] = procs[i].result()

    return out
