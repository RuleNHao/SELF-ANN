"""
@Author: Penghao Tian <rulenhao@mail.ustc.edu.cn>
@Date: 2022/11/27 04:36
@Description: IGRF代码, 基于pyIGRF库的numba重构

优势: 
通常精度0.5的全球网格计算需要时间 > 30分钟
用numba重构的代码计算需要时间 ~= 1秒

inclination = np.arctan2(Z, H) 
dip_latitude = np.arctan2(Z, 2*H)

reference:
Laundal, K.M., Richmond, A.D. Magnetic Coordinate Systems. Space Sci Rev 206, 27–59 (2017).
"""

import os
import numpy as np
from pathlib import Path
from numba import njit, prange, objmode

class ReadCoeffs():
    """
    读取igrf13coeffs.txt
    """
    def __init__(self, filename="./igrf13coeffs.txt"):
        self.gh = self._load_coeffs(filename)
        pass
    
    def _load_coeffs(self, filename):
        """
        load igrf12 coeffs from file
        :param filename: file which save coeffs (str)
        :return: g and h list one by one (list(float))
        """
        gh = []
        gh2arr = []
        with open(filename) as f:
            text = f.readlines()
            for a in text:
                if a[:2] == 'g ' or a[:2] == 'h ':
                    b = a.split()[3:]
                    b = [float(x) for x in b]
                    gh2arr.append(b)
            gh2arr = np.array(gh2arr).transpose()
            N = len(gh2arr)
            for i in range(N):
                if i < 19:
                    for j in range(120):
                        gh.append(gh2arr[i][j])
                else:
                    for p in gh2arr[i]:
                        gh.append(p)
            gh.append(0)
            pass
        return gh
    
    def _transform_gh(self, g, h):
        garray = np.zeros((len(g),14))
        harray = np.zeros((len(h),14))
        for i in range(len(g)):
            for j in range(len(g[i])):
                gvalue = g[i][j]
                if gvalue is None:
                    gvalue = 0.0
                garray[i, j] = gvalue
                pass
            pass
        
        for i in range(len(h)):
            for j in range(len(h[i])):
                hvalue = h[i][j]
                if gvalue is None:
                    gvalue = 0.0
                harray[i, j] = hvalue
                pass
            pass
        return garray, harray
    
    def get_coeffs(self, date):
        """
        :param gh: list from load_coeffs
        :param date: float
        :return: list: g, list: h
        """
        if date < 1900.0 or date > 2030.0:
            print('This subroutine will not work with a date of ' + str(date))
            print('Date must be in the range 1900.0 <= date <= 2030.0')
            print('On return [], []')
            return [], []
        elif date >= 2020.0:
            if date > 2025.0:
                # not adapt for the model but can calculate
                print('This version of the IGRF is intended for use up to 2025.0.')
                print('values for ' + str(date) + ' will be computed but may be of reduced accuracy')
            t = date - 2020.0
            tc = 1.0
            #     pointer for last coefficient in pen-ultimate set of MF coefficients...
            ll = 3060+195
            nmx = 13
            nc = nmx * (nmx + 2)
        else:
            t = 0.2 * (date - 1900.0)
            ll = int(t)
            t = t - ll
            #     SH models before 1995.0 are only to degree 10
            if date < 1995.0:
                nmx = 10
                nc = nmx * (nmx + 2)
                ll = nc * ll
            else:
                nmx = 13
                nc = nmx * (nmx + 2)
                ll = int(0.2 * (date - 1995.0))
                #     19 is the number of SH models that extend to degree 10
                ll = 120 * 19 + nc * ll
            tc = 1.0 - t

        g, h = [], []
        temp = ll-1
        for n in range(nmx+1):
            g.append([])
            h.append([])
            if n == 0:
                g[0].append(None)
            for m in range(n+1):
                if m != 0:
                    g[n].append(tc*self.gh[temp] + t*self.gh[temp+nc])
                    h[n].append(tc*self.gh[temp+1] + t*self.gh[temp+nc+1])
                    temp += 2
                    # print(n, m, g[n][m], h[n][m])
                else:
                    g[n].append(tc*self.gh[temp] + t*self.gh[temp+nc])
                    h[n].append(None)
                    temp += 1
                    # print(n, m, g[n][m], h[n][m])
                    pass
                pass
            pass
        return self._transform_gh(g, h)
    
    pass


@njit()
def geodetic2geocentric(theta, alt):
    """
    Conversion from geodetic to geocentric coordinates by using the WGS84 spheroid.
    :param theta: colatitude (float, rad)
    :param alt: altitude (float, km)
    :return gccolat: geocentric colatitude (float, rad)
            d: gccolat minus theta (float, rad)
            r: geocentric radius (float, km)
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    a2 = 40680631.6
    b2 = 40408296.0
    one = a2 * st * st
    two = b2 * ct * ct
    three = one + two
    rho = np.sqrt(three)
    r = np.sqrt(alt * (alt + 2.0 * rho) + (a2 * one + b2 * two) / three)
    cd = (alt + rho) / r
    sd = (a2 - b2) / rho * ct * st / r
    one = ct
    ct = ct * cd - st * sd
    st = st * cd + one * sd
    gccolat = np.arctan2(st, ct)
    d = np.arctan2(sd, cd)
    return gccolat, d, r

@njit()
def igrf12syn(g, h, date, itype, alt, lat, elong):
    """
     This is a synthesis routine for the 12th generation IGRF as agreed
     in December 2014 by IAGA Working Group V-MOD. It is valid 1900.0 to
     2020.0 inclusive. Values for dates from 1945.0 to 2010.0 inclusive are
     definitive, otherwise they are non-definitive.
   INPUT
     date  = year A.D. Must be greater than or equal to 1900.0 and
             less than or equal to 2025.0. Warning message is given
             for dates greater than 2020.0. Must be double precision.
     itype = 1 if geodetic (spheroid)
     itype = 2 if geocentric (sphere)
     alt   = height in km above sea level if itype = 1
           = distance from centre of Earth in km if itype = 2 (>3485 km)
     lat = latitude (-90~90)
     elong = east-longitude (0-360)
     alt, colat and elong must be double precision.
   OUTPUT
     x     = north component (nT) if isv = 0, nT/year if isv = 1
     y     = east component (nT) if isv = 0, nT/year if isv = 1
     z     = vertical component (nT) if isv = 0, nT/year if isv = 1
     f     = total intensity (nT) if isv = 0, rubbish if isv = 1
     To get the other geomagnetic elements (D, I, H and secular
     variations dD, dH, dI and dF) use routines ptoc and ptocsv.
     Adapted from 8th generation version to include new maximum degree for
     main-field models for 2000.0 and onwards and use WGS84 spheroid instead
     of International Astronomical Union 1966 spheroid as recommended by IAGA
     in July 2003. Reference radius remains as 6371.2 km - it is NOT the mean
     radius (= 6371.0 km) but 6371.2 km is what is used in determining the
     coefficients. Adaptation by Susan Macmillan, August 2003 (for
     9th generation), December 2004, December 2009, December 2014.
     Coefficients at 1995.0 incorrectly rounded (rounded up instead of
     to even) included as these are the coefficients published in Excel
     spreadsheet July 2005.
    """

    FACT = 180./np.pi
    p = np.zeros((105,))
    q = np.zeros((105,))
    cl = np.zeros((13,))
    sl = np.zeros((13,))

    x, y, z = 0., 0., 0.

    if date < 1900.0 or date > 2025.0:
        f = 1.0
        print('This subroutine will not work with a date of ' + str(date))
        print('Date must be in the range 1900.0 <= date <= 2025.0')
        print('On return f = 1.0, x = y = z = 0')
        return (x, y, z, f)
    
    # with objmode(g="float64[:,:]", h="float64[:,:]"):
    #     g, h = ReadCoeffs().get_coeffs(date)
    #     pass
    
    nmx = len(g)-1
    kmx = (nmx + 1) * (nmx + 2) // 2 + 1

    colat = 90-lat
    r = alt
    
    one = colat / FACT
    ct = np.cos(one)
    st = np.sin(one)

    one = elong / FACT
    cl[0] = np.cos(one)
    sl[0] = np.sin(one)

    cd = 1.0
    sd = 0.0

    l = 1
    m = 1
    n = 0

    if itype != 2:
        gclat, gclon, r = geodetic2geocentric(np.arctan2(st, ct), alt)
        ct, st = np.cos(gclat), np.sin(gclat)
        cd, sd = np.cos(gclon), np.sin(gclon)
    ratio = 6371.2 / r
    rr = ratio * ratio

    #     computation of Schmidt quasi-normal coefficients p and x(=q)
    p[0] = 1.0
    p[2] = st
    q[0] = 0.0
    q[2] = ct

    fn, gn = n, n-1
    for k in prange(2, kmx):
        if n < m:
            m = 0
            n = n + 1
            rr = rr * ratio
            fn = n
            gn = n - 1

        fm = m
        if m != n:
            gmm = m * m
            one = np.sqrt(fn * fn - gmm)
            two = np.sqrt(gn * gn - gmm) / one
            three = (fn + gn) / one
            i = k - n
            j = i - n + 1
            p[k - 1] = three * ct * p[i - 1] - two * p[j - 1]
            q[k - 1] = three * (ct * q[i - 1] - st * p[i - 1]) - two * q[j - 1]
        else:
            if k != 3:
                one = np.sqrt(1.0 - 0.5 / fm)
                j = k - n - 1
                p[k-1] = one * st * p[j-1]
                q[k-1] = one * (st * q[j-1] + ct * p[j-1])
                cl[m-1] = cl[m - 2] * cl[0] - sl[m - 2] * sl[0]
                sl[m-1] = sl[m - 2] * cl[0] + cl[m - 2] * sl[0]
        #     synthesis of x, y and z in geocentric coordinates
        one = g[n][m] * rr
        if m == 0:
            x = x + one * q[k - 1]
            z = z - (fn + 1.0) * one * p[k - 1]
            l = l + 1
        else:
            two = h[n][m] * rr
            three = one * cl[m-1] + two * sl[m-1]
            x = x + three * q[k-1]
            z = z - (fn + 1.0) * three * p[k-1]
            if st == 0.0:
                y = y + (one * sl[m - 1] - two * cl[m - 1]) * q[k - 1] * ct
            else:
                y = y + (one * sl[m-1] - two * cl[m-1]) * fm * p[k-1] / st
            l = l + 2
        m = m+1

    #     conversion to coordinate system specified by itype
    one = x
    x = x * cd + z * sd
    z = z * cd - one * sd
    f = np.sqrt(x * x + y * y + z * z)
    return (x, y, z, f)


@njit()
def igrf_value(g, h, lat, lon, alt, year):
    """
    因为计算全球网格分布时 通常year是固定的
    g, h也是固定的 因此可以和计算解耦
    """
    
    """
    :return
         D is declination (+ve east)
         I is inclination (+ve down)
         H is horizontal intensity
         X is north component
         Y is east component
         Z is vertical component (+ve down)
         F is total intensity
         Dip latitude is arctan(Z / (2*H))
    """
    FACT = 180./np.pi
    itype = 1
    x, y, z, f = igrf12syn(g, h, year, itype, alt, lat, lon)
    d = FACT * np.arctan2(y, x)
    h = np.sqrt(x * x + y * y)
    i = FACT * np.arctan2(z, h)
    dip_latitude = FACT * np.arctan2(z, 2*h)
    return dip_latitude, d, i, h, x, y, z, f

@njit()
def igrf_variation(g, h, lat, lon, alt, year):
    """
         Annual variation
         D is declination (+ve east)
         I is inclination (+ve down)
         H is horizontal intensity
         x is north component
         y is east component
         Z is vertical component (+ve down)
         F is total intensity
    """
    FACT = 180./np.pi
    x1, y1, z1, f1 = igrf12syn(g, h, year-1, 1, alt, lat, lon)
    x2, y2, z2, f2 = igrf12syn(g, h, year+1, 1, alt, lat, lon)
    x, y, z, f = (x1+x2)/2, (y1+y2)/2, (z1+z2)/2, (f1+f2)/2
    dx, dy, dz, df = (x2-x1)/2, (y2-y1)/2, (z2-z1)/2, (f2-f1)/2
    h = np.sqrt(x * x + y * y)

    dd = (FACT * (x * dy - y * dx)) / (h * h)
    dh = (x * dx + y * dy) / h
    ds = (FACT * (h * dz - z * dh)) / (f * f)
    df = (h * dh + z * dz) / f
    return dd, ds, dh, dx, dy, dz, df


class MAIN():
    
    def main(self):
        lat = 31.81
        lon = 117.28
        alt = 0
        year = 2022
        
        # with objmode(g="float64[:,:]", h="float64[:,:]"):
        #     g, h = ReadCoeffs().get_coeffs(year)
        #     pass
        g, h = ReadCoeffs().get_coeffs(year)
        
        d, i, h, x, y, z, f = igrf_value(g, h, lat, lon, alt, year)
        print(f"latitude: {lat}")
        print(f"longitude: {lon}")
        print(f"altitude (above sea level): {alt} km")
        print(f"year: {year}")
        print("===== IGRF 13th =====")
        print(f"declination (+ve east): {d}\ninclination (+ve down): {i}\nhorizontal intensity: {h} nT")
        print(f"north compontent: {x} nT\nease component: {y} nT\nvertical component (+ve down): {z} nT")
        print(f"total intensity: {f} nT")
        return
    
    pass


if __name__=="__main__":
    main = MAIN()
    main.main()
    pass
    
