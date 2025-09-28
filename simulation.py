import itertools
import numpy as np
from numpy import ndarray
from typing import Tuple, Union
from scipy import stats, integrate
from bisect import bisect_left
import warnings


class Simulation:
    def __init__(self, gamma: float, k: float, l_0: float, R: float, R_spring: float,
                 particles_cnt: int, spring_cnt: int,
                 T: float,
                 m: ndarray, m_spring: ndarray):
        self._k_boltz = 1.380 * 1e-2
        self._gamma = gamma
        self._k = k
        self._l_0 = l_0
        self._R = R
        self._R_spring = R_spring
        r = np.random.uniform(size=(2, particles_cnt))
        r_spring = Simulation._sample_r_sping(spring_cnt, k, self._k_boltz, l_0, gamma, T)
        self._r = np.hstack([r_spring, r])
        v = stats.norm.rvs(loc=0.0, scale=np.sqrt(self._k_boltz*T / m), size=(2, particles_cnt))
        v_spring = stats.norm.rvs(loc=0.0, scale=np.sqrt(self._k_boltz * T / m_spring), size=(2, spring_cnt))
        self._v = np.hstack([v_spring, v])
        self._m = np.hstack([m_spring, m])
        self._n_particles = particles_cnt
        self._n_spring = spring_cnt

        self._init_ids_pairs()

        self._potential_energy = []
        self._kinetic_energy = []

        self._E_full = self.calc_full_energy()
        self._T_tar = self.T
        self._frame_no = 1

    def _init_ids_pairs(self):
        spring_ids = np.arange(self._n_spring)
        self._spring_ids_pairs = np.asarray(list(itertools.combinations(spring_ids, 2)))

        particles_ids = np.arange(self._n_particles) + self._n_spring
        self._particles_ids_pairs = np.asarray(list(itertools.combinations(particles_ids, 2)))

        self._spring_particles_ids_paris = np.asarray(list(itertools.product(spring_ids, particles_ids)))

        self._available_spring_ids_pairs = self._spring_ids_pairs.copy()
        self._available_particles_ids_pairs = self._particles_ids_pairs.copy()
        self._available_spring_particles_ids_paris = self._spring_particles_ids_paris.copy()

    @staticmethod
    def _sample_r_sping(spring_cnt: int, k: float, k_boltz: float, l_0: float, gamma: float, T: float):
        assert spring_cnt == 2, "Not implemented"

        def f(l):
            return np.exp(-k * (np.abs(l - l_0) ** (gamma + 1)) / (2 * k_boltz * T * (gamma + 1)))

        r_sp = np.linspace(0, 0.5, num=10_000)
        F = integrate.cumulative_trapezoid(y=f(r_sp), x=r_sp, initial=0)
        const = F[-1]
        F /= const

        print(f"{f(0.5)=}\t{f(1.0)=}")

        un = np.random.rand(1)
        l_between = r_sp[bisect_left(F, un)]

        print(f"{l_between=}")

        r_0 = np.array([0.5, 0.5])
        phi = np.random.rand() * 2*np.pi
        r_1 = r_0 + l_between * np.array([np.cos(phi), np.sin(phi)])

        return np.vstack([r_0, r_1]).T

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[ndarray, ndarray, ndarray, ndarray, float]:
        f = self.motion(dt=0.00001)
        self._frame_no = (self._frame_no + 1) % 5

        self._potential_energy.append(self.calc_potential_energy())
        self._kinetic_energy.append(self.calc_kinetic_energy())

        if self._frame_no == 0:
            self._fix_energy()

        # with open("E_dump.txt", "a") as fl:
        #     print(f"{self.calc_full_energy():.4f}", file=fl)
        #
        # with open("T_dump.txt", "a") as fl:
        #     print(f"{self.T:5.2f}", file=fl)

        return self.r, self.r_spring, self.v, self.v_spring, f

    @property
    def T(self) -> float:
        return np.mean(((np.linalg.norm(self._v, axis=0) ** 2) * self._m)) / (2 * self._k_boltz)

    @T.setter
    def T(self, val: float):
        if val <= 0:
            raise ValueError("T  must be > 0")
        delta = val / self._T_tar
        self._v *= np.sqrt(delta)
        self._E_full = self.calc_full_energy()
        self._T_tar = val

    @property
    def gamma(self) -> float:
        return self._gamma

    @gamma.setter
    def gamma(self, val: float):
        self._gamma = val
        self._E_full = self.calc_full_energy()

    @property
    def k(self) -> float:
        return self._k

    @k.setter
    def k(self, val: float):
        self._k = val
        self._E_full = self.calc_full_energy()

    @property
    def l_0(self) -> float:
        return self._l_0

    @l_0.setter
    def l_0(self, val: float):
        self._l_0 = val
        self._E_full = self.calc_full_energy()

    @property
    def R(self) -> float:
        return self._R

    @R.setter
    def R(self, val: float):
        self._R = val

    @property
    def R_spring(self) -> float:
        return self._R_spring

    @R_spring.setter
    def R_spring(self, val: float):
        self._R_spring = val

    @property
    def r(self) -> ndarray:
        return self._r[:, self._n_spring:]

    @property
    def r_spring(self) -> ndarray:
        return self._r[:, :self._n_spring]

    @property
    def v(self) -> ndarray:
        return self._v[:, self._n_spring:]

    @property
    def v_spring(self) -> ndarray:
        return self._v[:, :self._n_spring]

    @property
    def m(self) -> ndarray:
        return self._m[self._n_spring:]

    @property
    def m_spring(self) -> ndarray:
        return self._m[:self._n_spring]

    @staticmethod
    def get_deltad2_pairs(r, ids_pairs):
        dx = np.diff(np.stack([r[0][ids_pairs[:, 0]], r[0][ids_pairs[:, 1]]]).T).squeeze()
        dy = np.diff(np.stack([r[1][ids_pairs[:, 0]], r[1][ids_pairs[:, 1]]]).T).squeeze()
        return dx ** 2 + dy ** 2

    @staticmethod
    def compute_new_v(v1, v2, r1, r2, m1, m2) -> Tuple[ndarray, ndarray]:
        m_s = m1 + m2
        dr = r1 - r2
        dr_norm_sq = np.linalg.norm(dr, axis=0) ** 2

        v1new = v1 - (np.sum((2 * m2 / m_s) * (v1 - v2) * dr, axis=0) * dr) / dr_norm_sq
        v2new = v2 - (np.sum((2 * m1 / m_s) * (v2 - v1) * dr, axis=0) * dr) / dr_norm_sq

        return v1new, v2new

    @staticmethod
    def _update_available(arr: ndarray, sub_arr: ndarray) -> ndarray:
        eq_mat = sub_arr.T[None, ...] == arr[..., None]
        na_idx = np.any(np.all(eq_mat, axis=1), axis=1)
        return arr[~na_idx, :]

    def motion(self, dt) -> float:
        if self._available_spring_ids_pairs.shape[0]:
            ic_spring = self._available_spring_ids_pairs[
                np.array([self.get_deltad2_pairs(self._r, self._available_spring_ids_pairs)])
                < (2 * self.R_spring) ** 2]
        else:
            ic_spring = np.zeros((0, 2), dtype=int)

        if self._available_particles_ids_pairs.shape[0]:
            ic_particles = self._available_particles_ids_pairs[
                self.get_deltad2_pairs(self._r, self._available_particles_ids_pairs) < (2 * self.R) ** 2]
        else:
            ic_particles = np.zeros((0, 2), dtype=int)

        if self._available_spring_particles_ids_paris.shape[0]:
            ic_spring_particles = self._available_spring_particles_ids_paris[
                self.get_deltad2_pairs(self._r, self._available_spring_particles_ids_paris) < (self.R + self.R_spring) ** 2]
        else:
            ic_spring_particles = np.zeros((0, 2), dtype=int)

        self._available_spring_ids_pairs = self._update_available(
            self._spring_ids_pairs, ic_spring
        )

        self._available_particles_ids_pairs = self._update_available(
            self._particles_ids_pairs, ic_particles
        )

        self._available_spring_particles_ids_paris = self._update_available(
            self._spring_particles_ids_paris, ic_spring_particles
        )

        ic = np.vstack([
            ic_spring,
            ic_particles,
            ic_spring_particles
        ])
        # print(f"DEBUG: {ic.shape}")

        self._v[:, ic[:, 0]], self._v[:, ic[:, 1]] = self.compute_new_v(
            self._v[:, ic[:, 0]], self._v[:, ic[:, 1]],
            self._r[:, ic[:, 0]], self._r[:, ic[:, 1]],
            self._m[ic[:, 0]], self._m[ic[:, 1]]
        )

        dr = self.r_spring[:, 0] - self.r_spring[:, 1]
        dr_sc = np.linalg.norm(dr)
        dx = dr * (1 - self.l_0 / dr_sc)
        dx_norm = np.abs(dr_sc - self.l_0)
        f = (self._k * (dx_norm ** (self._gamma - 1))) * dx
        self.v_spring[:, 0] -= f * (dt / self.m_spring[0])
        self.v_spring[:, 1] += f * (dt / self.m_spring[1])

        self._v[0, self._r[0] > 1] = -np.abs(self._v[0, self._r[0] > 1])
        self._v[0, self._r[0] < 0] = np.abs(self._v[0, self._r[0] < 0])
        self._v[1, self._r[1] > 1] = -np.abs(self._v[1, self._r[1] > 1])
        self._v[1, self._r[1] < 0] = np.abs(self._v[1, self._r[1] < 0])

        self._r = self._r + self._v * dt

        return f @ dr

    def add_particles(self, r: ndarray, v: ndarray, m: ndarray):
        if (r.shape != v.shape) or (r.shape[0] != self._r.shape[0]) or (r.shape[1] != m.shape[0]):
            raise ValueError("Incorrect shape")
        self._r = np.hstack([self._r, r])
        self._v = np.hstack([self._v, v])
        self._m = np.hstack([self._m, m])

        self._E_full = self.calc_full_energy()
        self._T_tar = self.T

        self._init_ids_pairs()

    def _set_particles_cnt(self, particles_cnt: int):
        if particles_cnt < 0:
            raise ValueError("particles_cnt must be >= 0")

        if particles_cnt < self._n_particles:
            idx = slice(self._n_spring + particles_cnt)
            self._r = self._r[:, idx]
            self._v = self._v[:, idx]
            self._m = self._m[idx]
        if particles_cnt > self._n_particles:
            new_cnt = particles_cnt - self._n_particles
            self.add_particles(
                r=np.random.uniform(size=(2, new_cnt)),
                v=np.full(shape=(new_cnt, 2), fill_value=np.std(self.v, axis=1)).T,
                m=np.full(shape=(new_cnt, ), fill_value=np.median(self.m))
            )

        if particles_cnt != self._n_particles:
            self._n_particles = particles_cnt
            self._init_ids_pairs()

        self._E_full = self.calc_full_energy()
        self._T_tar = self.T

    def set_params(self,
                   gamma: float = None, k: float = None, l_0: float = None,
                   R: float = None, R_spring: float = None, T: float = None,
                   m: float = None, m_spring: float = None,
                   particles_cnt: int = None):
        if gamma is not None:
            self.gamma = gamma
        if k is not None:
            self.k = k
        if l_0 is not None:
            self.l_0 = l_0
        if R is not None:
            self.R = R
        if R_spring is not None:
            self.R_spring = R_spring
        if T is not None:
            self.T = T
        if m is not None:
            if m <= 0:
                raise ValueError("m_scale must be > 0")
            self._m[self._n_spring:] = m
        if m_spring is not None:
            if m_spring <= 0:
                raise ValueError("m_spring_scale must be > 0")
            self._m[0:self._n_spring] = m_spring
        if particles_cnt is not None:
            self._set_particles_cnt(particles_cnt)

        self._E_full = self.calc_full_energy()
        self._T_tar = self.T

    def expected_potential_energy(self) -> float:
        return float((self._k_boltz * self._T_tar) / (self.gamma + 1))

    def expected_kinetic_energy(self) -> float:
        return float(self._k_boltz * self._T_tar)

    def calc_kinetic_energy(self) -> float:
        return np.mean((np.linalg.norm(self.v_spring, axis=0) ** 2) * self.m_spring) / 2

    def calc_full_kinetic_energy(self):
        E_spring = np.sum((np.linalg.norm(self.v_spring, axis=0) ** 2) * self.m_spring) / 2
        E_particles = np.sum((np.linalg.norm(self.v, axis=0) ** 2) * self.m) / 2
        return float(E_spring + E_particles)

    def _fix_energy(self):
        E_par = np.sum((np.linalg.norm(self.v, axis=0) ** 2) * self.m) / 2
        beta = (self._E_full - 2 * self.calc_potential_energy() - self._n_spring*self.calc_kinetic_energy()) / E_par
        self._v[:, self._n_spring:] *= np.sqrt(beta)
        # print(f"DEBUG: {self._E_full - self.calc_full_energy()}")

    def calc_full_energy(self):
        return self.calc_full_kinetic_energy() + 2 * self.calc_potential_energy()

    def calc_potential_energy(self) -> float:
        dr = self.r_spring[:, 0] - self.r_spring[:, 1]
        dr_sc = np.linalg.norm(dr)
        dx_norm = np.abs(dr_sc - self.l_0)
        return self._k * (dx_norm ** (self._gamma + 1)) / (self._gamma + 1)

    def mean_potential_energy(self, frames_c: Union[int, None] = None) -> float:
        """
        :param frames_c: if frames_c is None then the averaging is taken over all frames,
        otherwise the averaging is taken over the last frame_c frames
        """
        if frames_c is None:
            return float(np.mean(self._potential_energy))
        else:
            return float(np.mean(self._potential_energy[-frames_c:]))

    def mean_kinetic_energy(self, frames_c: Union[int, None] = None) -> float:
        """
        :param frames_c: if frames_c is None then the averaging is taken over all frames,
        otherwise the averaging is taken over the last frame_c frames
        """
        if frames_c is None:
            return float(np.mean(self._kinetic_energy))
        else:
            return float(np.mean(self._kinetic_energy[-frames_c:]))
