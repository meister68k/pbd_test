#
# PBDをシンプルにやり直し
#
# 「伸びない紐」を想定
# 伸びないのでXPBDでもPBDでも変わらない，らしい
# 一部コーディングにClaudeを使用した
#
# 本ソースコードはCC-0です
#

import os
import sys
import random
import math
import pprint

import numpy as np

os.environ["QT_API"] = "PySide6"

from PySide6.QtWidgets import QMainWindow, QApplication, QWidget, QVBoxLayout
from PySide6.QtCore import QTimer

from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib import colormaps
import matplotlib.patches as patches


def calc_rot_matrix(angle) :
    return np.array([
        [ math.cos(angle), -math.sin(angle), 0.0],
        [ math.sin(angle),  math.cos(angle), 0.0],
        [             0.0,              0.0, 1.0],
    ])


class Simulator:
    """
    PBD ロープシミュレータ（Direct Solver 拡張版）

    境界条件の切り替え:
        両端固定      : T=0  （デフォルト）
        片端固定+張力 : T=<float>[N]
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # 初期化
    # ------------------------------------------------------------------
    def init_param(self, num: int, T: float = 0.0):
        self.num     = num                  # 粒子数
        self.n_links = self.num - 1         # リンク数（拘束数）
        self.dt      = 0.001                 # [sec]
        self.substep = num                  # 接触用（伸び拘束は1回で収束）
        self.t       = 0.0
        self.T_ext   = T                    # 片端外力 [N]  bc='tension' 時のみ使用

        self.p1 = np.array([-1.0,  0.0,   0.0])
        self.p2 = np.array( [-0.1, 0.0, 0.0])

        # 円柱パラメータ（Z軸方向に軸を持つ無限円柱）
        self.cyl_center = np.array([0.0, 0.0, 0.0])  # 円柱軸の XY 座標
        self.cyl_radius = -self.p2[0]                # 円柱半径 [m]

        # 長方形柱パラメータ（Z軸方向に軸を持つ無限長方形柱）
        self.rect_hx     = -self.p2[0]                          # X 方向の半幅 [m]（0=円柱モード）
        self.rect_hy     = self.rect_hx * 2                     # Y 方向の半幅 [m]
        self.rect_theta = 0.0        # 初期角度（必要に応じて設定）

        # 確認表示用
        self.rect_corner = np.array([
            [-self.rect_hx, -self.rect_hy, 0.0],
            [ self.rect_hx, -self.rect_hy, 0.0],
            [ self.rect_hx,  self.rect_hy, 0.0],
            [-self.rect_hx,  self.rect_hy, 0.0],
            [-self.rect_hx, -self.rect_hy, 0.0],
        ])

        self.pivot = np.array([0.0, 0.0, 0.0])    # 回転中心

        self.omega = 2 * math.pi                            # 右端の回転角速度 [rad/s]

        self.mass = 0.050                   # ロープ質量

        # 粒子当たりの質量
        self.m  = self.mass / self.num
        # 逆質量 (num, 1)
        self.w = np.ones(self.num) / self.m

        if self.T_ext == 0.0 :
            self.w[0]  = 0.0                # 左端固定
        self.w[-1] = 0.0                    # 右端（xₙ）固定

        # 初期位置
        self.x = np.linspace(self.p1, self.p2, self.num)

        # 初期の長さ
        d = self.x[1:] - self.x[:-1]                      # (n_links, 3)
        l = np.linalg.norm(d, axis=1, keepdims=True)  # (n_links, 1)
        # 自然長：リンク数分 (n_links,)
        self.l0 = l.flatten() * 1.1

        self.v = np.zeros((self.num, 3))
        self.f = np.zeros((self.num, 3))

        # 減衰関係
        self.c_ext = 0.0                # 減衰係数を直接指定する場合はこれを使う

        # 流体抵抗パラメータ（空気）
        self.rho_f = 1.225    # 流体密度 [kg/m³]
        self.mu_f  = 1.81e-5  # 動粘性係数 [Pa·s]
        self.D     = 0.003    # ロープ直径 [m]
 
        # 内部粘性係数（ロープ素材の内部摩擦）
        self.c_int = 0.1   # [無次元]（リンク方向相対速度の減衰率）

    # ------------------------------------------------------------------
    # 外力の計算
    # ------------------------------------------------------------------
    def calc_force(self):
        self.f = np.zeros((self.num, 3))
        self.f += np.array([[0.0, -9.8 * self.m, 0.0]])  # 重力

        # bc='tension': 左端（x[0]）に -X 方向のテンション T を加える
        # T > 0 のとき (-1, 0, 0) 方向（マイナスX方向）に引っ張る
        if abs(self.T_ext) > 0.0:
            self.f[0] += self.T_ext * np.array([-1.0, 0.0, 0.0])
    # ------------------------------------------------------------------
    # 速度積分
    # ------------------------------------------------------------------
    def calc_velocity(self):
        self.v = self.v + self.dt * self.w.reshape(-1, 1) * self.f

    # ------------------------------------------------------------------
    # 流体抵抗力
    # ------------------------------------------------------------------
    def calc_drag_force(self):
        """
        円柱形状を仮定した流体抵抗力を計算し self.f に加算する。
 
        単位長さあたりの抗力（法線方向）:
            F = 0.5 * rho_f * C_D(Re) * D * dl * v^2
 
        Re = rho_f * |v| * D / mu_f
 
        C_D の近似（円柱）:
            Re < 1    : ストークス域  C_D = 8π/Re
            1 ≤ Re < 500  : 遷移域    C_D = 10 / Re^0.25
            500 ≤ Re < 2e5: 慣性域    C_D = 1.0
            Re ≥ 2e5  : 乱流域    C_D = 0.3
 
        空気中 D=10mm のとき Re はほぼ慣性域（1 m/s 以上で Re > 677）
        なので実用上 C_D ≈ 1.0 で十分な精度が出る。
 
        抗力は速度の逆方向（-v̂ 方向）に作用する。
        固定端は w=0 なので calc_velocity で自動的に無効化される。
        """
        dl   = np.linalg.norm(self.p2 - self.p1) / self.n_links  # リンク長 [m]
 
        v_norm = np.linalg.norm(self.v, axis=1, keepdims=True)    # (num, 1)
 
        # Re（ゼロ割り防止）
        Re = self.rho_f * v_norm * self.D / self.mu_f
        Re = np.maximum(Re, 1e-6)
 
        # C_D（円柱近似）
        CD = np.where(Re < 1,
                      8.0 * np.pi / Re,
             np.where(Re < 500,
                      10.0 / Re ** 0.25,
             np.where(Re < 2e5,
                      1.0,
                      0.3)))
 
        # 単位長さあたりの抗力係数 → 粒子1個分（リンク長 dl 担当）
        # F = 0.5 * rho_f * CD * D * dl * |v| * v（速度の2乗・方向付き）
        c_drag = 0.5 * self.rho_f * CD * self.D * dl  # (num, 1)
        drag   = -c_drag * v_norm * self.v             # (num, 3)
 
        self.f += drag

    def damp_velocity(self):
        """
        2種類のダンパを適用する。
 
        方法B（外部粘性）:
            空気抵抗・流体抵抗に相当。全粒子の速度に比例した減衰。
            dv = -c_ext * v * dt
            固定端（w=0）は calc_velocity で速度が変化しないため影響なし。
 
        方法C（内部粘性）:
            ロープ素材の内部摩擦に相当。隣接粒子間のリンク方向相対速度を減衰。
            伸び縮み方向の振動を選択的に抑制する。
        """
        # 方法B：外部粘性（全粒子の速度を減衰）
        if self.c_ext > 0.0:
            self.v -= self.c_ext * self.dt * self.v
        else :
            self.calc_drag_force()

        # 方法C：内部粘性（リンク方向の相対速度を減衰）
        if self.c_int > 0.0:
            for i in range(self.n_links):
                d = self.x[i + 1] - self.x[i]
                l = np.linalg.norm(d)
                if l < 1e-12:
                    continue
                d_hat = d / l
                # リンク方向の相対速度
                dv_along = np.dot(self.v[i + 1] - self.v[i], d_hat) * d_hat
                # 逆質量で重み付けして各粒子に分配
                denom = self.w[i] + self.w[i + 1]
                if denom < 1e-14:
                    continue
                self.v[i]     += self.c_int * self.w[i]     / denom * dv_along
                self.v[i + 1] -= self.c_int * self.w[i + 1] / denom * dv_along
 

    def clip_velocity(self, limit: float = 100.0):
        v_norm = np.linalg.norm(self.v, axis=1, keepdims=True)
        k = np.clip(limit / np.where(v_norm > 0, v_norm, 1.0), 0.0, 1.0)
        self.v = k * self.v

 
    # ------------------------------------------------------------------
    # 接触判定（汎用化を見越して分離）
    # ------------------------------------------------------------------
    def check_collision_cylinder(self, pos):
        """
        Z 軸方向に無限に伸びる円柱との接触判定。
 
        Args:
            pos: 粒子位置 (3,)
        Returns:
            penetration: 侵入量（正=侵入、負=外側）
            normal:      押し戻し方向の単位ベクトル (3,)（侵入時のみ有効）
        """
        # XY 平面での円柱中心からの距離（Z 成分は無視）
        d_xy   = pos[:2] - self.cyl_center[:2]
        dist   = np.linalg.norm(d_xy)
        penetration = self.cyl_radius - dist   # 正なら侵入
 
        if dist > 1e-12:
            normal = np.array([d_xy[0] / dist, d_xy[1] / dist, 0.0])
        else:
            normal = np.array([1.0, 0.0, 0.0])  # 中心一致時は任意方向
 
        return penetration, normal


 
    def check_collision_rectangle(self, pos):
        """
        Z 軸方向に無限に伸びる長方形柱との接触判定。
 
        rect_theta により回転した長方形に対応するため、
        粒子位置をローカル座標（回転を戻した座標系）に変換してから判定し、
        押し戻し法線をワールド座標に戻して返す。
 
        Args:
            pos: 粒子位置 (3,)
        Returns:
            penetration: 侵入量（正=侵入、負=外側）
            normal:      押し戻し方向の単位ベクトル (3,)（侵入時のみ有効）
        """
        # 中心からの相対位置をローカル座標に変換（Z は無視）
        dx = pos[0] - self.pivot[0]
        dy = pos[1] - self.pivot[1]
        cos_t = np.cos(-self.rect_theta)
        sin_t = np.sin(-self.rect_theta)
        lx =  cos_t * dx - sin_t * dy
        ly =  sin_t * dx + cos_t * dy
 
        # 各面からの距離（正=内側に余裕あり、負=外側）
        dists = [
            self.rect_hx - lx,   # +X 面
            self.rect_hx + lx,   # -X 面
            self.rect_hy - ly,   # +Y 面
            self.rect_hy + ly,   # -Y 面
        ]
 
        # 1つでも負なら外側（接触なし）
        if any(d < 0 for d in dists):
            return -1.0, np.array([1.0, 0.0, 0.0])
 
        # 最も浅い侵入面のローカル法線を選択
        normals_local = [
            np.array([ 1.0,  0.0]),  # +X 面
            np.array([-1.0,  0.0]),  # -X 面
            np.array([ 0.0,  1.0]),  # +Y 面
            np.array([ 0.0, -1.0]),  # -Y 面
        ]
        idx = int(np.argmin(dists))
 
        # ローカル法線をワールド座標に戻す
        n = normals_local[idx]
        cos_w = np.cos(self.rect_theta)
        sin_w = np.sin(self.rect_theta)
        normal = np.array([
            cos_w * n[0] - sin_w * n[1],
            sin_w * n[0] + cos_w * n[1],
            0.0
        ])
        return dists[idx], normal
  
 
    # ------------------------------------------------------------------
    # 接触拘束（直線拘束を含む）
    # ------------------------------------------------------------------
    def collision_constrains(self):
        """
        全粒子に対して接触判定を行い、侵入した粒子を表面に押し戻す。
 
        現在の障害物：Z 軸方向無限円柱（check_collision_cylinder）
        将来的に複数の障害物・形状を追加する場合は
        check_collision_* 関数を追加して呼び出す。
        """
        for i in range(self.num):
            if self.rect_hx > 0.0 :
                # 長方形柱との接触
                penetration, normal = self.check_collision_rectangle(self.p[i])
            else :
                # 円柱との接触
                penetration, normal = self.check_collision_cylinder(self.p[i])

            if penetration > 0.0:
                self.p[i] += penetration * normal
 
        # 左端の Y 座標を 0 に拘束（片端テンションモードのみ）
        if abs(self.T_ext) > 0.0:
            self.p[0, 1] = 0.0

    # ------------------------------------------------------------------
    # 伸び拘束（XPBD ガウスザイデル）
    # ------------------------------------------------------------------
    def project_constrains(self):
        """
        ガウスザイデルで伸び拘束を解く。
 
        bc='fixed'  : 両端 w=0 なので端点は自動的に動かない。
        bc='tension': 右端 w[-1]=0（固定）、左端 w[0]>0（自由）。
                      テンション T は predict ステップで速度に加算済み
                      なので、ここでは通常のガウスザイデルのみ実行する。
 
        たるみ処理: C ≤ 0 のリンクはスキップ（押しバネなし）。
        """
        p = self.p
        for i in range(self.n_links):
            d = p[i + 1] - p[i]
            l = np.linalg.norm(d)
            if l < 1e-12:
                continue
            C = l - self.l0[i]
            if C <= 0.0:
                continue
            denom = self.w[i] + self.w[i + 1]
            if denom < 1e-14:
                continue
            s = C / l
            p[i]     += self.w[i]     / denom * s * d
            p[i + 1] -= self.w[i + 1] / denom * s * d


    # ------------------------------------------------------------------
    # 右端の強制回転
    # ------------------------------------------------------------------
    def rotate_endpoint(self):
        """
        右端粒子（x[-1]）を pivot を中心に XY 平面内で回転させる。
 
        w[-1]=0（固定端）のため速度・力では動かないので位置を直接更新する。
        回転後に x[-1] を更新することで次の predict が正しい位置から出発する。
        速度の整合は pbd_step 末尾の v=(p-x)/dt で自動的に取れる。
        """
        if abs(self.omega) < 1e-12:
            return
        r = self.x[-1] - self.pivot             # 回転中心からの相対位置
        angle = self.omega * self.dt            # 今ステップの回転角
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        # XY 平面内の 2D 回転（Z 成分はそのまま）
        rx, ry = r[0], r[1]
        self.x[-1][0] = self.pivot[0] + cos_a * rx - sin_a * ry
        self.x[-1][1] = self.pivot[1] + sin_a * rx + cos_a * ry

    # ------------------------------------------------------------------
    # メインステップ
    # ------------------------------------------------------------------
    def pbd_step(self):
        self.t = (round(self.t / self.dt) + 1) * self.dt
        self.rect_theta = self.rect_theta + self.omega * self.dt
        self.calc_force()
        self.calc_velocity()
        self.damp_velocity()
        self.clip_velocity()

        # 右端を強制回転（omega != 0 のとき）
        self.rotate_endpoint()
        
        # 予測位置
        self.p = self.x + self.dt * self.v

        # 拘束解決ループ（接触・伸びを交互に繰り返す）
        # 接触拘束が伸び拘束に負けないよう同じループ内で解く。
        # 順序: 接触 → 伸び（接触を優先して先に確定させる）
        for _ in range(self.substep):
            self.collision_constrains()
            self.project_constrains()

        self.collision_constrains()

        # 速度・位置の更新
        self.v = (self.p - self.x) / self.dt
        self.x = self.p.copy()


    def step(self):
        self.pbd_step()



class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111, projection = '3d')
        super().__init__(self.fig)


class MplCanvas2D(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)


class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sim = Simulator()
        self.sim.init_param(30, 10)

        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas = MplCanvas2D(self, width=5, height=4, dpi=100)
        self.toolbar = NavigationToolbar2QT(self.canvas, self) # matplotlibのツールバーを作成
        self.sc = None

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.show()

        self.timer = QTimer()
#        self.timer.setInterval(self.sim.dt * 1000)
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.update)
        self.timer.start()


    def update(self):
        self.sim.step()
        self.update_plot2D()


    def update_plot2D(self):
        self.xdata = self.sim.x[:, 0]
        self.ydata = self.sim.x[:, 1]
        self.zdata = self.sim.x[:, 2]

        if self.sc is None:
            self.sc, = self.canvas.axes.plot(self.xdata, self.ydata, marker = 'o')
            self.canvas.axes.set_xlim(-1.5, 0.5)
            self.canvas.axes.set_ylim(-1, 1)
            self.canvas.axes.add_patch(patches.Circle((0, 0), self.sim.cyl_radius, edgecolor='green', facecolor='none', linewidth=2))

            self.rect, = self.canvas.axes.plot(self.sim.rect_corner[:, 0], self.sim.rect_corner[:, 1])
        else:
            print(self.sim.t)
            self.sc.set_data(self.xdata, self.ydata)

            rect_pts = (calc_rot_matrix(self.sim.rect_theta) @ self.sim.rect_corner.T).T
            self.rect.set_data(rect_pts[:, 0], rect_pts[:, 1])

        self.canvas.draw()
        self.canvas.flush_events()


app = QApplication(sys.argv)
w = MainWindow()
app.exec()

# 全力実行
#sim = Simulator()
#sim.init_param(30, 10)
#
#for _ in range(1000) :
#    print(sim.t)
#    sim.step()

