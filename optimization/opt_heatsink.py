import casadi as cas
import numpy as np

Pr = 0.71
kf = 0.02
k6061 = 150

class Opt:
  @staticmethod
  def optimize(width, max_depth, thickness, total_power, max_height, airspeed, max_rise, min_channel_width):
    h = cas.SX.sym('h') # fin height
    N = cas.SX.sym('N') # number of fins
    d = cas.SX.sym('d') # depth
    w = [h, d, N]

    m = (2 * 2.7e6 * 0.004 * width * max_depth) + (2 * 2.7e6 * N * h * thickness * d)
    b = (width - N*thickness) / (N-1) # channel width

    Reb_star = airspeed * b**2 / (d * 17e-6)
    Nui = ((Reb_star * Pr / 2)**-3 + (0.664*cas.sqrt(Reb_star) * Pr**(1/3) * cas.sqrt(1 + (3.65/cas.sqrt(Reb_star))))**-3)**(-1/3)
    mH = cas.sqrt(2 * Nui * (kf / k6061) * h**2 * (1 + thickness/d) / (b * thickness))
    Nub = cas.tanh(mH) / mH

    deltaT = (0.5 * total_power / (N-1)) * b / (Nub * kf * 2 * d * h)

    g = [deltaT, h/b, b]

    prob = {'f':m, 'x':cas.vertcat(*w), 'g':cas.vertcat(*g)}
    solver = cas.nlpsol('solver', 'ipopt', prob, {'verbose':False})
    sol = solver(x0=[max_height, max_depth, 30], lbx=[max_height, max_depth, 1], ubx=[max_height, max_depth, width / thickness], lbg=[0, 4, min_channel_width], ubg=[max_rise, cas.inf, cas.inf])

    deltaT_f = cas.Function('deltaT_f', w, [deltaT], ['h', 'd', 'N'], ['deltaT'])
    m_f = cas.Function('m_f', w, [m], ['h', 'd', 'N'], ['m'])
    b_f = cas.Function('b_f', w, [b], ['h', 'd', 'N'], ['b'])

    print("height (mm): ", sol['x'][0]*1000)
    print("depth (mm): ", sol['x'][1]*1000)
    print("max fins: ", width/thickness)
    print("num fins: ", sol['x'][2])
    print("delta T (K): ", deltaT_f(sol['x'][0], sol['x'][1], sol['x'][2]))
    print("m (g): ", m_f(sol['x'][0], sol['x'][1], sol['x'][2]))
    print("channel width (mm): ", b_f(sol['x'][0], sol['x'][1], sol['x'][2])*1000)

if __name__=="__main__":
  Opt.optimize(0.06, 0.01252, 0.0008, 20, 0.025, 1, 30, 0.0005)

  ## Inverter heat sinks
  # Opt.optimize(0.06, 0.01252, 0.001, 10, 0.025, 1, 40, 0.001)
