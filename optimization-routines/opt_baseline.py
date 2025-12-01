import casadi as cas
import numpy as np

## FIXED VARIABLES

resistivity_cu = 1.68e-8
permeability_vac = 4e-7 * cas.pi

class Core:
  # Convert mm input to m
  def __init__(self, Ac_, Vc_, lc_, Wh_, Wid_, Wod_, W_):
    self.Ac = Ac_ * 1e-6
    self.Vc = Vc_ * 1e-9
    self.lc = lc_ * 1e-3
    self.Wh = Wh_ * 1e-3
    self.Wid = Wid_ * 1e-3
    self.Wod = Wod_ * 1e-3
    self.Wa = 0.5 * self.Wh * (self.Wod - self.Wid)
    self.W = W_ * 1e-3
  
  # def __str__(self):
  #   return("Ac: " + str(self.Ac) + "Vc: " + str(self.Vc) + "lc: " + str(self.lc) + "Wh: " + str(self.Wh) + "Wid: " + str(self.Wid) + "Wod: " + str(self.Wod))

class Material:
  def __init__(self, k_, alpha_, beta_, mu_):
    self.k = k_
    self.alpha = alpha_
    self.beta = beta_
    self.mu = mu_

class Winding:
  def __init__(self, num_strands_, strand_awg_, d_cond_, d_ins_):
    self.num_strands = num_strands_
    self.strand_dia = 1e-3 * 0.127 * 92**((36-strand_awg_)/39) ## mm to m
    self.r_cond = 0.5 * d_cond_ * 1e-3
    self.r_ins = 0.5 * d_ins_ * 1e-3

PQ5050 = Core(332, 37630, 113, 36.1, 20, 44, 32)
ETD59 = Core(368, 51500, 139, 45, 21.65, 44.7, 21.65)
ETD54 = Core(280, 35600, 127, 39.6, 19.3, 40.1, 19.3)
ETD49 = Core(211, 24100, 114, 35.4, 16.7, 36.1, 16.7)
mtrl_3C95 = Material(0.0063, 1.357, 2.851, 3000)
wdg_48AWG_2700_Litz = Winding(2700, 48, 2.5, 2.5)
wdg_48AWG_1700_Litz = Winding(1700, 48, 1.5, 1.5)
wdg_48AWG_1050_Litz = Winding(1050, 48, 1.18, 1.18)
wdg_28AWG_13kV = Winding(1, 28, 0.32, 0.777)

class Opt:
  @staticmethod
  def optimize(core, inductor_core, material, primary, secondary, inductor_primary, P, V1, V2, fill_factor):

    ## Based on Steigerwald ??
    R0 = (V1**2 / P) * (8 / cas.pi**2)
    print(R0)

    ## Optimization variables
    N1 = cas.SX.sym('primary turns') # Primary turns
    N2 = cas.SX.sym('secondary turns')
    lg = cas.SX.sym('gap')
    fs = cas.SX.sym('series resonant frequency')
    Cp0 = cas.SX.sym('parallel capacitance')
    Cp = 1e-3 * Cp0
    f0 = cas.SX.sym('operating frequency')
    w0 = 2 * cas.pi * f0
    N_ind = cas.SX.sym('inductor turns')

    w = [N1, f0, lg, fs, Cp0, N_ind, N2]

    # Current amplitude
    I1 = 2 * P / V1
    I2 = 2 * P / V2

    # Secondary turns
    # N2 = N1*(V2/V1)

    # DC resistance
    R1dc = resistivity_cu * N1 * (0.5*core.Wid + 0.5*core.Wod) * cas.pi / (primary.num_strands * cas.pi * (primary.strand_dia/2)**2)
    R2dc = resistivity_cu * N2 * (0.5*core.Wid + 0.5*core.Wod) * cas.pi / (secondary.num_strands * cas.pi * (secondary.strand_dia/2)**2)
    Rdc = resistivity_cu * N_ind * (0.5*inductor_core.Wid + 0.5*inductor_core.Wod) * cas.pi / (inductor_primary.num_strands * cas.pi * (inductor_primary.strand_dia/2)**2)

    # Num layers
    turns_per_layer_sec = core.Wh / (2 * secondary.r_ins)
    num_layers_sec = 0.5 * N2 / turns_per_layer_sec

    N1_max = fill_factor * core.Wa / (cas.pi*(primary.r_ins**2 + (V2/V1)*secondary.r_ins**2))
    print(N1_max)

    N_ind_max = fill_factor * inductor_core.Wa / (cas.pi * primary.r_ins**2)

    # Secondary AC factor
    skin_depth = cas.sqrt(resistivity_cu / (cas.pi * f0 * permeability_vac))
    xi = (cas.pi/4)**0.5 * secondary.strand_dia / skin_depth
    first_term = (cas.sinh(xi) + cas.sin(xi)) / (cas.cosh(xi) - cas.cos(xi))
    second_term = (cas.sinh(xi) - cas.sin(xi)) / (cas.cosh(xi) + cas.cos(xi))
    Fr2 = 0.5 * xi * (first_term + ((4*num_layers_sec**2 - 1)/3)*second_term)

    # Inductance
    L = N_ind**2 * permeability_vac * inductor_core.Ac / lg

    # Series capacitance
    Cs = 1 / (4 * cas.pi**2 * fs**2 * L)

    # Parallel capacitance
    # Due to the possibility of negative determinant, cannot solve for Cp as done in param sweep
    # Wait, interior point works via barrier method, so if you apply a constraint you would not leave the feasible region?s
    # a = (L_shunt*w0**2 - 1/Cs)**2
    # b = (2/Cs - 2*L_shunt*w0**2)
    # c = ((w0*L_shunt/R0) - (1/(w0*R0*Cs)))**2 + 1 - (4/cas.pi)**2
    # Cp = (b - cas.sqrt(b**2 - 4*a*c)) / (2 * a)

    # Tank current
    I_tank = (4 / cas.pi) * (1 / R0) * V1 * (1 + (R0*w0*Cp)**2) / cas.sqrt(1 + ((R0*w0**2 * Cp**2 + (1/R0))*(w0*L - 1/(w0*Cs)) - R0*w0*Cp)**2)

    # Flux density
    B0_ind = N_ind * I_tank * permeability_vac / lg
    B0_transformer = V1 / (2 * cas.pi * f0 * N1 * core.Ac)

    # Core loss
    Pco_transformer = 2 * 1e3 * core.Vc * material.k * f0**material.alpha * B0_transformer**material.beta
    Pco_ind = 2 * 1e3 * inductor_core.Vc * material.k * f0**material.alpha * B0_ind**material.beta

    # Winding loss
    Pcu1 = I1**2 * R1dc / 2
    Pcu2 = I2**2 * R2dc * Fr2 / 2
    Pcu_ind = I_tank**2 * Rdc

    # Nominal gain
    G0 = (4/cas.pi) / cas.sqrt((L*Cp*w0**2 - (Cp/Cs) - 1)**2 + ((w0*L/R0) - 1/(R0*w0*Cs))**2)

    # Impedance imaginary component
    AngleZ = ((R0*w0*Cp)**2 + 1)*(w0*L - 1/(w0*Cs)) - R0**2 * w0*Cp

    # Total loss
    Ptot = Pcu1 + Pcu2 + Pco_transformer + Pco_ind + Pcu_ind

    # Functions [N1, f0, lg, fs, Cp0, N_ind]
    var_strings = ['N1', 'f0', 'lg', 'fs', 'Cp0', 'N_ind', 'N2']

    Ptot_f = cas.Function('Ptot_f', w, [Ptot], var_strings, ['Ptot'])
    Pco_transformer_f = cas.Function('Pco_f', w, [Pco_transformer], var_strings, ['Pco'])
    Pco_ind_f = cas.Function('Pco_ind_f', w, [Pco_ind], var_strings, ['Pco_ind'])
    Pcu_ind_f = cas.Function('Pcu_ind_f', w, [Pcu_ind], var_strings, ['Pcu_ind'])
    Pcu1_f = cas.Function('Pcu1_f', w, [Pcu1], var_strings, ['Pcu1'])
    Pcu2_f = cas.Function('Pcu2_f', w, [Pcu2], var_strings, ['Pcu2'])
    Fr2_f = cas.Function('Fr2_f', w, [Fr2] ,var_strings, ['Fr2'])
    R1dc_f = cas.Function('R1dc_f', w, [R1dc] ,var_strings, ['R1dc'])
    R2dc_f = cas.Function('R2dc_f', w, [R2dc] ,var_strings, ['R2dc'])
    B0_ind_f = cas.Function('B0_ind_f', w, [B0_ind], var_strings, ['Nominal flux density inductor'])
    B0_transformer_f = cas.Function('B0_f', w, [B0_transformer], var_strings, ['Nominal flux density transformer'])
    L_ind_f = cas.Function('L_ind', w, [L], var_strings, ['Inductance'])
    layers_f = cas.Function('layers', w, [num_layers_sec], var_strings, ['layers'])
    I_tank_f = cas.Function('I_tank', w, [I_tank], var_strings, ['I_tank'])
    Cs_f = cas.Function('Cs', w, [Cs], var_strings, ['Cs'])
    G0_f = cas.Function('Cs', w, [G0], var_strings, ['G0'])
    AngleZ_f = cas.Function('AngleZ', w, [AngleZ], var_strings, ['AngleZ'])

    g = [L, G0, 1e3 * (Cs - 100 * Cp), AngleZ, 1e3 * Cs, (V2/V1) - (N2/N1)] # (V2/V1) - G0*(N2/N1),]

    prob = {'f':Ptot, 'x':cas.vertcat(*w), 'g':cas.vertcat(*g)}
    solver = cas.nlpsol('solver', 'ipopt', prob, {'verbose':False})
    # sol = solver(x0=[9, 200e3, 2e-3, 100e3, 1e-5, 8], lbx=[1, 50e3, 1e-4, 10e3, 1e-5, 1], ubx=[N1_max, 200e3, 4e-3, 180e3, 1e-3, N_ind_max], lbg=[1e-6, 1, -cas.inf, 50, 0], ubg=[cas.inf, 1, 0, cas.inf, 120e-6])
    sol = solver(x0=[10, 180e3, 3e-3, 100e3, 1e-5, 14, 150], lbx=[1, 180e3, 1e-4, 10e3, 22e-6, 1, 1], ubx=[N1_max, 180e3, 4e-3, 180e3, 1e-3, N_ind_max, cas.inf], lbg=[1e-6, 1, -cas.inf, 5, 1e-6, 0], ubg=[cas.inf, 2, 0, cas.inf, 400e-6, 0])

    print("Primary turns: ", sol['x'][0])
    print("Secondary turns: ", sol['x'][6])
    print("Inductor turns: ", sol['x'][5])
    print("Secondary layers: ", layers_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5], sol['x'][6]))
    print("Ind gap: ", sol['x'][2])
    print("Series resonance: ", sol['x'][3])
    print("Operating frequency", sol['x'][1])
    print("AC factor", Fr2_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5], sol['x'][6]))
    tf_pcu1 = Pcu1_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5], sol['x'][6])
    print("Primary loss", tf_pcu1)
    tf_pcu2 = Pcu2_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5], sol['x'][6])
    print("Secondary loss", tf_pcu2)
    print("Primary resistance", R1dc_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5], sol['x'][6]))
    print("Secondary resistance", R2dc_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5], sol['x'][6]))
    print("Ind flux density: ", B0_ind_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5], sol['x'][6]))
    print("Transformer flux density: ", B0_transformer_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5], sol['x'][6]))
    ind_pco = Pco_ind_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5], sol['x'][6])
    print("Inductor core loss: ", ind_pco)
    ind_pcu = Pcu_ind_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5], sol['x'][6])
    print("Inductor winding loss: ", ind_pcu)
    tf_pco = Pco_transformer_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5], sol['x'][6])
    print("Transformer core loss: ", tf_pco)
    print("Total loss: ", Ptot_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5], sol['x'][6]))
    print("Inductance: ", L_ind_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5], sol['x'][6]))
    print("N1 max area: ", N1_max)
    print("I tank: ", I_tank_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5], sol['x'][6]))
    print("Cs: ", Cs_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5], sol['x'][6]))
    print("Cp: ", 1e-3 * sol['x'][4])
    print("G0: ", G0_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5], sol['x'][6]))
    print("Angle Z: ", AngleZ_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5], sol['x'][6]))
    print("transformer total loss: ", tf_pco + tf_pcu1 + tf_pcu2)
    print("Inductor total loss: ", ind_pcu + ind_pco)

if __name__=="__main__":
  # print(resistivity_cu * 50 * (0.5*ETD59.Wid + 0.5*ETD59.Wod) * cas.pi / (wdg_28AWG_13kV.num_strands * cas.pi * (wdg_28AWG_13kV.strand_dia/2)**2))
  Opt.optimize(ETD59, ETD59, mtrl_3C95, wdg_48AWG_2700_Litz, wdg_28AWG_13kV, wdg_48AWG_2700_Litz, 1600, 250, 5000, 0.75)
