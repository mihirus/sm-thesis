import casadi as cas
import numpy as np

## FIXED VARIABLES

resistivity_cu = 1.68e-8
permeability_vac = 4e-7 * cas.pi

class Core:
  # Convert mm input to m
  def __init__(self, Ac_, Vc_, lc_, Wh_, Wid_, Wod_, W_, H_, m_):
    self.Ac = Ac_ * 1e-6
    self.Vc = Vc_ * 1e-9
    self.lc = lc_ * 1e-3
    self.Wh = Wh_ * 1e-3
    self.Wid = Wid_ * 1e-3
    self.Wod = Wod_ * 1e-3
    self.Wa = 0.5 * self.Wh * (self.Wod - self.Wid)
    self.W = W_ * 1e-3
    self.H = H_ * 1e-3
    self.m = m_
  
  # def __str__(self):
  #   return("Ac: " + str(self.Ac) + "Vc: " + str(self.Vc) + "lc: " + str(self.lc) + "Wh: " + str(self.Wh) + "Wid: " + str(self.Wid) + "Wod: " + str(self.Wod))

class Material:
  def __init__(self, k_, alpha_, beta_, mu_):
    self.k = k_
    self.alpha = alpha_
    self.beta = beta_
    self.mu = mu_

class Winding:
  def __init__(self, num_strands_, strand_awg_, d_cond_, d_ins_, rho_):
    self.num_strands = num_strands_
    self.strand_dia = 1e-3 * 0.127 * 92**((36-strand_awg_)/39) ## mm to m
    self.r_cond = 0.5 * d_cond_ * 1e-3
    self.r_ins = 0.5 * d_ins_ * 1e-3
    self.rho = rho_ # Linear density (g/m)

PQ5050 = Core(332, 37630, 113, 36.1, 20, 44, 32, 50, 190)
ETD59 = Core(368, 51500, 139, 45, 21.65, 44.7, 21.65, 62, 260)
ETD54 = Core(280, 35600, 127, 39.6, 19.3, 40.1, 19.3, 55.6, 180)
ETD49 = Core(211, 24100, 114, 35.4, 16.7, 36.1, 16.7, 49.8, 124)
mtrl_3C95 = Material(0.0063, 1.357, 2.851, 3000)
wdg_48AWG_2700_Litz = Winding(2700, 48, 2.5, 2.5, 20) # 20g per m
wdg_48AWG_1700_Litz = Winding(1700, 48, 1.5, 1.5, 12.6) # 12.6g per m
wdg_48AWG_1050_Litz = Winding(1050, 48, 1.18, 1.18, 7.8) # 7.8g per m
wdg_28AWG_13kV = Winding(1, 28, 0.32, 0.777, 1.4) # 1.4g per m

class Opt:
  @staticmethod
  def optimize(core, material, primary, secondary, P, V1, V2, fill_factor, N1_constraint, ts_constraint, separator_thickness):

    # f0 = 380000 # Set operating frequency to 200kHz (to save the diodes)
    R0 = (V2**2 / P) * (8 / cas.pi**2)
    N = V2/V1
    print(R0)

    ## Optimization variables
    N1 = cas.SX.sym('primary turns') # Primary turns
    lg = cas.SX.sym('shunt gap')
    ts = cas.SX.sym('shunt thickness')
    fs = cas.SX.sym('series resonant frequency')
    Cp0 = cas.SX.sym('parallel capacitance')
    Cp = 1e-3 * Cp0
    f0 = cas.SX.sym('operating frequency')
    w0 = 2 * cas.pi * f0

    w = [N1, lg, ts, fs, Cp0, f0]

    N2 = N*N1

    # Current
    I1 = P / V1
    I2 = P / V2

    ri = (core.Wid / 2) + 0.001 # 1 mm clearance for bobbin
    ro = (core.Wod / 2) - 0.001 # 1mm clearance from edge
    print('ro-ri: ', ro-ri)
    print('secondary r, primary r: ', secondary.r_ins, primary.r_ins)

    # DC resistance
    R1dc = resistivity_cu * N1 * (0.5*core.Wid + 0.5*core.Wod) * cas.pi / (primary.num_strands * cas.pi * (primary.strand_dia/2)**2)
    R2dc = resistivity_cu * N2 * (0.5*core.Wid + 0.5*core.Wod) * cas.pi / (secondary.num_strands * cas.pi * (secondary.strand_dia/2)**2)

    # Num layers
    # num_layers_prim = N1 * 2 * primary.r_ins / (0.5 * (core.Wod - core.Wid))
    turns_per_layer_sec = cas.floor((ro - ri) / (2 * secondary.r_ins))
    num_layers_sec = cas.ceil(N2 / (2 * turns_per_layer_sec))

    N1_max_raw = 0.25 * cas.pi * fill_factor * (core.Wa) / (cas.pi*(primary.r_ins**2 + (V2/V1)*secondary.r_ins**2))
    # total_area_consumed = (cas.pi*(N1*primary.r_ins**2 + N2*secondary.r_ins**2)) / (0.25 * cas.pi * fill_factor) + 2*ts*(ro-ri)
    total_area_consumed = ((ro-ri)*separator_thickness)*((num_layers_sec + 1)*2 + 2) + 1.1*4*(N1*primary.r_ins**2 + N2*secondary.r_ins**2) + (2*ts*(ro-ri))

    total_height_consumed = (2*num_layers_sec * 2*secondary.r_ins) + ((2*(num_layers_sec+1) + 2) * separator_thickness) + (2 * ts) + 0.005
    # N1_max = 0.5 * fill_factor * (core.Wa - 2*num_portions_prim*0.5*(core.Wod-core.Wid)*ts) / (cas.pi*(primary.r_ins**2 + N*secondary.r_ins**2))

    # Secondary AC factor
    skin_depth = cas.sqrt(resistivity_cu / (cas.pi * f0 * permeability_vac))
    xi = (cas.pi/4)**0.5 * secondary.strand_dia / skin_depth
    first_term = (cas.sinh(xi) + cas.sin(xi)) / (cas.cosh(xi) - cas.cos(xi))
    second_term = (cas.sinh(xi) - cas.sin(xi)) / (cas.cosh(xi) + cas.cos(xi))
    Fr2 = 0.5 * xi * (first_term + ((4*num_layers_sec**2 - 1)/3)*second_term)

    # Inductance
    L_shunt = N1**2 * permeability_vac * core.W * ts / (2 * lg)

    # Series capacitance
    Cs = 1 / (4 * cas.pi**2 * fs**2 * L_shunt)

    # Parallel capacitance
    # Due to the possibility of negative determinant, cannot solve for Cp as done in param sweep
    # Wait, interior point works via barrier method, so if you apply a constraint you would not leave the feasible region?s
    # a = (L_shunt*w0**2 - 1/Cs)**2
    # b = (2/Cs - 2*L_shunt*w0**2)
    # c = ((w0*L_shunt/R0) - (1/(w0*R0*Cs)))**2 + 1 - (4/cas.pi)**2
    # Cp = (b - cas.sqrt(b**2 - 4*a*c)) / (2 * a)

    # Tank current
    I_tank = (4 / cas.pi) * (1 / R0) * V1 * (1 + (R0*w0*Cp)**2) / cas.sqrt(1 + ((R0*w0**2 * Cp**2 + (1/R0))*(w0*L_shunt - 1/(w0*Cs)) - R0*w0*Cp)**2)

    # Flux density
    B0_shunt = 0.5 * N1 * I_tank * permeability_vac / lg
    B0_transformer = V1 / (2 * cas.pi * f0 * N1 * core.Ac)

    # Core loss density / core loss
    Pco_transformer = 2 * 1e3 * core.Vc * material.k * f0**material.alpha * B0_transformer**material.beta
    # V_shunt = 4 * 0.5 * (core.Wod - core.Wid) * core.W * ts
    V_shunt = 4 * ts * (ri*cas.sqrt(ro**2 - ri**2) + cas.atan(ri / cas.sqrt(ro**2 - ri**2))*ro**2 - (cas.pi * 0.5 * ri**2))
    Pco_shunt = 2 * 1e3 * V_shunt * material.k * f0**material.alpha * B0_shunt**material.beta

    # Winding loss
    Pcu1 = I_tank**2 * R1dc
    Pcu2 = I2**2 * R2dc * Fr2

    # Nominal gain
    G0 = (4/cas.pi) / cas.sqrt((L_shunt*Cp*w0**2 - (Cp/Cs) - 1)**2 + ((w0*L_shunt/R0) - 1/(R0*w0*Cs))**2)

    # Impedance imaginary component
    AngleZ = ((R0*w0*Cp)**2 + 1)*(w0*L_shunt - 1/(w0*Cs)) - R0**2 * w0*Cp

    # Total loss
    Ptot = Pcu1 + Pcu2 + Pco_transformer + Pco_shunt

    # Potting mass
    mpotting = 3.1e6 * (core.Wh*cas.pi*(ro**2 - ri**2) - V_shunt - (cas.pi*(ro+ri)*N1*cas.pi*primary.r_ins**2) - (cas.pi*(ro+ri)*N2*cas.pi*secondary.r_ins**2))

    # Parasitic capacitance

    # Post rise
    dT_post = (Pco_transformer / core.Vc) * (core.H / 2)**2 * 0.5 / 4.0 # Ferrite conductivity is 4 W / m K

    # Winding thermal resistance
    

    # Heat sink size

    # Mass
    mwinding = 0.5*(core.Wod+core.Wid)*cas.pi * (N1*primary.rho + N2*secondary.rho)
    mshunt = V_shunt * 4.7e6
    mtotal = mshunt + mwinding + mpotting + core.m

    var_strings = ['N1', 'lg', 'ts', 'fs', 'Cp', 'f0']

    # Functions
    Ptot_f = cas.Function('Ptot_f', w, [Ptot], var_strings, ['Ptot'])
    Pco_transformer_f = cas.Function('Pco_f', w, [Pco_transformer], var_strings, ['Pco'])
    Pco_shunt_f = cas.Function('Pco_f', w, [Pco_shunt], var_strings, ['Pco'])
    Pcu1_f = cas.Function('Pcu1_f', w, [Pcu1], var_strings, ['Pcu1'])
    Pcu2_f = cas.Function('Pcu2_f', w, [Pcu2], var_strings, ['Pcu2'])
    Fr2_f = cas.Function('Fr2_f', w, [Fr2] ,var_strings, ['Fr2'])
    R1dc_f = cas.Function('R1dc_f', w, [R1dc] ,var_strings, ['R1dc'])
    R2dc_f = cas.Function('R2dc_f', w, [R2dc] ,var_strings, ['R2dc'])
    B0_shunt_f = cas.Function('B0_f', w, [B0_shunt], var_strings, ['Nominal flux density shunt'])
    B0_transformer_f = cas.Function('B0_f', w, [B0_transformer], var_strings, ['Nominal flux density transformer'])
    L_shunt_f = cas.Function('L_shunt', w, [L_shunt], var_strings, ['Inductance'])
    layers_f = cas.Function('layers', w, [num_layers_sec], var_strings, ['Secondary layers'])
    # layers_prim_f = cas.Function('layers_prim', w, [num_layers_prim], var_strings, ['Primary layers'])
    V_shunt_f = cas.Function('V_shunt', w, [V_shunt], var_strings, ['V shunt'])
    I_tank_f = cas.Function('I_tank', w, [I_tank], var_strings, ['I_tank'])
    Cs_f = cas.Function('Cs', w, [Cs], var_strings, ['Cs'])
    G0_f = cas.Function('Cs', w, [G0], var_strings, ['G0'])
    AngleZ_f = cas.Function('AngleZ', w, [AngleZ], var_strings, ['AngleZ'])
    mwdg_f = cas.Function('mwdg', w, [mwinding], var_strings, ['mwinding'])
    mshunt_f = cas.Function('mshunt', w, [mshunt], var_strings, ['mshunt'])
    dT_post_f = cas.Function('dT_post', w, [dT_post], var_strings, ['dT_post'])
    total_area_consumed_f = cas.Function('total_area', w, [total_area_consumed], var_strings, ['total area consumed'])
    total_height_consumed_f = cas.Function('total_height', w, [total_height_consumed], var_strings, ['total height consumed'])
    mpotting_f = cas.Function('mpotting', w, [mpotting], var_strings, ['potting mass'])

    # g = [L_shunt, G0, 1e3 * (Cs - 20 * Cp), AngleZ, 1e3 * Cs, total_area_consumed - core.Wa]
    g = [L_shunt, G0, 1e3 * (Cs - 100 * Cp), AngleZ, 1e3 * Cs, total_height_consumed - core.Wh]

    prob = {'f':(Pco_shunt), 'x':cas.vertcat(*w), 'g':cas.vertcat(*g)}
    solver = cas.nlpsol('solver', 'ipopt', prob, {'verbose':False})

    ts_constraint_to_use_hi = 0
    ts_constraint_to_use_lo = 0
    N1_constraint_to_use_hi = 0
    N1_constraint_to_use_lo = 0
    if ts_constraint > 0:
      ts_constraint_to_use_hi = ts_constraint
      ts_constraint_to_use_lo = ts_constraint
    else:
      ts_constraint_to_use_hi = 0.012
      ts_constraint_to_use_lo = 0.002
    
    if N1_constraint > 0:
      N1_constraint_to_use_hi = N1_constraint
      N1_constraint_to_use_lo = N1_constraint
    else:
      N1_constraint_to_use_hi = N1_max_raw
      N1_constraint_to_use_lo = 1
      print("N1 initial: ", 0.5*(N1_constraint_to_use_lo + N1_constraint_to_use_hi))

    sol = solver(
      x0=[0.5*(N1_constraint_to_use_lo + N1_constraint_to_use_hi), 2e-4, 0.5*(ts_constraint_to_use_lo+ts_constraint_to_use_hi), 10e3, 15e-6, 180e3],
      lbx=[N1_constraint_to_use_lo, 1e-4, ts_constraint_to_use_lo, 10e3, 10e-6, 50e3],
      ubx=[N1_constraint_to_use_hi, 5e-3, ts_constraint_to_use_hi, 180e3, 1e-3, 180e3],
      lbg=[1e-6, 1, -cas.inf, 1, 0, -cas.inf],
      ubg=[cas.inf, 1, 0, cas.inf, 400e-6, 0])

    print("Primary turns: ", sol['x'][0])
    print("Secondary turns: ", sol['x'][0]*(V2/V1))
    print("Secondary layers in each section: ", layers_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5]))
    print("Shunt gap: ", sol['x'][1])
    print("Shunt thickness: ", sol['x'][2])
    print("Series resonance: ", sol['x'][3])
    print("Operating frequency", sol['x'][5])
    print("AC factor", Fr2_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5]))
    print("Primary loss", Pcu1_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5]))
    print("Secondary loss", Pcu2_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5]))
    print("Primary resistance", R1dc_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5]))
    print("Secondary resistance", R2dc_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5]))
    print("Shunt flux density: ", B0_shunt_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5]))
    print("Transformer flux density: ", B0_transformer_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5]))
    print("Shunt loss: ", Pco_shunt_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5]))
    print("Transformer core loss: ", Pco_transformer_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5]))
    print("Total loss: ", Ptot_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5]))
    print("Inductance: ", L_shunt_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5]))
    print("N1 max: ", N1_max_raw)
    print("total area consumed: ", 1e6 * total_area_consumed_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5]))
    print("total height consumed: ", 1e3 * total_height_consumed_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5]))
    print("Height available: ", core.Wh * 1e3)
    print("Window area available: ", core.Wa * 1e6)
    print("V shunt: ", 1e9 * V_shunt_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5]), " mm3")
    print("I tank: ", I_tank_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5]))
    print("Cs: ", Cs_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5]))
    print("Cp: ", 1e-3 * sol['x'][4])
    print("G0: ", G0_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5]))
    print("Im Z: ", AngleZ_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5]))
    mass_winding = mwdg_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5])
    print("Winding mass: ", mass_winding)
    mass_shunt = mshunt_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5])
    print("Shunt mass: ", mass_shunt)
    mass_potting = mpotting_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5])
    print("Potting mass: ", mass_potting)
    print("Total mass: ", mass_winding + mass_shunt + core.m + mass_potting)
    print("dT post: ", dT_post_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5]))

    return (sol['x'][0], sol['x'][2], N1_max_raw, mass_winding + mass_shunt + core.m + mass_potting, Ptot_f(sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5]))

if __name__=="__main__":
  core_to_use = ETD59
  # r0 = Opt.optimize(ETD59, mtrl_3C95, wdg_48AWG_2700_Litz, wdg_28AWG_13kV, 1600, 250, 5000, 0.5, 7, -1, 0.001)
  r0 = Opt.optimize(core_to_use, mtrl_3C95, wdg_48AWG_2700_Litz, wdg_28AWG_13kV, 1600, 250, 5000, 0.5, 7, -1, 0.001)


  # r0 = [float(i) for i in r0]

  # N1_hi = cas.ceil(r0[0])
  # N1_lo = cas.floor(r0[0])
  # ts_hi = 0.003 + r0[1] - r0[1]%0.003
  # ts_lo = r0[1] - r0[1]%0.003

  # r1 = Opt.optimize(core_to_use, mtrl_3C95, wdg_48AWG_2700_Litz, wdg_28AWG_13kV, 1750, 250, 5500, 22, 0.5, N1_hi, ts_lo)
  # r2 = Opt.optimize(core_to_use, mtrl_3C95, wdg_48AWG_2700_Litz, wdg_28AWG_13kV, 1750, 250, 5500, 22, 0.5, N1_hi, ts_hi)
  # r3 = Opt.optimize(core_to_use, mtrl_3C95, wdg_48AWG_2700_Litz, wdg_28AWG_13kV, 1750, 250, 5500, 22, 0.5, N1_lo, ts_lo)
  # r4 = Opt.optimize(core_to_use, mtrl_3C95, wdg_48AWG_2700_Litz, wdg_28AWG_13kV, 1750, 250, 5500, 22, 0.5, N1_lo, ts_hi)

  # r1 = [float(i) for i in r1]
  # r2 = [float(i) for i in r2]
  # r3 = [float(i) for i in r3]
  # r4 = [float(i) for i in r4]
  # print(ts_hi)
  # print('\t[N1, ts, N1 max, total mass, total loss]')
  # print('\t', r0, '\n\t', r1, '\n\t', r2, '\n\t', r3, '\n\t', r4)

  # Manual
  # r_final = Opt.optimize(core_to_use, mtrl_3C95, wdg_48AWG_2700_Litz, wdg_28AWG_13kV, 1750, 250, 5500, 22, 0.5, N1_hi, ts_lo)



