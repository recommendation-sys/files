import math
import random
import sys
import time
from mip import Model, xsum, CutPool
from mip.constants import *
import traceback


def read_instance(filename):
    path = "instances//{}".format(filename)
    m = Model(solver_name=CBC)
    m.read(path)
    m.verbose = 0

    # -----------------------------------------------------
    # Select constraints
    # -----------------------------------------------------
    constraints = []
    for con in m.constrs:
        if con.expr.sense == '<' or con.expr.sense == '=':
            if check_constraint(m, con):
                constraints.append(con.idx)

    # -----------------------------------------------------

    st = m.optimize(relax=True)

    if st != OptimizationStatus.OPTIMAL:
        print("Error! {};{}".format(filename, st))
        exit()

    # print("initial_bound = {}".format(m.objective_value))

    if m.sense == 'MIN':
        return m, 1, constraints
    else:
        return m, -1, constraints


def down_diving(m, fracVars, down_locks, up_locks, origSol):
    best_var = -1
    best_dir = -1
    best_down_dist = sys.float_info.max
    for idx in fracVars:

        if down_locks[idx] > 0 and up_locks[idx] > 0:

            value = origSol[idx]
            cur_dist = value - math.floor(value)

            if m.vars[idx].var_type != 'B':
                cur_dist *= 1000

            if cur_dist < best_down_dist:
                best_var = idx
                best_down_dist = cur_dist

    return best_var, best_dir


def up_diving(m, fracVars, down_locks, up_locks, origSol):
    best_var = -1
    best_dir = 1
    best_up_dist = sys.float_info.max
    for idx in fracVars:

        if down_locks[idx] > 0 and up_locks[idx] > 0:

            value = origSol[idx]
            cur_dist = math.ceil(value) - value

            if m.vars[idx].var_type != 'B':
                cur_dist *= 1000

            if cur_dist < best_up_dist:
                best_var = idx
                best_up_dist = cur_dist

    return best_var, best_dir


def fractional_diving(m, fracVars, down_locks, up_locks, objCoefs, direction, origSol):
    best_var = -1
    best_dir = 0
    best_frac = sys.float_info.max
    for idx in fracVars:

        if down_locks[idx] > 0 and up_locks[idx] > 0:

            value = origSol[idx]
            cur_frac = value - math.floor(value)

            if cur_frac < 0.5:
                cur_dir = -1
            elif cur_frac > 0.5:
                cur_dir = 1
                cur_frac = 1 - cur_frac
            else:
                objCoef = direction * objCoefs[idx]
                if objCoef >= 0:
                    cur_dir = -1
                else:
                    cur_dir = 1
                    cur_frac = 1.0 - cur_frac

            if m.vars[idx].var_type != 'B':
                cur_frac *= 1000

            if cur_frac < best_frac:
                best_var = idx
                best_dir = cur_dir
                best_frac = cur_frac

    return best_var, best_dir


def coefficient_diving(m, fracVars, down_locks, up_locks, objCoefs, direction, origSol):
    best_var = -1
    best_dir = 0
    best_frac = -1
    best_lock = sys.float_info.max

    for idx in fracVars:
        if down_locks[idx] > 0 and up_locks[idx] > 0:

            value = origSol[idx]

            cur_lock = min(down_locks[idx], up_locks[idx])
            cur_frac = value - math.floor(value)

            if down_locks[idx] < up_locks[idx]:
                cur_dir = -1
            elif down_locks[idx] > up_locks[idx]:
                cur_dir = 1
                cur_frac = 1 - cur_frac
            else:
                if cur_frac < 0.5:
                    cur_dir = -1
                elif cur_frac > 0.5:
                    cur_dir = 1
                    cur_frac = 1 - cur_frac
                else:
                    objCoef = direction * objCoefs[idx]
                    if objCoef >= 0:
                        cur_dir = -1
                    else:
                        cur_dir = 1
                        cur_frac = 1 - cur_frac

            if m.vars[idx].var_type != 'B':
                cur_frac *= 1000

            if cur_lock < best_lock or (cur_lock == best_lock and cur_frac < best_frac):
                best_var = idx
                best_lock = cur_lock
                best_dir = cur_dir
                best_frac = cur_frac

    return best_var, best_dir


def vector_l_diving(m, fracVars, down_locks, up_locks, columnLength, objCoefs, direction, origSol):
    best_var = -1
    best_dir = 0
    best_frac = sys.float_info.max
    best_score = sys.float_info.max

    for idx in fracVars:
        if down_locks[idx] > 0 and up_locks[idx] > 0:

            value = origSol[idx]
            fraction = value - math.floor(value)
            objCoef = direction * objCoefs[idx]

            if objCoef >= 0:
                Round = 1
                objDelta = (math.ceil(value) - value) * objCoef
                fraction = 1 - fraction
            else:
                Round = -1
                objDelta = (math.floor(value) - value) * objCoef

            score = objDelta / (columnLength[idx] + 1.0)

            if m.vars[idx].var_type != 'B':
                score *= 1000

            if score < best_score or (best_score == score and fraction < best_frac):
                best_var = idx
                best_dir = Round
                best_frac = fraction
                best_score = score

    return best_var, best_dir


def line_search_diving(m, fracVars, down_locks, up_locks, rootNodeLPSol, objCoefs, direction, xStar):
    bestColumn = -1
    bestRound = 0
    bestDistance = sys.float_info.max
    bestFraction = sys.float_info.max

    for idx in fracVars:
        if down_locks[idx] > 0 and up_locks[idx] > 0:

            value = xStar[idx]
            objCoef = direction * objCoefs[idx]
            rootValue = rootNodeLPSol[idx]
            fraction = value - math.floor(value)

            if value + 0.0001 <= rootValue:
                round = -1
                distance = fraction / (rootValue - value + 0.0001)
            elif value >= rootValue + 0.0001:
                round = 1
                fraction = 1 - fraction
                distance = fraction / (value - rootValue + 0.0001)
            else:
                distance = sys.float_info.max / 5000
                if fraction < 0.5:
                    round = -1
                elif fraction > 0.5:
                    round = 1
                    fraction = 1 - fraction
                elif objCoef >= 0:
                    round = -1
                else:
                    round = 1
                    fraction = 1 - fraction

            if m.vars[idx].var_type != 'B':
                distance *= 1000

            if distance < bestDistance or (distance == bestDistance and fraction < bestFraction):
                bestColumn = idx
                bestDistance = distance
                bestFraction = fraction
                bestRound = round

    return bestColumn, bestRound


def conflict_diving(m, fracVars, down_locks, up_locks, objCoefs, direction, columnLength, check_degree, degree,
                    origSol):
    best_var = -1
    best_score = 0.0
    best_dir = 0

    for idx in fracVars:
        if down_locks[idx] > 0 and up_locks[idx] > 0:
            if m.vars[idx].var_type != 'B':
                continue

            if check_degree[idx] == 0:
                degree[idx][0], degree[idx][1] = calculate_degree(m, idx)
                check_degree[idx] = 1

            confsUp = degree[idx][0]
            confsDown = degree[idx][1]

            if confsDown <= 1 and confsUp <= 1:
                continue

            value = origSol[idx]
            score = 0.0
            Round = 0

            if confsDown > confsUp:
                Round = -1
                score = confsDown + (1.0 - (value - math.floor(value)))
            if confsDown < confsUp:
                Round = 1
                score = confsUp + (1.0 - (math.ceil(value) - value))

            if score >= best_score + 0.0001:
                best_var = idx
                best_dir = Round
                best_score = score

    if best_score == 0.0:
        best_var, best_dir = vector_l_diving(m, fracVars, down_locks, up_locks, columnLength, objCoefs, direction,
                                             origSol)
    return best_var, best_dir


def mod_degree_diving(m, fracVars, down_locks, up_locks, objCoefs, direction, columnLength, check_mDegree, modDegree,
                      origSol):
    best_var = -1
    best_dir = 0
    best_score = 0.0

    for idx in fracVars:
        if down_locks[idx] > 0 and up_locks[idx] > 0:
            if m.vars[idx].var_type != 'B':
                continue

            if check_mDegree[idx] == 0:
                modDegree[idx][0], modDegree[idx][1] = calculate_mod_degree(m, idx)
                check_mDegree[idx] = 1

            modDegreeUp = modDegree[idx][0]
            modDegreeDown = modDegree[idx][1]

            if modDegreeDown <= 2 and modDegreeUp <= 2:
                continue

            score = 0.0
            Round = 0
            value = origSol[idx]

            if modDegreeDown > modDegreeUp:
                Round = -1
                score = modDegreeDown + (1.0 - (value - math.floor(value)))
            if modDegreeDown < modDegreeUp:
                Round = 1
                score = modDegreeUp + (1.0 - (math.ceil(value) - value))

            if score >= best_score + 0.0001:
                best_var = idx
                best_dir = Round
                best_score = score

    if best_score == 0.0:
        best_var, best_dir = vector_l_diving(m, fracVars, down_locks, up_locks, columnLength, objCoefs, direction,
                                             origSol)

    return best_var, best_dir


def calculate_degree(m, idx):
    l1, l2 = m.conflict_graph.conflicting_assignments(m.vars[idx] == 1)
    confsUp = l1.__len__() + l2.__len__()
    l1, l2 = m.conflict_graph.conflicting_assignments(m.vars[idx] == 0)
    confsDown = l1.__len__() + l2.__len__()
    return confsUp, confsDown


def calculate_mod_degree(m, idx):
    l1, l2 = m.conflict_graph.conflicting_assignments(m.vars[idx] == 1)

    modDegreeUp = l1.__len__() + l2.__len__()

    for k in l1:
        aux1, aux2 = m.conflict_graph.conflicting_assignments(k == 1)
        modDegreeUp += aux1.__len__() + aux2.__len__()
    for k in l2:
        aux1, aux2 = m.conflict_graph.conflicting_assignments(k == 0)
        modDegreeUp += aux1.__len__() + aux2.__len__()

    l1, l2 = m.conflict_graph.conflicting_assignments(m.vars[idx] == 0)

    modDegreeDown = l1.__len__() + l2.__len__()

    for k in l1:
        aux1, aux2 = m.conflict_graph.conflicting_assignments(k == 1)
        modDegreeDown += aux1.__len__() + aux2.__len__()
    for k in l2:
        aux1, aux2 = m.conflict_graph.conflicting_assignments(k == 0)
        modDegreeDown += aux1.__len__() + aux2.__len__()
    return modDegreeUp, modDegreeDown


def set_fpump_obj(m, vars_bin_int, origObjCoefs, origSol, index_delete):
    if len(m.vars) > origObjCoefs.__len__():
        remlist = []
        for i in index_delete:
            remlist.append(m.constrs[i])
        m.remove(remlist)
        r = len(m.vars) - origObjCoefs.__len__()
        for i in range(r):
            m.remove(m.vars[len(m.vars) - 1])
        index_delete.clear()

    coefs = origObjCoefs.copy()

    count = 0
    sumConstants = 0.0
    for idx in vars_bin_int:
        xTilde = math.floor(origSol[idx] + 0.5)
        if xTilde <= m.vars[idx].lb + 0.0001:
            coefs[idx] = 1.0
            sumConstants += (-1.0 * m.vars[idx].lb)
        elif xTilde >= m.vars[idx].ub - 0.0001:
            coefs[idx] = -1.0
            sumConstants += m.vars[idx].ub
        else:
            if m.vars[idx].var_type == 'B':
                raise Exception("Error in FP heuristic.\n")
            m.add_var(name='xminus{}'.format(idx), var_type=CONTINUOUS, lb=0.0)
            m.add_var(name='xplus{}'.format(idx), var_type=CONTINUOUS, lb=0.0)
            index_delete.append(len(m.constrs))
            m += 1.0 * m.vars[idx] - 1.0 * m.vars[len(m.vars) - 1] + 1.0 * m.vars[len(m.vars) - 2] == xTilde
            count += 2

    m.add_var(name='constantFPump', var_type=CONTINUOUS, lb=sumConstants, ub=sumConstants)
    count += 1

    m.objective = xsum(coefs[i] * m.vars[i]
                       for i in range(coefs.__len__())) \
                  + xsum(1.0 * m.vars[i]
                         for i in range(coefs.__len__(), coefs.__len__() + count))


def trivially_round(fracVars, down_locks, up_locks, objCoefs, direction, origSol):
    for idx in fracVars:
        # value = m.vars[idx].x
        value = origSol[idx]
        if down_locks[idx] == 0 and up_locks[idx] == 0:
            objCoef = direction * objCoefs[idx]
            if objCoef >= 0:
                origSol[idx] = math.floor(value)
            else:
                origSol[idx] = math.ceil(value)
        elif down_locks[idx] == 0:
            origSol[idx] = math.floor(value)
        else:
            origSol[idx] = math.ceil(value)


def check_trivially(down_locks, up_locks, fracVars):
    for v in fracVars:
        if down_locks[v] > 0 and up_locks[v] > 0:
            return False
    return True


def getkey1(item):
    return item[0]


def colect_info(m, con):
    coefs = []
    index = []
    info = []

    cap = con.rhs

    for i, j in con.expr.expr.items():
        var = m.var_by_name('{}'.format(i))
        if 0.0 <= i.lb < 0.0001 and 0.0 < i.ub < 0.0001:
            continue
        if i.var_type == 'B' or (i.var_type == 'I' and i.lb > 0.0 and i.ub < 1.0001):
            if abs(i.ub - i.lb) < 0.0002:
                if j < 0:
                    cap += abs(j) * abs(i.ub)
                else:
                    cap -= j * abs(i.ub)
                continue
            coefs.append(j)
            index.append(var.idx)
            info.append([j, var.name])
        else:
            if j < 0:
                if i.ub >= 0.0001:
                    cap += abs(j) * i.ub
            else:
                if i.lb <= -0.0001:
                    cap += j * abs(i.lb)
    return coefs, index, cap, info


def CDC_pos(model, LC, n, w, cap, R, varname, names):

    # print(">>> Searching... CDC cuts (all coeffs are positive) <<<")

    v_min = 0.005
    eps = 0.0001

    cp = CutPool()

    model2 = Model(sense=MAXIMIZE, solver_name=CBC)

    varx = [[model2.add_var(
        name='v{}v{}'.format(LC[i][0], LC[i][j]),
        var_type=BINARY)
        for j in range(LC[i].__len__())]
        for i in range(LC.__len__())]

    varz = [model2.add_var(
        name='v{}'.format(i),
        var_type=BINARY)
        for i in range(n)]

    v_bar = model2.add_var(
        name='v_bar',
        var_type=CONTINUOUS)

    model2.objective = v_bar + xsum(eps * varx[i][j]
                                              for i in range(LC.__len__()) for j in range(1, LC[i].__len__())) + \
                        xsum(eps * varz[i] for i in range(n))

    # Constraint 7
    for i in range(n):
        model2 += varx[i][0] + xsum(varx[j][LC[j].index(i)] for j in R[i]) + varz[i] <= 1

    # Constraint 8
    for j in range(n):
        for i in range(j + 1, n):
            if w[i] > w[j]:
                model2 += varx[i][0] + varz[j] <= 1

    # Constraint 9
    for i in range(LC.__len__()):
        for j in range(1, LC[i].__len__()):
            model2 += varx[i][0] - varx[i][j] >= 0

    # Constraint 10
    for i in range(LC.__len__()):
        for j in range(1, LC[i].__len__() - 1):
            for p in range(j + 1, LC[i].__len__()):
                if not LC[LC[i][j]].__contains__(LC[i][p]) and \
                        not LC[LC[i][p]].__contains__(LC[i][j]):
                    model2 += varx[i][j] + varx[i][p] <= 1

    # Constraint 11
    model2 += xsum(w[LC[i][0]] * varx[i][0]
                        for i in range(LC.__len__())) - cap >= 0.005

    # Constraint 12
    model2 += v_bar == xsum(model.var_by_name("{}".format(varname[LC[i][j]])).x * varx[i][j]
                            for i in range(LC.__len__())
                            for j in range(1, LC[i].__len__())) \
              + xsum((model.var_by_name("{}".format(varname[LC[i][0]])).x - 1) * varx[i][0]
                     for i in range(LC.__len__())) + \
              xsum(model.var_by_name("{}".format(varname[i])).x * varz[i] for i in range(n)) + 1

    # Constraint 13
    model2 += v_bar >= v_min

    model2.verbose = 0

    stg = model2.optimize()

    if stg == OptimizationStatus.OPTIMAL:

        for k in range(model2.num_solutions):

            rhs = 0

            cut_aux = [0 for _ in range(n)]

            for i in range(n):
                if varz[i].xi(k) >= 0.99:
                    cut_aux[i] = 1

            for i in range(n):
                if varx[i][0].xi(k) >= 0.99:
                    cut_aux[i] = 1
                    rhs += 1
                for j in range(1, LC[i].__len__()):
                    if varx[i][j].xi(k) >= 0.99:
                        cut_aux[LC[i][j]] = 1

            if rhs == 1 and sum(cut_aux) == 1:
                continue

            cutvar = []
            for i in range(n):
                pos = varname.index(names[i])
                if cut_aux[pos] == 1:
                    var = model.var_by_name("{}".format(names[i]))
                    cutvar.append(var)

            cut = xsum(var for var in cutvar) <= rhs - 1
            cp.add(cut)

    return cp


def CDC_neg(model, LC, n, w, w2, c2, R, varname, frac):

    # print("\n>>> Searching... CDC cuts (there are negative coefficients) <<<\n")

    v_min = 0.005
    eps = 0.0001

    cp = CutPool()

    model2 = Model(sense=MAXIMIZE, solver_name=CBC)

    varx = [[model2.add_var(
        name='v{}v{}'.format(LC[i][0], LC[i][j]),
        var_type=BINARY)
        for j in range(LC[i].__len__())]
        for i in range(LC.__len__())]

    varz = [model2.add_var(
        name='v{}'.format(i),
        var_type=BINARY)
        for i in range(n)]

    v_bar = model2.add_var(
        name='v_bar',
        var_type=CONTINUOUS)

    model2.objective = v_bar + xsum(eps * varx[i][j]
                                         for i in range(LC.__len__()) for j in range(1, LC[i].__len__())) + \
                            xsum(eps * varz[i] for i in range(n))

    # Constraint 7
    for i in range(n):
        model2 += varx[i][0] + xsum(varx[j][LC[j].index(i)] for j in R[i]) + varz[i] <= 1

    # Constraint 8
    for j in range(n):
        for i in range(j + 1, n):
            if w2[i] > w2[j]:
                model2 += varx[i][0] + varz[j] <= 1

    # Constraint 9
    for i in range(LC.__len__()):
        for j in range(1, LC[i].__len__()):
            model2 += varx[i][0] - varx[i][j] >= 0

    # Constraint 10
    for i in range(LC.__len__()):
        for j in range(1, LC[i].__len__() - 1):
            for p in range(j + 1, LC[i].__len__()):
                if not LC[LC[i][j]].__contains__(LC[i][p]) and \
                        not LC[LC[i][p]].__contains__(LC[i][j]):
                    model2 += varx[i][j] + varx[i][p] <= 1

    # Constraint 11
    model2 += xsum(w2[LC[i][0]] * varx[i][0]
                        for i in range(LC.__len__())) - c2 >= 0.005

    # Constraint 12
    model2 += v_bar == xsum(frac[LC[i][j]] * varx[i][j]
                            for i in range(LC.__len__())
                            for j in range(1, LC[i].__len__())) \
              + xsum((frac[LC[i][0]] - 1) * varx[i][0]
                     for i in range(LC.__len__())) + \
              xsum(frac[i] * varz[i] for i in range(n)) + 1

    # Constraint 13
    model2 += v_bar >= v_min

    model2.verbose = 0

    stg = model2.optimize(max_seconds=60)

    if stg == OptimizationStatus.OPTIMAL:

        for k in range(model2.num_solutions):

            rhs = 0

            cut_aux = [0 for _ in range(n)]

            for i in range(n):
                if varz[i].xi(k) >= 0.99:
                    cut_aux[i] = 1

            for i in range(n):
                if varx[i][0].xi(k) >= 0.99:
                    cut_aux[i] = 1
                    rhs += 1
                for j in range(1, LC[i].__len__()):
                    if varx[i][j].xi(k) >= 0.99:
                        cut_aux[LC[i][j]] = 1

            cutvarP = []
            cutvarN = []

            for i in range(n):
                if cut_aux[i] == 1:
                    var = model.var_by_name("{}".format(varname[i]))
                    if w[i] < 0:
                        cutvarN.append(var)
                    else:
                        cutvarP.append(var)

            cut = xsum(var for var in cutvarP) + xsum(1 - var for var in cutvarN) <= rhs - 1
            cp.add(cut)
    return cp


def constroiLC_R_neg(m, lis):

    w2 = [i[0] for i in lis]
    w = [i[1] for i in lis]
    varname = [i[2] for i in lis]
    frac = [i[3] for i in lis]
    n = varname.__len__()

    LC = [[] for _ in range(n)]  # set C in the paper
    for i in range(n):
        LC[i].append(i)

    R = [[] for _ in range(n)]  # set R in the paper

    for i in range(n):
        for j in range(i + 1, n):
            if w[i] > 0:
                if w[j] > 0:
                    if m.conflict_graph.conflicting(m.var_by_name(varname[i]) == 1, m.var_by_name(varname[j]) == 1):
                        LC[i].append(j)
                        R[j].append(i)
                else:
                    if m.conflict_graph.conflicting(m.var_by_name(varname[i]) == 1, m.var_by_name(varname[j]) == 0):
                        LC[i].append(j)
                        R[j].append(i)
            else:
                if w[j] > 0:
                    if m.conflict_graph.conflicting(m.var_by_name(varname[i]) == 0, m.var_by_name(varname[j]) == 1):
                        LC[i].append(j)
                        R[j].append(i)
                else:
                    if m.conflict_graph.conflicting(m.var_by_name(varname[i]) == 0, m.var_by_name(varname[j]) == 0):
                        LC[i].append(j)
                        R[j].append(i)

    return LC, R, w, w2, varname, frac


def constroiLC_R_pos(m, lis):

    w = [i[0] for i in lis]
    n = w.__len__()

    varname = [i[1] for i in lis]

    LC = [[] for _ in range(n)]  # set C in the paper
    for i in range(n):
        LC[i].append(i)

    R = [[] for _ in range(n)]  # set R in the paper

    for i in range(n):
        for j in range(i + 1, n):
            if m.conflict_graph.conflicting(m.var_by_name(varname[i]) == 1, m.var_by_name(varname[j]) == 1):
                LC[i].append(j)
                R[j].append(i)

    return LC, varname, w, R


def cdc_separation(m, constraints):

    cp_cdc = CutPool()

    random.shuffle(constraints)

    for r in constraints:
        con = m.constrs[r]
        vals, idxs, c, pares = colect_info(m, con)

        if vals.__len__() == 0:
            continue

        if min(vals) < 0:
            pares_novo = []
            for i in range(pares.__len__()):
                if pares[i][0] < 0:
                    pares_novo.append([abs(pares[i][0]), vals[i], pares[i][1], 1.0 - m.vars[idxs[i]].x])
                    c += abs(vals[i])
                else:
                    pares_novo.append([pares[i][0], vals[i], pares[i][1], m.vars[idxs[i]].x])

            lis = sorted(pares_novo, key=getkey1, reverse=False)

            LC, R, w, w2, varname, frac = constroiLC_R_neg(m, lis)

            cpX = CDC_neg(m, LC, varname.__len__(), w, w2, c, R, varname, frac)
            if cpX.cuts.__len__() > 0:
                for cut in cpX.cuts:
                    cp_cdc.add(cut)
                break
        else:
            lis = sorted(pares, key=getkey1, reverse=False)
            LC, varname, w, R = constroiLC_R_pos(m, lis)
            names = [m.vars[idxs[i]].name for i in range(idxs.__len__())]
            cpX = CDC_pos(m, LC, varname.__len__(), w, c, R, varname, names)
            if cpX.cuts.__len__() > 0:
                for cut in cpX.cuts:
                    cp_cdc.add(cut)
                break
    return cp_cdc


def list_frac_vars(m, vars_bin_int):
    fracVars = []
    for v in vars_bin_int:
        value_x = m.vars[v].x
        value_xt = str(value_x)
        if value_x >= 0.0001 and not value_xt.__contains__('.9999') and not value_xt.__contains__('.0000') \
                and abs(value_x - math.floor(value_x)) >= 0.0001:
            fracVars.append(v)

    return fracVars


def calculate_locks(m):
    down_locks = []
    up_locks = []

    root = []
    vars_bin_int = []
    columnLength = []

    dic = m.objective.expr

    origObjCoefs = []

    for v in m.vars:
        if v.var_type == 'B' or v.var_type == "I":
            vars_bin_int.append(v.idx)
        origObjCoefs.append(dic.get(v, 0))
        columnLength.append(0)
        down_locks.append(0)
        up_locks.append(0)
        root.append(v.x)

    for constraint in range(len(m.constrs)):
        sense = m.constrs[constraint].expr.sense
        dic = m.constrs[constraint].expr.expr
        for var, coef in dic.items():
            if var.var_type != 'B' and var.var_type != "I":
                continue
            columnLength[var.idx] += 1
            if sense == '<':
                if coef > 0:
                    up_locks[var.idx] += 1
                else:
                    down_locks[var.idx] += 1
            elif sense == '=':
                down_locks[var.idx] += 1
                up_locks[var.idx] += 1
            else:
                if coef > 0:
                    down_locks[var.idx] += 1
                else:
                    up_locks[var.idx] += 1

    return down_locks, up_locks, vars_bin_int, columnLength, root, origObjCoefs


def execute(strategy, useFPumpObj, m, direction, down_locks, up_locks, vars_bin_int, columnLength, root, origObjCoefs,
            degree, check_degree, constraints):
    Pass = 0
    TIME_LIMIT = 10800
    ini = time.time()
    index_delete = []

    while time.time() - ini <= TIME_LIMIT:

        print("<> Pass {} [obj={}]".format(Pass, m.objective_value))

        fracVars = list_frac_vars(m, vars_bin_int)

        if fracVars.__len__() > 0:

            origSol = [v.x for v in m.vars]

            if not check_trivially(down_locks, up_locks, fracVars):

                if useFPumpObj == 1:
                    set_fpump_obj(m, vars_bin_int, origObjCoefs, origSol, index_delete)

                if strategy == 0:
                    var, Round = fractional_diving(m, fracVars, down_locks, up_locks, origObjCoefs, direction, origSol)
                elif strategy == 1:
                    var, Round = coefficient_diving(m, fracVars, down_locks, up_locks, origObjCoefs, direction, origSol)
                elif strategy == 2:
                    var, Round = vector_l_diving(m, fracVars, down_locks, up_locks, columnLength, origObjCoefs,
                                                 direction, origSol)
                elif strategy == 3:
                    var, Round = line_search_diving(m, fracVars, down_locks, up_locks, root, origObjCoefs, direction,
                                                    origSol)
                elif strategy == 4:
                    var, Round = conflict_diving(m, fracVars, down_locks, up_locks, origObjCoefs, direction,
                                                 columnLength, check_degree, degree, origSol)
                elif strategy == 5:
                    var, Round = mod_degree_diving(m, fracVars, down_locks, up_locks, origObjCoefs, direction,
                                                   columnLength, check_degree, degree, origSol)
                elif strategy == 6:
                    var, Round = down_diving(m, fracVars, down_locks, up_locks, origSol)
                else:
                    var, Round = up_diving(m, fracVars, down_locks, up_locks, origSol)

                if Round == -1:
                    m.vars[var].ub = math.floor(origSol[var])
                else:
                    m.vars[var].lb = math.ceil(origSol[var])

                st = m.optimize(relax=True)

                if st != OptimizationStatus.OPTIMAL:
                    if time.time() - ini >= TIME_LIMIT:
                        return 0, time.time() - ini, Pass, False, 0
                    else:
                        return 1, time.time() - ini, Pass, False, 0

                # generate Cuts
                cp1 = m.generate_cuts([CutType.ODD_WHEEL])
                cp2 = m.generate_cuts([CutType.CLIQUE])
                cp3 = m.generate_cuts([CutType.KNAPSACK_COVER])
                cp4 = m.generate_cuts([CutType.MIR])
                cp5 = m.generate_cuts([CutType.GOMORY])

                cp = CutPool()

                for cut in cp1.cuts:
                    cp.add(cut)
                for cut in cp2.cuts:
                    cp.add(cut)
                for cut in cp3.cuts:
                    cp.add(cut)
                for cut in cp4.cuts:
                    cp.add(cut)
                for cut in cp5.cuts:
                    cp.add(cut)

                # generate CDC cut
                cp_cdc = cdc_separation(m, constraints)
                for cut in cp_cdc.cuts:
                    cp.add(cut)

                if cp.cuts.__len__() > 0:
                    for cut in cp.cuts:
                        m += cut
                    st = m.optimize(relax=True)
                    if st != OptimizationStatus.OPTIMAL:
                        if time.time() - ini >= TIME_LIMIT:
                            return 0, time.time() - ini, Pass, False, 0
                        else:
                            return 1, time.time() - ini, Pass, False, 0
            else:
                trivially_round(fracVars, down_locks, up_locks, origObjCoefs, direction, origSol)
                Fo = sum(origObjCoefs[i] * origSol[i] for i in range(origObjCoefs.__len__()))
                return -1, time.time() - ini, Pass, True, Fo
        else:
            return -1, time.time() - ini, Pass, False, 0
        Pass += 1
    return 0, time.time() - ini, Pass, False, 0


def objvalue(m, origObjCoefs):
    soma = 0.0
    for var in range(origObjCoefs.__len__()):
        soma += m.vars[var].x * origObjCoefs[var]
    return soma


def check_constraint(m, con):
    cap = con.rhs
    sum_w = 0
    count = 0
    count_1 = 0
    maxitems = 30
    for i, j in con.expr.expr.items():
        # print(i, j, i.var_type, i.ub)
        if 0.0 <= i.lb < 0.0001 and 0.0 <= i.ub < 0.0001:
            continue
        if abs(i.ub - i.lb) < 0.0001:
            continue
        if j < 0:
            return False
        if i.var_type == 'B' or (i.var_type == 'I' and i.lb >= 0.0 and i.ub < 1.0001):
            sum_w += j
            count += 1
            if j == 1 or j == -1:
                count_1 += 1
            if count > maxitems:
                return False
        else:
            if j > 0:
                if i.lb < 0:
                    cap += j * abs(i.lb)
    if count_1 == count:
        return False
    elif 3 <= count <= maxitems and sum_w > cap:
        return True
    else:
        return False


if __name__ == '__main__':

    random.seed(42)

    heuristics = ["fractional", "coefficient", "vectorLenght", "lineSearch", "conflicts", "modifiedDegree", "down",
                  "up"]

    # -----------------------------------------------------
    # Inputs
    # -----------------------------------------------------

    filename = sys.argv[1]
    heuristic = int(sys.argv[2])
    useFPumpObj = int(sys.argv[3])

    print("\n", filename)

    # -----------------------------------------------------
    # Create MIP Model
    # -----------------------------------------------------
    m, direction, constraints = read_instance(filename)

    # -----------------------------------------------------
    # Create locks
    # -----------------------------------------------------

    down_locks, up_locks, vars_bin_int, columnLength, root, origObjCoefs = calculate_locks(m)

    # -----------------------------------------------------
    # Degree and Modified Degree
    # -----------------------------------------------------

    degree = [[0, 0] for _ in range(len(m.vars))]
    check_degree = [0 for _ in range(len(m.vars))]

    # -----------------------------------------------------
    # Start Diving Heuristic
    # -----------------------------------------------------
    try:
        status, time, Pass, Trivially, Fo = execute(heuristic, useFPumpObj, m, direction, down_locks, up_locks,
                                                    vars_bin_int,
                                                    columnLength, root, origObjCoefs, degree, check_degree, constraints)
    except Exception as e:
        print(traceback.format_exc())
        exit()

    text = ''
    if useFPumpObj == 1:
        text = '_fpump'

    if status == -1:
        if not Trivially:
            result = '{};{}{};{};{};{};{};{}\n'.format(filename, heuristics[heuristic], text, Pass, round(time, 2),
                                                       objvalue(m, origObjCoefs), heuristic, useFPumpObj)
        else:
            result = '{};{}{};{};{};{};{};{}\n'.format(filename, heuristics[heuristic], text, Pass, round(time, 2),
                                                       Fo, heuristic, useFPumpObj)
    elif status == 0:
        result = '{};{}{};{};{};TimeLimit;{};{}\n'.format(filename, heuristics[heuristic], text, Pass, round(time, 2),
                                                          heuristic, useFPumpObj)
    else:
        result = '{};{}{};{};{};Infeasible;{};{}\n'.format(filename, heuristics[heuristic], text, Pass, round(time, 2),
                                                           heuristic, useFPumpObj)

    print("\n=== FINAL RESULT ===\n", result)

