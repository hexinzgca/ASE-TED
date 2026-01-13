import argparse
from copy import deepcopy
from pprint import pprint
import numpy as np


def lammps_real_to_ase(energy_lmp, forces_lmp):
    """LAMMPS real单位 转 ASE默认单位（eV, eV/Å）"""
    energy_conversion = 0.0433641
    force_conversion = 0.0433641
    energy_ase = energy_lmp * energy_conversion
    if isinstance(forces_lmp, np.ndarray):
        forces_ase = forces_lmp * force_conversion
    else:
        forces_ase = forces_lmp
    return energy_ase, forces_ase


def uniq_function(tag, atom_uniqs):
    if tag == 'Bonds':
        sorted_uniqs = sorted([atom_uniqs[0], atom_uniqs[1]])
        bond_uniq = f'{sorted_uniqs[0]}-{sorted_uniqs[1]}'
        return bond_uniq
    if tag == 'Angles':
        sorted_uniqs = sorted([atom_uniqs[0], atom_uniqs[2]])
        angle_uniq = f'{sorted_uniqs[0]}-{atom_uniqs[1]}-{sorted_uniqs[1]}'
        return angle_uniq
    if tag == 'Dihedrals':
        if atom_uniqs[0] < atom_uniqs[3]:
            dihedral_uniq = f'{atom_uniqs[0]}-{atom_uniqs[1]}-{atom_uniqs[2]}-{atom_uniqs[3]}'
        else:
            dihedral_uniq = f'{atom_uniqs[3]}-{atom_uniqs[2]}-{atom_uniqs[1]}-{atom_uniqs[0]}'
        return dihedral_uniq
    if tag == 'Impropers':
        sorted_uniqs = sorted([atom_uniqs[0], atom_uniqs[2], atom_uniqs[3]])
        improper_uniq = f'{sorted_uniqs[0]}-[{atom_uniqs[1]}]-{sorted_uniqs[1]}-{sorted_uniqs[2]}'
        return improper_uniq
    return None

def sym_from_mass(mass):
    mass = float(mass)
    if mass > 0.9 and mass < 1.1: return 'H'
    if mass > 11.9 and mass < 12.1: return 'C'
    if mass > 15.9 and mass < 16.1: return 'O'
    return None

def _guess_element_from_mass(mass, tol=0.5):
    """根据质量猜测元素符号（常见有机元素）"""
    if isinstance(mass, str): mass = float(mass)
    mass_table = {
        'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
        'F': 18.998, 'P': 30.974, 'S': 32.065, 'Cl': 35.453,
    }
    for elem, ref_mass in mass_table.items():
        if abs(mass - ref_mass) < tol:
            return elem
    return None

# @save nxskit.utils.parser.load_lammps_data
def load_lammps_data_0(data_block: str):
    data = {
        'Infos': {}, 'Atom Coeffs': {'Masses': [], 'Pair Coeffs': []},
        'Bond Coeffs': [], 'Angle Coeffs': [], 'Dihedral Coeffs': [], 'Improper Coeffs': [],
        'Atoms': [], 'Bonds': [], 'Angles': [], 'Dihedrals': [], 'Impropers': [],
    }
    info_set = [
            'atoms', 'bonds', 'angles', 'dihedrals', 'impropers',
            'atom types', 'bond types', 'angle types', 'dihedral types', 'improper types',
            'xlo xhi', 'ylo yhi', 'zlo zhi',
    ]
    lines = data_block.splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].strip() or lines[i].strip().startswith('#'): 
            i += 1; continue
        
        parts = lines[i].split('#')[0].strip().split(); k = ''
        if len(parts) == 2: k, v = parts[1], parts[0]
        if len(parts) == 3: k, v = ' '.join(parts[1:]), parts[0]
        if len(parts) == 4: k, v = ' '.join(parts[2:]), ' '.join(parts[:2])
        # if k!='': print(k)
        if k in info_set: 
            data['Infos'][k] = v; i += 1; continue

        for tag in ['Masses', 'Pair Coeffs', 'Bond Coeffs', 'Angle Coeffs', 'Dihedral Coeffs', 'Improper Coeffs']:
            if lines[i].startswith(tag):
                i += 1
                data_ref = data['Atom Coeffs'] if tag in ['Masses', 'Pair Coeffs'] else data
                while lines[i].strip() == '' or lines[i].startswith('#'): i += 1
                while  i < len(lines) and lines[i].strip() != '':
                    parts = lines[i].strip().split('#')[0].split()
                    coeff_index = int(parts[0])
                    coeff_value = ' '.join(parts[1:])
                    data_ref[tag].append({
                        'index': coeff_index,
                        'value': coeff_value,
                        'uniq': '' if tag != 'Masses' else f'[{_guess_element_from_mass(coeff_value)}]',
                    })
                    i += 1

        if lines[i].startswith('Atoms'):
            i += 1
            sym = []; xyz = []; chg = []; atom_index = []
            while lines[i].strip() == '' or lines[i].startswith('#'): i += 1
            while lines[i].strip() != '' and i < len(lines):
                parts = lines[i].split('#')[0].strip().split()
                term = {
                    'index': int(parts[0]), 'mol': int(parts[1]), 'type': int(parts[2]),
                    'q': float(parts[3]), 'x': float(parts[4]), 'y': float(parts[5]), 'z': float(parts[6])
                }
                atom_index += [term['index']]
                sym += [_guess_element_from_mass(data['Atom Coeffs']['Masses'][term['type']-1]['value'])]
                xyz += [[term['x'], term['y'], term['z']]]
                chg += [term['q']]
                data['Atoms'].append(term)
                i += 1
            data['sym'] = np.array(sym)
            data['xyz'] = np.array(xyz)
            data['chg'] = np.array(chg)
            data['atom_index'] = np.array(atom_index)

        for tag in ['Bonds', 'Angles', 'Dihedrals', 'Impropers']:
            if i < len(lines) and lines[i].startswith(tag):
                i += 1
                while lines[i].strip() == '' or lines[i].startswith('#'): i += 1
                while i < len(lines) and lines[i].strip() != '':
                    parts = lines[i].split('#')[0].strip().split()
                    term = {
                        'index': int(parts[0]),
                        'type': int(parts[1]),
                        'atom_index': np.array([int(k) for k in parts[2:]])
                    }
                    data[tag].append(term)
                    i += 1
        i += 1
    return data


def dump_lammps_data_0(data, comment = False):
    block = 'LAMMPS Atom File\n\n'
    for info in ['atoms', 'bonds', 'angles', 'dihedrals', 'impropers', '', 'atom types', 'bond types', 'angle types', 'dihedral types', 'improper types', '', 'xlo xhi', 'ylo yhi', 'zlo zhi']:
        if info == '': block += '\n'; continue
        block += f"{data['Infos'][info]} {info}\n"
    block += '\n'
    for tag in ['Masses', 'Pair Coeffs', 'Bond Coeffs', 'Angle Coeffs', 'Dihedral Coeffs', 'Improper Coeffs']:
        block += f"{tag}\n\n"
        data_ref = data[tag] if tag not in ['Masses', 'Pair Coeffs'] else data['Atom Coeffs'][tag]
        for i, term in enumerate(data_ref):
            if comment:
                block += f"{i+1:<6d} {term['value']} # {term['uniq']}\n"
            else:
                block += f"{i+1:<6d} {term['value']}\n"
        if f'{tag} Append' in data:
            for i, term in enumerate(data[f'{tag} Append']):
                block += f"{i+1:<6d} {term['value']}\n"
        block += '\n'
    for tag in ['Atoms']:
        block += f"{tag}\n"
        block += f"#  atom  mol    type   charge         X           Y           Z\n"
        for i, term in enumerate(data[tag]):
            block += f"{i+1:<6d} {term['mol']:<6d} {term['type']:<6d} {term['q']:15.8f} {term['x']:15.8f} {term['y']:15.8f} {term['z']:15.8f}\n"
        block += '\n'
    for tag in ['Bonds', 'Angles', 'Dihedrals', 'Impropers']:
        block += f"{tag}\n"
        block += f"# ..............................\n"
        for i, term in enumerate(data[tag]):
            block += f"{i+1:<6d} {term['type']:<6d} {' '.join(map(str, term['atom_index']))}\n"
        block += '\n'
    return block


def update_lammps_data(data, update_atom_index: bool = False, make_uniq: bool = False,
    update_info: bool = False, update_box: bool = False, neutralize: bool = False):
    if update_atom_index: # only update atom index (not bond, angle, dihedral, improper!)
        data['type'] = np.array([term['type'] for term in data['Atoms']])
        data['mol'] = np.array([term['mol'] for term in data['Atoms']])
        index_tmp = np.array([term['index'] for term in data['Atoms']])

        #print("======")
        #print(data['atom_index'][:10])
        #print(index_tmp[:10])
        
        index = np.argsort(index_tmp)
        data['atom_index'] = index_tmp[index]

        #print(data['atom_index'][:10])

        data['xyz'] = data['xyz'][index]
        data['chg'] = data['chg'][index]
        data['type'] = data['type'][index]
        data['mol'] = data['mol'][index]
        data['sym'] = data['sym'][index]
        # data_back = deepcopy(data)
        for i, item in enumerate(data['Atoms']):
            item['index'] = i + 1
            item['q'] = data['chg'][i]
            item['x'], item['y'], item['z'] = data['xyz'][i][:]
            item['type'] = data['type'][i]
            item['mol'] = data['mol'][i]

    if neutralize:
        data['chg'] -= np.sum(data['chg']) / len(data['chg'])
        for i, item in enumerate(data['Atoms']):
            item['q'] = data['chg'][i]

    if True:
        data['Atom Coeffs Uniq'] = []
        data['Atom Coeffs Map'] = {}
        for i, term_mass in enumerate(data['Atom Coeffs']['Masses']):
            term_pair = data['Atom Coeffs']['Pair Coeffs'][i]
            term_mass['uniq'] += ';'.join([term_mass['value'], term_pair['value']])
            term_pair['uniq'] = term_mass['uniq']
            if term_mass['uniq'] not in data['Atom Coeffs Uniq']:
                data['Atom Coeffs Uniq'].append(term_mass['uniq'])
                data['Atom Coeffs Map'][i+1] = len(data['Atom Coeffs Uniq'])
            else:
                uniq_idx = data['Atom Coeffs Uniq'].index(term_mass['uniq'])
                data['Atom Coeffs Map'][i+1] = uniq_idx + 1

        for tag in ['Bonds', 'Angles', 'Dihedrals', 'Impropers']:
            coeff_tag = tag[:-1] + ' Coeffs'
            data[coeff_tag + ' Uniq'] = []
            data[coeff_tag + ' Map'] = {}
            for i, term in enumerate(data[tag]):
                tag_type = term['type']
                tag_index = term['atom_index']
                tag_atype = [data['Atoms'][i-1]['type'] for i in tag_index]
                tag_auniq = [data['Atom Coeffs']['Masses'][i-1]['uniq'] for i in tag_atype]
                tag_uniq1 = uniq_function(tag, tag_auniq)
                tag_uniq2 = data[coeff_tag][tag_type-1]['value']
                tag_uniqs = '%'.join([tag_uniq1, tag_uniq2])
                data[tag][i]['uniq'] = tag_uniqs
                if tag_uniq1 not in data[coeff_tag][tag_type-1]['uniq'].split('@'):
                    data[coeff_tag][tag_type-1]['uniq'] += f'@{tag_uniq1}'

                if tag_uniqs not in data[coeff_tag + ' Uniq']:
                    data[coeff_tag + ' Uniq'].append(tag_uniqs)
                    data[coeff_tag + ' Map'][tag_type] = len(data[coeff_tag + ' Uniq'])
                else:
                    uniq_idx = data[coeff_tag + ' Uniq'].index(tag_uniqs)
                    data[coeff_tag + ' Map'][tag_type] = uniq_idx + 1

    if make_uniq:
        update_info = True # info will be updated
        for i, term in enumerate(data['Atoms']):
            a_type = term['type']
            term['type'] = data['Atom Coeffs Map'][a_type]

        for tag in ['Bonds', 'Angles', 'Dihedrals', 'Impropers']:
            coeff_tag = tag[:-1] + ' Coeffs'
            for i, term in enumerate(data[tag]):
                tag_type = term['type']
                term['type'] = data[coeff_tag + ' Map'][tag_type]
        
        data['Atom Coeffs Back'] = deepcopy(data['Atom Coeffs'])
        data['Atom Coeffs']['Masses'] = []; data['Atom Coeffs']['Pair Coeffs'] = []
        for i, term in enumerate(data['Atom Coeffs Uniq']):
            data['Atom Coeffs']['Masses'].append({
                'index': i+1, 'value': term.split(';')[0], 'uniq': term
                })
            data['Atom Coeffs']['Pair Coeffs'].append({
                'index': i+1, 'value': term.split(';')[1], 'uniq': term
                })
        for tag in ['Bonds', 'Angles', 'Dihedrals', 'Impropers']:
            coeff_tag = tag[:-1] + ' Coeffs'
            coeff_back_tag = coeff_tag + ' Back'
            data[coeff_back_tag] = deepcopy(data[coeff_tag])
            data[coeff_tag] = []
            for i, term in enumerate(data[coeff_tag + ' Uniq']):
                a_uniqs, t_uniq = term.split('%')[0], term.split('%')[1]
                a_uniqs = a_uniqs.split('-')
                data[coeff_tag].append({
                    'index': i+1, 'value': t_uniq, 'uniq': term
                })
    if update_info:
        data['Infos']['atoms'] = len(data['Atoms'])
        data['Infos']['bonds'] = len(data['Bonds'])
        data['Infos']['angles'] = len(data['Angles'])
        data['Infos']['dihedrals'] = len(data['Dihedrals'])
        data['Infos']['impropers'] = len(data['Impropers'])
        data['Infos']['atom types'] = len(data['Atom Coeffs']['Masses'])
        data['Infos']['bond types'] = len(data['Bond Coeffs'])
        data['Infos']['angle types'] = len(data['Angle Coeffs'])
        data['Infos']['dihedral types'] = len(data['Dihedral Coeffs'])
        data['Infos']['improper types'] = len(data['Improper Coeffs'])
    if update_box:
        xmin = np.min(data['xyz'][:, 0]); xmax = np.max(data['xyz'][:, 0])
        ymin = np.min(data['xyz'][:, 1]); ymax = np.max(data['xyz'][:, 1])
        zmin = np.min(data['xyz'][:, 2]); zmax = np.max(data['xyz'][:, 2])
        data['Infos']['xlo xhi'] = f"{xmin-5:.6f} {xmax+5:.6f}"
        data['Infos']['ylo yhi'] = f"{ymin-5:.6f} {ymax+5:.6f}"
        data['Infos']['zlo zhi'] = f"{zmin-5:.6f} {zmax+5:.6f}"
    return data

arg = argparse.ArgumentParser()
arg.description = 'Update LAMMPS data file'
arg.add_argument('--input', '-i', type=str, required=True)
arg.add_argument('--output', '-o', type=str, required=True)
arg.add_argument('--update_info', '-u', type=bool, default=False)
arg.add_argument('--update_atom_index', '-a', type=bool, default=False)
arg.add_argument('--update_box', '-b', type=bool, default=False)
arg.add_argument('--neutralize', '-n', type=bool, default=False)
arg.add_argument('--comment', '-c', type=bool, default=False)
arg.add_argument('--make_uniq', '-m', type=bool, default=False)

if __name__ == '__main__':
    args = arg.parse_args()
    with open(args.input, 'r') as f:
        data = load_lammps_data_0(f.read())
    print('Before update:')
    pprint(data['Infos'])
    data = update_lammps_data(data, 
    update_atom_index=args.update_atom_index, update_info=args.update_info, update_box=args.update_box, 
    neutralize=args.neutralize, make_uniq=args.make_uniq)
    print('After update:')
    pprint(data['Infos'])
    with open(args.output, 'w') as f:
        f.write(dump_lammps_data_0(data, comment = args.comment))


# 补充：解析LAMMPS data文件的拓扑信息（复用你之前的拓扑解析逻辑）
def _parse_lammps_data_topology(self, data_file):
    """
    从LAMMPS data文件解析拓扑信息（Bonds/Angles/Dihedrals/Impropers）
    返回：拓扑字典
    """
    topology = {'bonds': [], 'angles': [], 'dihedrals': [], 'impropers': []}
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    current_section = None
    for line in lines:
        # 识别拓扑区块
        if line.startswith('Bonds'):
            current_section = 'bonds'
            continue
        elif line.startswith('Angles'):
            current_section = 'angles'
            continue
        elif line.startswith('Dihedrals'):
            current_section = 'dihedrals'
            continue
        elif line.startswith('Impropers'):
            current_section = 'impropers'
            continue
        # 非拓扑区块，停止解析
        elif line.startswith('Velocities') or line.startswith('Masses') or line.startswith('Atoms') and current_section is not None:
            current_section = None
            continue

        # 解析拓扑数据
        if current_section is not None and line.strip() and not line.startswith('#'):
            parts = line.split()
            if current_section == 'bonds' and len(parts) >= 4:
                # Bonds格式：bond_id bond_type atom1_id atom2_id
                bond_type = int(parts[1])
                atom1_id = int(parts[2])
                atom2_id = int(parts[3])
                topology['bonds'].append((bond_type, atom1_id, atom2_id))
            elif current_section == 'angles' and len(parts) >= 5:
                # Angles格式：angle_id angle_type atom1_id atom2_id atom3_id
                angle_type = int(parts[1])
                atom1_id = int(parts[2])
                atom2_id = int(parts[3])
                atom3_id = int(parts[4])
                topology['angles'].append((angle_type, atom1_id, atom2_id, atom3_id))
            elif current_section == 'dihedrals' and len(parts) >= 6:
                # Dihedrals格式：dihedral_id dihedral_type a1 a2 a3 a4
                dihedral_type = int(parts[1])
                a1, a2, a3, a4 = map(int, parts[2:6])
                topology['dihedrals'].append((dihedral_type, a1, a2, a3, a4))
            elif current_section == 'impropers' and len(parts) >= 6:
                # Impropers格式：improper_id improper_type a1 a2 a3 a4
                improper_type = int(parts[1])
                a1, a2, a3, a4 = map(int, parts[2:6])
                topology['impropers'].append((improper_type, a1, a2, a3, a4))
    return topology

def parse_lammps_data_to_ase_atoms(data):
    """
    从lammps.data文件解析原子坐标、盒子信息、原子类型，构建ase.Atoms对象
    参数：
        data_file: lammps.data文件路径
        atom_type_map: 原子类型->元素符号的映射（若为None，自动从Masses部分解析）
    返回：
        ase.Atoms: 包含坐标、盒子、元素信息的原子对象
    """
    from ase import Atoms
    # 1. 构建盒子尺寸
    if 'xlo xhi' in data['Infos'] and 'ylo yhi' in data['Infos'] and 'zlo zhi' in data['Infos']:
        xmin, xmax = map(float, data['Infos']['xlo xhi'].split())
        ymin, ymax = map(float, data['Infos']['ylo yhi'].split())
        zmin, zmax = map(float, data['Infos']['zlo zhi'].split())
        cell = np.diag([xmax - xmin, ymax - ymin, zmax - zmin])
    else:
        xmin = data['xyz'][:,0].min(); xmax = data['xyz'][:,0].max()
        ymin = data['xyz'][:,1].min(); ymax = data['xyz'][:,1].max()
        zmin = data['xyz'][:,2].min(); zmax = data['xyz'][:,2].max()
        cell = np.diag([xmax - xmin + 10, ymax - ymin + 10, zmax - zmin + 10])
        print("Warning: no PBC information, but put molecule into a box!")
    # 2. 解析原子坐标（Atoms部分）
    if 'xyz' in data and data['xyz'].shape[0] > 0:
        atom_coords = np.copy(data['xyz'])
    else:
        raise RuntimeError("未从lammps.data解析到有效原子信息")
    # 3. 解析原子符号
    atom_elems = []
    for i, term in enumerate(data['Atoms']):
        atom_type = term['type']
        atom_mass = float(data['Atom Coeffs']['Masses'][atom_type-1]['value'])
        atom_elems += [_guess_element_from_mass(atom_mass)]
    # 4. 构建ASE Atoms对象
    return Atoms(
        symbols=atom_elems,
        positions=np.array(atom_coords),
        cell=cell,
        pbc=True
    )

