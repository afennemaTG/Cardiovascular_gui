
def pre_sets(name):

    if name == 'normal':
        preset = {
            'RPM': 0, 
            'contractility': 1, 
            'SVR': 1, 
            'compliance': 1,
            'HR': 70
        }

    elif name == 'cardiogenic shock':
        preset = {
            'RPM': 3000,
            'contractility': 0.25,
            'SVR': 1.3,
            'compliance': 1,
            'HR': 70
        }

    elif name == 'septic shock':
        preset = {
            'RPM': 3000,
            'contractility': 0.5,
            'SVR': 0.5,
            'compliance': 1,
            'HR': 90
        }

    elif name == 'case 1':
        preset = {
            'RPM': 2800,
            'contractility': 0.5,
            'SVR': 1,
            'compliance': 1.5,
            'HR': 95,
            'baroreceptor': True
        }

    elif name == 'case 2':
        preset = {
            'RPM': 3000,
            'contractility': 0.25,
            'SVR': 1.2,
            'compliance': 0.6,
            'HR': 65,
            'baroreceptor': False
        }
        
    return preset