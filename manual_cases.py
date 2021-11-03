from condition import Condition

no_exception =  {
    'own_rate': -1.0,
    'pair_rate': -1.0,
    'samples': [0.0, 0.0]
}

manual_cases = [
    #High strategic flexibility (SF > 6.25) with high technological turbulence (TT > 3) [20 samples]
    {
    'label': 'High',
    'category': True,
    'conditions': [Condition('SF', '>', 6.25), Condition('TT', '>', 3.0)],
    'tree_id': 1,
    'pair_rate': 0,
    'exception': no_exception
    },
    #Highly entrepreneurial organization [EO > 5.3] with high competitor turbulence (COT > 3.1) and high customer turbulence (CUT > 5.1)
    #...
    #negatively influences organizational growth (the var we predict) unless the alignment is high (AL > 5.8)
    {
        'label': 'Low',
        'category': False,
        'conditions': [Condition('EO', '>', 5.3), Condition('COT', '>', 3.1), Condition('CUT', '>', 5.1)],
        'tree_id': 2,
        'pair_rate': 0,
        'exception': {
            'label': 'High',
            'category': True,
            'conditions': [Condition('AL', '>', 5.8), Condition('EO', '>', 5.3), Condition('COT', '>', 3.1), Condition('CUT', '>', 5.1)],
            'tree_id': 2,
            'pair_rate': 0,
            'exception': no_exception
        }
    },
    #Lower entrepreneurial orientation (EO < 4.7)
    #negatively influences organizational growth (the var we predict) unless the alignment is high (AL > 5.8)
    {
        'label': 'Low',
        'category': False,
        'conditions': [Condition('EO', '<', 4.7)],
        'tree_id': 2,
        'pair_rate': 0,
        'exception': {
            'label': 'High',
            'category': True,
            'conditions': [Condition('AL', '>', 5.8), Condition('EO', '<', 4.7)],
            'tree_id': 2,
            'pair_rate': 0,
            'exception': no_exception
        }
    },
    #High customer turbulence (CUT > 5.8) and high technological turbulence
    {
        'label': 'High',
        'category': True,
        'conditions': [Condition('CUT', '>', 5.8), Condition('TT', '>', 6.5)],
        'tree_id': 3,
        'pair_rate': 0,
        'exception': no_exception
    },
    #High customer turbulence (CUT > 5.8) and high alignment (AL > 4.8)
    {
        'label': 'High',
        'category': True,
        'conditions': [Condition('CUT', '>', 5.8), Condition('AL', '>', 4.8)],
        'tree_id': 3,
        'pair_rate': 0,
        'exception': no_exception
    },
    #High adaptability (AD > 4.3) and high strategic flexibility (SF > 4.9) unless competitor turbulence is too high (COT > 3.6)
    {
        'label': 'High',
        'category': True,
        'conditions': [Condition('AD', '>', 4.3), Condition('SF', '>', 4.9)],
        'tree_id': 3,
        'pair_rate': 0,
        'exception': {
            'label': 'Low',
            'category': False,
            'conditions': [Condition('COT', '>', 3.6), Condition('AD', '>', 4.3), Condition('SF', '>', 4.9)],
            'tree_id': 3,
            'pair_rate': 0,
            'exception': no_exception
        }
    },
    #High customer turbulence (CUT > 5.8) and very high alignment (AL > 6.8)
    {
        'label': 'High',
        'category': True,
        'conditions': [Condition('CUT', '>', 5.8), Condition('AL', '>', 4.8)],
        'tree_id': 4,
        'pair_rate': 0,
        'exception': no_exception
    },
    #High customer turbulence (CUT > 3.1) and lower adaptability (<' 6.8) with lower strategic flexibility (SF <= 4.9)
    # negatively influences organizational growth (the var we predict)
    {
        'label': 'Low',
        'category': False,
        'conditions': [Condition('CUT', '>', 3.1), Condition('AD', '<', 6.8), Condition('SF', '<', 4.9)],
        'tree_id': 4,
        'pair_rate': 0,
        'exception': no_exception
    },
    #High customer turbulence (CUT > 3.1) and lower adaptability (<= 6.8) with higher strategic flexibility (SF > 4.9)
    # positively influences organizational growth (the var we predict) unless the technological turbulence (TT) is too high (> 4)
    {
        'label': 'High',
        'category': True,
        'conditions': [Condition('CUT', '>', 3.1), Condition('AD', '<', 6.8), Condition('SF', '>', 4.9)],
        'tree_id': 4,
        'pair_rate': 0,
        'exception': {
            'label': 'Low',
            'category': False,
            'conditions': [Condition('TT', '>', 4.0), Condition('CUT', '>', 3.1), Condition('AD', '<', 6.8), Condition('SF', '>', 4.9)],
            'tree_id': 4,
            'pair_rate': 0,
            'exception': no_exception
        }
    },
    #Organizations that are not very entrepreneurial (EO <= 5.5) while competitor turbulence is high (COT > 5.1)
    #are not likely to exhibit high growth (the var we predict)
    {
        'label': 'Low',
        'category': False,
        'conditions': [Condition('EO', '<', 5.5), Condition('COT', '>', 5.1)],
        'tree_id': 5,
        'pair_rate': 0,
        'exception': no_exception
    },
    #The growth is likely with competitor turbulence being lower (COT <=5.1)
    {
        'label': 'High',
        'category': True,
        'conditions': [Condition('COT', '<', 5.1)],
        'tree_id': 5,
        'pair_rate': 0,
        'exception': no_exception
    }
]