import unittest

def load_strings(input_fn):
    res = []
    with open(input_fn) as f:
        for line in f:
            ss = line.split(' ')
            if len(ss) != 6:
                raise IOError('Error reading GW20-style annotations file.')
            res.append(ss[-1].rstrip())
    return res


class TestLoad(unittest.TestCase):
    def test_loadgw20(self):
        a = load_strings('../GW20/GW10_firsthalf.txt')
        b = load_strings('../GW20/GW10_secondhalf.txt')
        self.assertEqual(len(a), 2404)
        self.assertEqual(len(b), 2456)

if __name__=='__main__':
    unittest.main()
