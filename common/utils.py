class Utils:
    @staticmethod
    def print_tree(tree, level=0):
        if isinstance(tree, dict):
            for k, v in tree.items():
                if isinstance(v, dict):
                    print(' ' * level * 4 + f'[-] {k}: ')
                    Utils.print_tree(v, level + 1)
                else:
                    print(' ' * level * 4 + f'[+] {k}: {v}')
        else:
            print(' ' * level * 4 + f'* {tree}')