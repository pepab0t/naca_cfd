
def main():
    DEGREES = [x/10 for x in range(-100, 101, 5)]

    # print(len(DEGREES))
    with open('profiles_list', 'w') as f:
        for x in DEGREES:
            for p in ['0012', '0015', '2412']:
                f.write(f'{p} {x}\n')

if __name__ == '__main__':
    main()
