import os

def main():
    files = os.listdir('../experiments/training_data/')
    files.sort()
    for file in files:
        with open('../experiments/training_data/' + file, 'r') as f:
            max = 0
            for line in f:
                accuracy = float(line.split(',')[2])
                if accuracy > max:
                    max = accuracy
        max *= 100
        file = " ".join(file.strip('.txt').split('_'))
        print(f'{file} : {max:.2f}%')

if __name__ == '__main__': main()
