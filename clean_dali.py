
import csv

with open('DALI_v2.0/dali_remarks.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=';')
    header = ['id', 'url', 'PCO', 'AAE', 'corrupt from', 'noisy', 'offset', 'non-english', 'vocalizations',
              'repeated words/lines', 'multiple singers', 'split words', 'missing words']
    
    writer.writerow(header)

    writer.writerow(['1b9c139f491c41f5b0776eefd21c122d', 'o61rTPowBq0', -1, -1, 0, False, 0, False, False, False, False, False, False])
    # was removing this song when working on txt folder (get_dali() removed all non-monotonic and this one)

    for i in range(3, 9, 5):  # i = 3, 8
        
        metadata = []
        with open(f'txt/dali_0{i}_wrong_words.txt', 'r') as wrong_words:
            lines = wrong_words.readlines()
            for j in range(len(lines) - 1):
                if lines[j].startswith('id: '):
                    id_line = lines[j].strip('\n').strip(' ')
                    PCO_line = lines[j + 1].strip('\n').strip(' ')
                    metadata.append([id_line[4:36], id_line[42:], PCO_line[5:11], PCO_line[26:]])

        with open(f'txt/dali_0{i}_remarks.txt', 'r') as remarks:
            lines = remarks.readlines()
            j = 0
            row = []
            for line in lines:
                
                if line.startswith('id'):
                    assert metadata[j][0] == line[4:36], f'{metadata[j][0]}, {line[4:36]}'
                    row = metadata[j] + [1e10, False, 0, False, False, False, False, False, False]
                    continue
                elif line.startswith('\n'):
                    writer.writerow(row)
                    j += 1
                elif line.startswith('corrupt'):
                    row[4] = 0
                elif line.startswith('wrong end from '):
                    row[4] = line[15:].strip('\n')
                elif line.startswith('noisy'):
                    row[5] = True
                elif line.startswith('offset: '):
                    row[6] = line[8:].strip('\n')
                elif line.startswith('+') or line.startswith('-'):
                    row[6] = line[:-1]
                elif line.startswith('non-english'):
                    row[7] = True
                elif line.startswith('vocalizations'):
                    row[8] = True
                elif line.startswith('repeated '):
                    row[9] = True
                elif line.startswith('multiple singers'):
                    row[10] = True
                elif line.startswith('split words'):
                    row[11] = True
                elif line.startswith('missing words'):
                    row[12] = True
                else:
                    print(line)

